#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external/")
sys.path.append('/ws/external/ai_module/src/exploration/scripts')

import rospy
import numpy as np
import math
import heapq
import cv2
import tf2_ros
import tf2_geometry_msgs
import time

from typing import List

from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from python_tsp.exact import solve_tsp_dynamic_programming
from multi_goal_planner import MultiGoalPlanner
from visualize_utils import renew_marker_array
from ai_module.src.visual_grounding.scripts.utils.utils_point import make_marker_array_from_points
from exploration.srv import AstarPath, AstarPathRequest

from ai_module.src.utils.logger import Logger

INF = 1e9


class AStarPlanner:
    def __init__(self, grid_map, resolution):
        """
        Initializes the A* planner.
        Input:
            grid_map (np.ndarray): The 2D occupancy grid.
            resolution (float): The resolution of the map in meters/pixel.
        Output:
            None
        Parameter description:
            self.grid_map: The occupancy grid map.
            self.resolution: The map resolution.
            self.height, self.width: Dimensions of the grid map.
            self.neighbors: 8-connectivity neighborhood for searching.
        """
        quiet = rospy.get_param('~quiet', False)
        self.quiet = quiet
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def _heuristic(self, a, b):
        """
        Calculates the Euclidean distance heuristic between two points.
        Input:
            a (tuple): The first point (x, y).
            b (tuple): The second point (x, y).
        Output:
            float: The Euclidean distance between a and b.
        Parameter description:
            None
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def plan(self, start_pixel, end_pixel):
        """
        Plans a path from a start to an end pixel using the A* algorithm.
        Input:
            start_pixel (tuple): The starting (x, y) pixel coordinates.
            end_pixel (tuple): The ending (x, y) pixel coordinates.
        Output:
            (List[tuple], float): A tuple containing the path as a list of pixels and the total path length in meters.
                                  Returns (None, float('inf')) if no path is found.
        Parameter description:
            None
        """
        if not (0 <= start_pixel[0] < self.width and 0 <= start_pixel[1] < self.height and
                0 <= end_pixel[0] < self.width and 0 <= end_pixel[1] < self.height):
            if not self.quiet:
                rospy.logwarn(f"A* Planner: Start or end pixel is out of bounds. Start: {start_pixel}, End: {end_pixel}")
            return None, float('inf')
        if self.grid_map[start_pixel[1], start_pixel[0]] == 100 or \
           self.grid_map[end_pixel[1], end_pixel[0]] == 100:
            if not self.quiet:
                rospy.logwarn("A* Planner: Start or end point is on an obstacle.")
            return None, float('inf')

        open_set = []
        heapq.heappush(open_set, (0, start_pixel))
        came_from = {}
        g_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
        g_score[start_pixel] = 0
        f_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
        f_score[start_pixel] = self._heuristic(start_pixel, end_pixel)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_pixel:
                return self._reconstruct_path(came_from, current)
            for dx, dy in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if self.grid_map[neighbor[1], neighbor[0]] == 100:
                    continue
                move_cost = math.hypot(dx, dy)
                tentative_g_score = g_score[current] + move_cost
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_pixel)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        """
        Reconstructs the final path from the came_from map.
        Input:
            came_from (dict): A dictionary mapping each node to its predecessor on the path.
            current (tuple): The goal node from which to start backtracking.
        Output:
            (List[tuple], float): A tuple containing the reconstructed path and its length in meters.
        Parameter description:
            None
        """
        path_length = 0.0
        path_pixels = [current]
        while current in came_from:
            prev = came_from[current]
            dist = math.hypot(current[0] - prev[0], current[1] - prev[1])
            path_length += dist
            current = prev
            path_pixels.append(current)
        path_pixels.reverse()
        return path_pixels, path_length * self.resolution

class WaypointPathPlanner:
    def __init__(self):
        """
        Initializes the WaypointPathPlanner ROS node.
        Input:
            None
        Output:
            None
        Parameter description:
            ~map_frame (str): The target frame for all coordinates, typically 'map'.
            ~local_grid_resolution (float): The resolution for the dynamically generated local grid.
            ~inflate_radius (float): The radius in meters to inflate obstacles in the local grid.
        """
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='PathFollower', log_path="/ws/external/log/exploration/multi_view.log")

        self.logger.loginfo("Initializing Waypoint Path Planner...")

        self.is_real_world = rospy.get_param('~real_world', False)
        if self.is_real_world:
            self.logger.loginfo("Hello Real World!!")
            self.frame_id = "world"
        else:
            self.frame_id = "map"

        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.local_grid_resolution = rospy.get_param('~local_grid_resolution', 0.05)
        self.inflate_radius_m = rospy.get_param('~inflate_radius', 0.0)

        self.local_grid_info = {}
        self.waypoints = []
        self.current_pose = None
        self.v_x, self.v_y = None, None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tsp_path_pub = rospy.Publisher("/waypoint_tsp_path", Path, queue_size=2)
        self.local_map_pub = rospy.Publisher("/local_traversable_map", OccupancyGrid, queue_size=2)
        self.unreachable_nodes_pub = rospy.Publisher("/unreachable_nodes", MarkerArray, queue_size=2)

        # Add A* service client like object_navigation.py and coverage_path.py
        self.logger.loginfo("Waiting for A* path service...")
        rospy.wait_for_service('/astar_path')
        self.astar_path_service = rospy.ServiceProxy('/astar_path', AstarPath)
        self.logger.loginfo("A* path service found.")

        self.traversable_area_ready = False
        self.prev_traversable_points_count = 0
        self.points_sub = rospy.Subscriber("/traversable_area_filtered", PointCloud2, self._pointcloud_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=1)

        self.prev_num_waypoints = 0
        self.waypoints_is_updated = False
        self.active_waypoints_sub = rospy.Subscriber("/active_waypoints", MarkerArray, self._active_waypoints_callback, queue_size=10)

        self.running = False
        self.mean_plan_delay = 1.0
        self.timer = rospy.Timer(rospy.Duration(0.2), self._timer_callback)

        # Visualize
        self.target_waypoints_pub = rospy.Publisher("/multi_view/waypoints", MarkerArray, queue_size=10)
        
        self.logger.loginfo("Node ready. Waiting for topics: '/state_estimation', '/traversable_area', '/frontier_markers'...")

    def _active_waypoints_callback(self, msg):
        waypoints = []
        for m in msg.markers:
            waypoint_coords = np.array([m.pose.position.x, m.pose.position.y, 0.0])
            waypoints.append(waypoint_coords)
        self.logger.loginfo(f"Received {len(waypoints)} active waypoints.")

        # Visualize
        filtered_marker = make_marker_array_from_points(
            waypoints, ns="waypoints", color=(1.0, 0.5, 0.5, 0.5), frame_id=self.frame_id)
        renew_marker_array(self.target_waypoints_pub, filtered_marker)

        # Update waypoints
        self.waypoints = waypoints
        self.prev_num_waypoints = len(waypoints)
        self.waypoints_is_updated = True

    def _odom_callback(self, msg: Odometry):
        """
        Callback to process Odometry messages and update the robot's current pose.
        Input:
            msg (nav_msgs.msg.Odometry): The incoming odometry message.
        Output:
            None
        Parameter description:
            self.current_pose: Stores the latest pose (geometry_msgs/Pose) of the robot.
            self.v_x, self.v_y: Linear velocity
        """
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        self.current_pose = msg.pose.pose
        self.v_x, self.v_y = v_x, v_y

    def _timer_callback(self, event):
        if self.running:
            return
        self.running = True

        if self.traversable_area_ready:
            self._plan_path()

        self.running = False

    def _pointcloud_callback(self, msg: PointCloud2):
        """
        Callback to process a traversable area point cloud and generate a local occupancy grid.
        Input:
            msg (sensor_msgs.msg.PointCloud2): Point cloud of the traversable area.
        Output:
            None
        Parameter description:
            self.local_grid_info: A dictionary containing the generated grid, its metadata (origin, resolution, etc.), and the original message header.
        """
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        current_points_count = len(points_list)

        # Check if the number of traversable points has changed
        if current_points_count == self.prev_traversable_points_count:
            return  # No change in traversable points, skip update

        if current_points_count < 10:
            self.logger.logwarn("Point cloud has too few points to generate a map.")
            return

        self.logger.loginfo(f"Traversable points changed from {self.prev_traversable_points_count} to {current_points_count}. Generating local map...")
        
        self.traversable_area_ready = False
        points_2d = np.array([p[:2] for p in points_list])
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)

        padding = 2.0
        local_origin = (min_x - padding, min_y - padding)
        local_width = int((max_x - min_x + 2 * padding) / self.local_grid_resolution)
        local_height = int((max_y - min_y + 2 * padding) / self.local_grid_resolution)

        local_grid = np.full((local_height, local_width), 100, dtype=np.uint8)

        for point in points_2d:
            px = int((point[0] - local_origin[0]) / self.local_grid_resolution)
            py = int((point[1] - local_origin[1]) / self.local_grid_resolution)
            if 0 <= px < local_width and 0 <= py < local_height:
                local_grid[py, px] = 0

        inflate_radius_px = int(self.inflate_radius_m / self.local_grid_resolution)
        if inflate_radius_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflate_radius_px + 1, 2 * inflate_radius_px + 1))
            inflated_local_grid = cv2.dilate(local_grid, kernel)
        else:
            inflated_local_grid = local_grid

        self.local_grid_info = {
            "grid": inflated_local_grid,
            "origin": local_origin,
            "resolution": self.local_grid_resolution,
            "height": local_height,
            "width": local_width,
            "header": msg.header
        }

        self._publish_local_map()
        self.logger.loginfo("Local map generated and published.")

        # Update the count and set ready flag
        self.prev_traversable_points_count = current_points_count
        self.traversable_area_ready = True

    def _get_astar_path(self, start_x: float, start_y: float, end_x: float, end_y: float) -> (Path, float):
        """
        Get A* path using the external A* service (same as object_navigation.py and coverage_path.py).
        This handles cases where robot position is not on traversable area.
        """
        try:
            req = AstarPathRequest()
            req.start_x = start_x
            req.start_y = start_y
            req.end_x = end_x
            req.end_y = end_y
            resp = self.astar_path_service(req)

            if resp.path_length < 0:
                self.logger.logwarn(f"A* service could not find a path from ({start_x:.2f}, {start_y:.2f}) to ({end_x:.2f}, {end_y:.2f}).")
                return Path(), float('inf')
            
            resp.path.header.stamp = rospy.Time.now()
            resp.path.header.frame_id = self.frame_id

            return resp.path, resp.path_length
        
        except rospy.ServiceException as e:
            self.logger.logerr(f"A* service call failed: {e}")
            return Path(), float('inf')

    def _plan_path(self):
        """
        Main planning function. Calculates a distance matrix using A*, solves the TSP, and publishes the final path.
        Input:
            None
        Output:
            None
        Parameter description:
            None
        """
        if 'grid' not in self.local_grid_info:
            self.logger.logwarn("Map is required before planning. Waiting for '/traversable_area'.")
            return
        if not self.waypoints:
            self.logger.loginfo("No waypoints available. Waiting for '/frontier_markers'.")
            return
        
        if self.current_pose is None:
            self.logger.logwarn("Have not received robot's current pose (Odometry) yet. Check the '/state_estimation' topic.")
            return

        if self.waypoints_is_updated:
            current_position = self.current_pose.position
            start_node = np.array([current_position.x, current_position.y, current_position.z])
            self.logger.loginfo(f"Acquired current robot position (Odom): {start_node[:2]}")

            nodes = [start_node] + self.waypoints
            num_nodes = len(nodes)

            grid = self.local_grid_info["grid"]
            resolution = self.local_grid_info["resolution"]
            origin = self.local_grid_info["origin"]

            # Use A* service for distance matrix calculation instead of internal planner
            distance_matrix = np.full((num_nodes, num_nodes), INF, dtype=np.float16)

            self.logger.loginfo(f"Starting A* service-based distance matrix calculation for {num_nodes} nodes...")
            prev_time = time.time()

            unreachable_pairs = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        distance_matrix[i, j] = 0.0
                    else:
                        start_world = nodes[i]
                        end_world = nodes[j]
                        _, dist = self._get_astar_path(start_world[0], start_world[1], end_world[0], end_world[1])
                        distance_matrix[i, j] = dist if math.isfinite(dist) else INF

                        if dist == INF:
                            unreachable_pairs.append((i, j))

            # Create MarkerArray for unreachable node pairs
            unreachable_markers = MarkerArray()
            for idx, (i, j) in enumerate(unreachable_pairs):
                # Create line marker connecting unreachable nodes
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "unreachable_connections"
                marker.id = idx
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05  # Line width
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                # Add start and end points
                start_point = Point()
                start_point.x = nodes[i][0]
                start_point.y = nodes[i][1]
                start_point.z = 0.5  # Slightly elevated for visibility
                
                end_point = Point()
                end_point.x = nodes[j][0]
                end_point.y = nodes[j][1]
                end_point.z = 0.5
                
                marker.points = [start_point, end_point]
                unreachable_markers.markers.append(marker)
            
            self.logger.loginfo(f"Found {len(unreachable_pairs)} unreachable node pairs")
            self.unreachable_nodes_pub.publish(unreachable_markers)

            distance_matrix[:, 0] = 0.0
            self.logger.loginfo(f" > distance time: {time.time() - prev_time}")
            self.logger.loginfo("Set return cost to 0 for open-loop TSP.")

            self.logger.loginfo("Solving TSP...")
            prev_time = time.time()
            try:
                permutation, total_cost = solve_tsp_dynamic_programming(distance_matrix)
            except Exception as e:
                self.logger.logerr(f"Error occurred while running TSP solver: {e}")
                return
            self.logger.loginfo(f" > tsp time: {time.time() - prev_time}")

            if total_cost >= 1e9:
                self.logger.logwarn("Could not find a TSP path (waypoints may be unreachable).")
                return

            self.logger.loginfo(f"TSP solved! Optimal order: {permutation}, Estimated cost: {total_cost:.2f}")

            final_path = Path(header=self.local_grid_info["header"])
            final_path.header.stamp = rospy.Time.now()
            final_path.header.frame_id = self.frame_id

            path_indices = [p for p in permutation if p != 0]
            path_indices.insert(0, 0)

            self.logger.loginfo(f"Generating detailed path... Order: {path_indices}")

            for i in range(len(path_indices) - 1):
                start_node_idx = path_indices[i]
                end_node_idx = path_indices[i+1]

                start_node_coords = nodes[start_node_idx]
                end_node_coords = nodes[end_node_idx]

                # Use A* service for detailed path generation
                segment_path, _ = self._get_astar_path(
                    start_node_coords[0], start_node_coords[1], end_node_coords[0], end_node_coords[1]
                )
                
                if segment_path.poses:
                    if final_path.poses and segment_path.poses:
                        final_path.poses.extend(segment_path.poses[1:])  # Skip first to avoid duplicates
                    else:
                        final_path.poses.extend(segment_path.poses)

            final_path_list = self.path_to_list(final_path)
            smoothed_path_list = self.smooth_path_moving_average(final_path_list, window_size=15)
            smoothed_path_msg = self.list_to_path(smoothed_path_list, frame_id=final_path.header.frame_id)
            self.tsp_path_pub.publish(smoothed_path_msg)
            self.logger.loginfo("Published the final detailed path to '/waypoint_tsp_path'.")
            self.waypoints_is_updated = False

    def _get_detailed_astar_path(self, start_world, end_world, header) -> List[PoseStamped]:
        """
        Generates a detailed path segment between two world coordinates using A* service.
        Input:
            start_world (tuple): The (x, y) world coordinates of the segment start.
            end_world (tuple): The (x, y) world coordinates of the segment end.
            header (std_msgs.msg.Header): The header to use for the generated poses.
        Output:
            List[PoseStamped]: A list of stamped poses that form the path segment.
        Parameter description:
            None
        """
        # Use A* service instead of internal planner
        segment_path, _ = self._get_astar_path(start_world[0], start_world[1], end_world[0], end_world[1])
        return segment_path.poses if segment_path.poses else []

    def _publish_local_map(self):
        """
        Publishes the generated local occupancy grid.
        Input:
            None
        Output:
            None
        Parameter description:
            None
        """
        if 'grid' not in self.local_grid_info: return
        msg = OccupancyGrid(header=self.local_grid_info["header"])
        msg.header.stamp = rospy.Time.now()
        msg.info.resolution = self.local_grid_info["resolution"]
        msg.info.width = self.local_grid_info["width"]
        msg.info.height = self.local_grid_info["height"]
        msg.info.origin.position.x = self.local_grid_info["origin"][0]
        msg.info.origin.position.y = self.local_grid_info["origin"][1]
        msg.info.origin.orientation.w = 1.0
        msg.data = self.local_grid_info["grid"].flatten().tolist()
        self.local_map_pub.publish(msg)
        
    def path_to_list(self,path_msg):
        return [[p.pose.position.x, p.pose.position.y] for p in path_msg.poses]

    def list_to_path(self,path_list, frame_id="map"):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id
        for x, y in path_list:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        return path_msg

    def smooth_path_moving_average(self,path, window_size=20):
        if window_size < 3:
            return path
            
        path_np = np.array(path)
        smoothed_path = path_np.copy()
        w = window_size // 2
        
        for i in range(w, len(path_np) - w):
            smoothed_path[i] = np.mean(path_np[i-w:i+w+1], axis=0)
            
        return smoothed_path.tolist()

if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_path_planner', anonymous=True)
        WaypointPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
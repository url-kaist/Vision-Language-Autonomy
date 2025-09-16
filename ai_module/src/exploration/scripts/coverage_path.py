#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import cv2

from scipy.spatial import ConvexHull, cKDTree
from typing import Tuple, List
from scipy.interpolate import splprep, splev 
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Header, String
from sklearn.cluster import DBSCAN
from exploration.srv import AstarPath, AstarPathRequest



class CoveragePathPlanner:
    def __init__(self):
        rospy.loginfo("Boundary-Following Coverage Path Planner 초기화 중...")

        # Boundary following parameters
        self.safety_margin = rospy.get_param('~safety_margin', 0.0)  # Reduced from 0.1 to 0.05 (5cm safety margin)
        self.boundary_resolution = rospy.get_param('~boundary_resolution', 0.2)  # 50cm between boundary points
        self.min_boundary_length = rospy.get_param('~min_boundary_length', 1.0)  # Minimum 2m boundary to consider
        self.local_grid_resolution = rospy.get_param('~local_grid_resolution', 0.05)
        
        # Path smoothing parameters
        self.enable_smoothing = rospy.get_param('~enable_smoothing', False)
        self.bspline_smoothness = rospy.get_param('~bspline_smoothness', 0.1)  # Higher = smoother (good for boundaries)
        self.moving_avg_window = rospy.get_param('~moving_avg_window', 7)  # Points to average (odd number works better)
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_pose_received = False

        self.traversable_points_3d = None
        self.kdtree = None

        # Publishers
        self.boundary_markers_pub = rospy.Publisher("/boundary_markers", MarkerArray, queue_size=2)
        self.boundary_path_pub = rospy.Publisher("/coverage_tsp_path", Path, queue_size=2)
        self.local_map_pub = rospy.Publisher("/local_traversable_map", OccupancyGrid, queue_size=2)
        self.contour_debug_pub = rospy.Publisher("/contour_debug_markers", MarkerArray, queue_size=2)

        # Subscribers
        self.points_sub = rospy.Subscriber("/traversable_area_filtered", PointCloud2, self._pointcloud_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=1)
        self.occupancy_sub = rospy.Subscriber("/occupancy_map", OccupancyGrid, self._occupancy_callback, queue_size=1)
        
        # Subscribe to exploration strategy to publish saved path when coverage_planning is requested
        self.strategy_sub = rospy.Subscriber("/exploration_strategy", String, self._strategy_callback, queue_size=1)
        
        # Subscribe to dedicated coverage regeneration topic to avoid conflicts with main strategy topic
        self.regen_sub = rospy.Subscriber("/coverage_path_regenerate", String, self._regeneration_callback, queue_size=1)
        
        # Track current strategy and periodically republish coverage path if needed
        self.current_strategy = ""
        self.path_republish_timer = rospy.Timer(rospy.Duration(5.0), self._republish_timer_callback)
        
        # Store occupancy map for boundary detection
        self.occupancy_map = None
        self.occupancy_info = None
        
        # Coverage path calculation - boundaries calculated once, paths calculated per robot position
        self.boundaries_calculated = False
        self.saved_boundary_loops = None
        self.saved_global_grid = None
        self.saved_global_origin = None
        self.saved_global_resolution = None
        self.saved_header = None
        
        # Current coverage path - calculated once per coverage planning session
        self.current_coverage_path = None
        self.coverage_path_start_position = None
        
        # A* service client
        rospy.loginfo("Waiting for A* path service...")
        rospy.wait_for_service('/astar_path')
        self.astar_service = rospy.ServiceProxy('/astar_path', AstarPath)
        rospy.loginfo("A* path service found.")
        
        rospy.loginfo("Boundary coverage planner ready. Will calculate boundaries once, then generate paths from current robot position.")

    def _odom_callback(self, msg: Odometry):
        """Update robot position for path planning"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_pose_received = True

    def _occupancy_callback(self, msg: OccupancyGrid):
        """Store occupancy map for reference (not used for boundary generation)"""
        self.occupancy_map = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.occupancy_info = msg.info

    def _strategy_callback(self, msg: String):
        """Respond to exploration strategy changes - generate path from current robot position ONCE per session"""
        strategy = msg.data
        previous_strategy = self.current_strategy
        self.current_strategy = strategy
        
        if strategy == "coverage_planning":
            # Only generate new path if switching TO coverage_planning from a different strategy
            if previous_strategy != "coverage_planning":
                if self.boundaries_calculated and self.saved_boundary_loops:
                    rospy.loginfo(f"NEW coverage planning session - generating path from current robot position ({self.robot_x:.2f}, {self.robot_y:.2f})")
                    self._generate_coverage_path_from_current_position()
                else:
                    rospy.logwarn("Coverage planning strategy requested but boundaries not calculated yet!")
            else:
                # Already in coverage planning, just republish existing path
                if self.current_coverage_path:
                    rospy.loginfo("Continuing coverage planning - republishing existing path")
                    self.boundary_path_pub.publish(self.current_coverage_path)
        elif previous_strategy == "coverage_planning":
            # Switching away from coverage planning - clear the current path
            rospy.loginfo("Switching away from coverage planning - clearing current coverage path")
            self.current_coverage_path = None
            self.coverage_path_start_position = None

    def _regeneration_callback(self, msg: String):
        """Handle coverage path regeneration requests"""
        rospy.loginfo("Received coverage path regeneration request")
        self._generate_coverage_path_from_current_position()

    def _republish_timer_callback(self, event):
        """Periodically republish the SAME coverage path if coverage_planning is active"""
        if (self.current_strategy == "coverage_planning" and 
            self.current_coverage_path):
            rospy.loginfo("Republishing existing coverage path (timer-based)")
            self.boundary_path_pub.publish(self.current_coverage_path)

    def _generate_coverage_path_from_current_position(self):
        """Generate coverage path starting from current robot position using saved boundaries - ONCE per session"""
        if not self.boundaries_calculated or not self.saved_boundary_loops:
            rospy.logwarn("Cannot generate coverage path - boundaries not available")
            return
            
        if not self.robot_pose_received:
            rospy.logwarn("Cannot generate coverage path - robot position not available")
            return
            
        rospy.loginfo(f"Generating coverage path from robot position ({self.robot_x:.2f}, {self.robot_y:.2f})")
        
        # Generate path using saved boundary data and current robot position
        boundary_path = self._generate_boundary_following_path(
            self.saved_boundary_loops, 
            self.saved_header, 
            self.saved_global_grid, 
            self.saved_global_origin, 
            self.saved_global_resolution
        )
        
        if boundary_path and boundary_path.poses:
            # Update timestamp for current request
            boundary_path.header.stamp = rospy.Time.now()
            
            # Save this path for the current coverage planning session
            self.current_coverage_path = boundary_path
            self.coverage_path_start_position = (self.robot_x, self.robot_y)
            
            self.boundary_path_pub.publish(boundary_path)
            rospy.loginfo(f"SAVED coverage path with {len(boundary_path.poses)} poses for this session. Will NOT regenerate until strategy changes.")
        else:
            rospy.logwarn("Failed to generate coverage path from current robot position")

    def _pointcloud_callback(self, msg: PointCloud2):
        """Process global traversable area and generate clean boundaries - ONLY ONCE"""
        if not self.robot_pose_received:
            rospy.logwarn("Robot pose not received yet. Skipping boundary generation.")
            return
        
        # Calculate boundaries only once (but paths will be recalculated per robot position)
        if self.boundaries_calculated:
            rospy.loginfo("Boundaries already calculated - skipping boundary processing")
            return
            
        rospy.loginfo("Calculating boundaries for the first time...")
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if len(points_list) < 10:
            rospy.logwarn("Insufficient points for boundary detection.")
            return
            
        rospy.loginfo(f"Processing {len(points_list)} traversable points...")

        points_2d = np.array([p[:2] for p in points_list])
        
        # Store traversable points for later A* planning
        self.traversable_points_3d = np.array(points_list, dtype=np.float32)
        
        # Create global occupancy grid from traversable area
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)
        
        padding = 2.0
        global_origin = (min_x - padding, min_y - padding)
        global_width = int((max_x - min_x + 2 * padding) / self.local_grid_resolution)
        global_height = int((max_y - min_y + 2 * padding) / self.local_grid_resolution)

        MAX_GRID_SIZE = 2000
        if global_width > MAX_GRID_SIZE or global_height > MAX_GRID_SIZE:
            scale_factor = max(global_width / MAX_GRID_SIZE, global_height / MAX_GRID_SIZE)
            new_resolution = self.local_grid_resolution * scale_factor
            global_width = int((max_x - min_x + 2 * padding) / new_resolution)
            global_height = int((max_y - min_y + 2 * padding) / new_resolution)
            rospy.loginfo(f"Using reduced resolution: {new_resolution:.3f}, new size: {global_width}x{global_height}")
        else:
            new_resolution = self.local_grid_resolution

        global_grid = np.full((global_height, global_width), 100, dtype=np.uint8)
        
        for point in points_2d:
            px = int((point[0] - global_origin[0]) / new_resolution)
            py = int((point[1] - global_origin[1]) / new_resolution)
            if 0 <= px < global_width and 0 <= py < global_height:
                global_grid[global_height - 1 - py, px] = 0

        boundary_loops = self._extract_clean_boundaries_from_grid(global_grid, global_origin, new_resolution, msg.header)
        
        if boundary_loops:
            # Save boundary data for later path generation
            self.saved_boundary_loops = boundary_loops
            self.saved_global_grid = global_grid
            self.saved_global_origin = global_origin
            self.saved_global_resolution = new_resolution
            self.saved_header = msg.header
            self.boundaries_calculated = True
            
            rospy.loginfo(f"BOUNDARIES CALCULATED SUCCESSFULLY! {len(boundary_loops)} boundary loops saved.")
            rospy.loginfo("Boundaries will be reused. Paths will be calculated from current robot position when coverage_planning is requested.")
        else:
            rospy.logwarn("No boundaries detected in global traversable area")

    def _extract_clean_boundaries_from_grid(self, grid, origin, resolution, header):
        """Extract clean boundaries from occupancy grid using morphological operations"""
        rospy.loginfo("Extracting clean boundaries from global grid...")
        
        clean_map = np.zeros_like(grid, dtype=np.uint8)
        clean_map[grid == 0] = 255
        clean_map[grid == 100] = 0
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_OPEN, kernel_small)
        clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_CLOSE, kernel_medium)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_map, connectivity=8)
        
        min_area = 500
        filtered_map = np.zeros_like(clean_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_map[labels == i] = 255
        
        contours, _ = cv2.findContours(filtered_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boundary_loops = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            min_perimeter_pixels = self.min_boundary_length / resolution
            
            if area > 200 and perimeter > min_perimeter_pixels:
                epsilon = 0.02 * perimeter
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                world_contour = []
                for point in smoothed_contour.reshape(-1, 2):
                    px, py = point[0], point[1]
                    wx = origin[0] + (px + 0.5) * resolution
                    wy = origin[1] + (grid.shape[0] - 1 - py + 0.5) * resolution
                    world_contour.append([wx, wy])
                
                if len(world_contour) > 3:
                    boundary_loops.append(np.array(world_contour))
                    rospy.loginfo(f"Added boundary loop with {len(world_contour)} points, area: {area:.0f} pixels")
        
        rospy.loginfo(f"Extracted {len(boundary_loops)} clean boundary loops from global grid")
        
        # Visualize all contours together after processing
        if boundary_loops:
            self._publish_all_contour_debug_markers(boundary_loops, header)
        
        self._publish_debug_grid(filtered_map, origin, resolution, header)
        
        return boundary_loops

    def _publish_debug_grid(self, grid, origin, resolution, header):
        """Publish cleaned grid for debugging visualization"""
        msg = OccupancyGrid()
        msg.header = header
        msg.header.stamp = rospy.Time.now()
        msg.info.resolution = resolution
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        msg.info.origin.orientation.w = 1.0
        
        occupancy_data = np.zeros_like(grid, dtype=np.int8)
        occupancy_data[grid == 0] = 100
        occupancy_data[grid == 255] = 0
        
        flipped_data = np.flipud(occupancy_data)
        msg.data = flipped_data.flatten().tolist()
        
        self.local_map_pub.publish(msg)



    def _publish_all_contour_debug_markers(self, boundary_loops, header):
        """Publish visualization markers for all world_contour points"""
        marker_array = MarkerArray()
        
        # Clear all previous markers
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Define colors for different loops
        colors = [
            ColorRGBA(0.0, 1.0, 0.0, 0.8),  # Green for outer boundary
            ColorRGBA(1.0, 1.0, 0.0, 0.8),  # Yellow for inner boundaries
            ColorRGBA(1.0, 0.0, 1.0, 0.8),  # Magenta
            ColorRGBA(0.0, 1.0, 1.0, 0.8),  # Cyan
            ColorRGBA(1.0, 0.5, 0.0, 0.8),  # Orange
            ColorRGBA(0.5, 0.0, 1.0, 0.8),  # Purple
        ]
        
        # Process each boundary loop
        for loop_id, world_contour in enumerate(boundary_loops):
            color = colors[loop_id % len(colors)]
            
            # Create markers for each contour point
            for i, point in enumerate(world_contour):
                # Point marker
                marker = Marker()
                marker.header = header
                marker.ns = f"contour_debug_{loop_id}"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1] 
                marker.pose.position.z = 0.2 + loop_id * 0.05  # Slightly different heights
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.08
                marker.scale.y = 0.08
                marker.scale.z = 0.08
                marker.color = color
                marker_array.markers.append(marker)
                
                # Text marker showing point index (only for first few points to avoid clutter)
                if i < 5 or i % 3 == 0:  # Show text for first 5 points and every 3rd point
                    text_marker = Marker()
                    text_marker.header = header
                    text_marker.ns = f"contour_text_{loop_id}"
                    text_marker.id = i
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.pose.position.x = point[0]
                    text_marker.pose.position.y = point[1]
                    text_marker.pose.position.z = 0.35 + loop_id * 0.05  # Above the sphere
                    text_marker.pose.orientation.w = 1.0
                    text_marker.scale.z = 0.08  # Text size
                    text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White text
                    text_marker.text = f"L{loop_id}P{i}"
                    marker_array.markers.append(text_marker)
            
            # Create line strip connecting the contour points
            line_marker = Marker()
            line_marker.header = header
            line_marker.ns = f"contour_line_{loop_id}"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.03  # Line width
            
            # Make line color slightly more transparent
            line_color = ColorRGBA(color.r, color.g, color.b, 0.6)
            line_marker.color = line_color
            
            # Add all contour points to the line
            for point in world_contour:
                p = Point()
                p.x, p.y, p.z = point[0], point[1], 0.15 + loop_id * 0.05
                line_marker.points.append(p)
            
            # Close the loop by connecting back to first point
            if len(world_contour) > 0:
                p = Point()
                p.x, p.y, p.z = world_contour[0][0], world_contour[0][1], 0.15 + loop_id * 0.05
                line_marker.points.append(p)
            
            marker_array.markers.append(line_marker)
        
        # Publish the markers
        self.contour_debug_pub.publish(marker_array)
        rospy.loginfo(f"Published debug visualization for {len(boundary_loops)} contours on topic /contour_debug_markers")
              
    def _generate_boundary_following_path(self, boundary_loops, header, planning_grid, planning_origin, planning_resolution):
        """Generate path following boundaries with safety margin - each contour processed separately"""
        if not boundary_loops:
            return None
            
        # Sort boundaries by size (outer boundary first, then inner obstacles)
        boundary_loops.sort(key=lambda loop: self._calculate_loop_area(loop), reverse=True)
        
        # Find which contour is closest to robot to determine starting point and sequence
        robot_pos = np.array([self.robot_x, self.robot_y])
        contour_distances = []
        
        for i, loop in enumerate(boundary_loops):
            # Find closest point in this contour to robot
            distances_to_robot = [np.linalg.norm(np.array(point) - robot_pos) for point in loop]
            min_distance = np.min(distances_to_robot)
            contour_distances.append(min_distance)
            rospy.loginfo(f"Contour {i} ({'outer' if i == 0 else 'inner'}): closest distance to robot = {min_distance:.2f}m")
        
        # Find the closest contour to robot
        closest_contour_idx = np.argmin(contour_distances)
        is_starting_with_outer = (closest_contour_idx == 0)
        
        rospy.loginfo(f"Robot closest to contour {closest_contour_idx} ({'outer boundary' if is_starting_with_outer else 'inner obstacle'})")
        
        # Determine processing order based on starting contour type
        if is_starting_with_outer:
            # Start with outer boundary, then inner obstacles
            processing_order = list(range(len(boundary_loops)))  # [0, 1, 2, 3, ...]
            rospy.loginfo("Sequence: Outer boundary → Inner obstacles")
        else:
            # Start with inner obstacles, then outer boundary
            inner_contours = list(range(1, len(boundary_loops)))  # [1, 2, 3, ...]
            # Put closest inner contour first, then other inner contours, then outer boundary
            closest_inner_idx = closest_contour_idx
            other_inner_contours = [idx for idx in inner_contours if idx != closest_inner_idx]
            processing_order = [closest_inner_idx] + other_inner_contours + [0]  # [closest_inner, other_inners..., outer]
            rospy.loginfo(f"Sequence: Inner obstacles (starting with {closest_inner_idx}) → Outer boundary")
        
        rospy.loginfo(f"Processing order: {processing_order}")
        
        # Process each contour according to the determined order
        contour_paths = []
        
        for order_idx, contour_idx in enumerate(processing_order):
            loop = boundary_loops[contour_idx]
            contour_type = 'outer boundary' if contour_idx == 0 else 'inner obstacle'
            rospy.loginfo(f"Processing contour {contour_idx} ({contour_type}) as step {order_idx + 1}/{len(processing_order)}...")
            
            # Apply safety margin and resampling
            shrunk_boundary = self._apply_safety_margin(loop, self.safety_margin)
            
            if shrunk_boundary is not None and len(shrunk_boundary) > 2:
                resampled_boundary = self._resample_boundary(shrunk_boundary, self.boundary_resolution)
                
                # Generate ordered path for this specific contour
                # Only use robot position for the first contour in the processing order
                use_robot_pos = (order_idx == 0)
                contour_path = self._order_single_contour_path(resampled_boundary, use_robot_position=use_robot_pos)
                
                if contour_path:
                    contour_paths.append({
                        'path': contour_path,
                        'contour_id': contour_idx,
                        'original_order': order_idx,
                        'type': 'outer' if contour_idx == 0 else 'inner'
                    })
                    rospy.loginfo(f"Contour {contour_idx}: {len(loop)} raw -> {len(shrunk_boundary)} shrunk -> {len(resampled_boundary)} resampled -> {len(contour_path)} ordered points")
                else:
                    rospy.logwarn(f"Failed to generate path for contour {contour_idx}")
            else:
                rospy.logwarn(f"Contour {contour_idx} invalid after safety margin application")
        
        if not contour_paths:
            rospy.logwarn("No valid contour paths generated")
            return None
        
        # Connect contours in the determined sequence
        ordered_path = self._connect_contour_paths(contour_paths)
        
        path_msg = Path()
        path_msg.header = header
        path_msg.header.stamp = rospy.Time.now()
        
        for point in ordered_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        if self.enable_smoothing and len(path_msg.poses) > 10:
            try:
                smoothed_poses = self.smooth_path_with_bspline(
                    path_msg, 
                    num_points_multiplier=1.2,
                    s=self.bspline_smoothness
                )
                if len(smoothed_poses) < len(path_msg.poses) * 0.7:
                    raise ValueError("B-spline reduced path points too much")
                path_msg.poses = smoothed_poses
            except Exception as e:
                rospy.logwarn(f"B-spline smoothing failed ({e}), using moving average instead")
                smoothed_poses = self.smooth_path_with_moving_average(
                    path_msg, 
                    window_size=self.moving_avg_window
                )
                path_msg.poses = smoothed_poses
        
        self._publish_boundary_markers(boundary_loops, header)
        
        return path_msg

    def _calculate_loop_area(self, loop):
        """Calculate area of a boundary loop using shoelace formula"""
        if len(loop) < 3:
            return 0
        x = loop[:, 0]
        y = loop[:, 1]
        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

    def _apply_safety_margin(self, boundary, margin):
        """Apply safety margin by shrinking boundary inward"""
        if len(boundary) < 3:
            return None
            
        try:
            centroid = np.mean(boundary, axis=0)
            shrunk_points = []
            for point in boundary:
                direction = centroid - point
                distance = np.linalg.norm(direction)
                if distance > margin:
                    unit_direction = direction / distance
                    new_point = point + unit_direction * margin
                    shrunk_points.append(new_point)
                    
            return np.array(shrunk_points) if len(shrunk_points) > 2 else None
            
        except Exception as e:
            rospy.logwarn(f"Failed to apply safety margin: {e}")
            return boundary

    def _resample_boundary(self, boundary, resolution):
        """Resample boundary points at specified resolution"""
        if len(boundary) < 2:
            return boundary.tolist()
            
        resampled = [boundary[0]]
        
        for i in range(1, len(boundary)):
            prev_point = resampled[-1]
            curr_point = boundary[i]
            
            distance = np.linalg.norm(curr_point - prev_point)
            
            if distance > resolution:
                num_points = int(distance / resolution)
                for j in range(1, num_points + 1):
                    t = j / num_points
                    interp_point = prev_point + t * (curr_point - prev_point)
                    resampled.append(interp_point)
            else:
                resampled.append(curr_point)
                
        return resampled



    def _ensure_contour_connectivity(self, contour_points, grid, origin, resolution):
        """Ensure contour points are connected by marking corridors between nearby points"""
        corridor_width = 3  # cells
        
        for i, point1 in enumerate(contour_points):
            for j, point2 in enumerate(contour_points[i+1:], i+1):
                # Only connect nearby points to avoid long connections
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance < 5.0:  # 5 meter max connection distance
                    
                    # Convert to grid coordinates
                    x1 = int((point1[0] - origin[0]) / resolution)
                    y1 = int((point1[1] - origin[1]) / resolution)
                    x2 = int((point2[0] - origin[0]) / resolution)
                    y2 = int((point2[1] - origin[1]) / resolution)
                    
                    # Draw corridor between points
                    line_points = self._bresenham_line(x1, y1, x2, y2)
                    
                    for px, py in line_points:
                        # Draw corridor around line
                        for dy in range(-corridor_width//2, corridor_width//2 + 1):
                            for dx in range(-corridor_width//2, corridor_width//2 + 1):
                                grid_x = px + dx
                                grid_y = grid.shape[0] - 1 - (py + dy)  # Flip Y
                                if 0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]:
                                    grid[grid_y, grid_x] = 0



    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to get points between two coordinates"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    def _order_single_contour_path(self, boundary_points, use_robot_position=False):
        """Order boundary points by following natural contour order"""
        if not boundary_points:
            return []
        
        if use_robot_position:
            # Only for the first contour - find nearest point to robot as starting point
            robot_pos = np.array([self.robot_x, self.robot_y])
            distances_to_robot = [np.linalg.norm(np.array(bp) - robot_pos) for bp in boundary_points]
            start_idx = np.argmin(distances_to_robot)
            
            # Reorder boundary points to start from the nearest point to robot
            ordered_boundary_points = boundary_points[start_idx:] + boundary_points[:start_idx]
            rospy.loginfo(f"First contour: starting from point nearest to robot (idx {start_idx})")
        else:
            # For subsequent contours - use natural order without robot position consideration
            ordered_boundary_points = boundary_points
            rospy.loginfo(f"Subsequent contour: using natural boundary order ({len(ordered_boundary_points)} points)")
        
        # Generate detailed A* path connecting the boundary points in natural order using A* service
        return self._create_detailed_astar_contour_path(ordered_boundary_points)



    def _create_detailed_astar_contour_path(self, ordered_boundary_points):
        """Create detailed A* path connecting boundary points in natural order within a single contour (OPEN path)"""
        if not ordered_boundary_points or len(ordered_boundary_points) < 2:
            return ordered_boundary_points
        
        detailed_path = []
        
        rospy.loginfo(f"Creating A* path through {len(ordered_boundary_points)} boundary points in natural order using A* service")
        
        # Start from first boundary point (contour is self-contained)
        current_pos = np.array(ordered_boundary_points[0])
        detailed_path.append([current_pos[0], current_pos[1]])  # Add starting point
        
        # Connect boundary points in sequence (skip first since it's already added)
        for i in range(1, len(ordered_boundary_points)):
            target_pos = np.array(ordered_boundary_points[i])
            
            # Generate A* path from current position to target boundary point using A* service
            segment_points = self._get_astar_path_via_service(current_pos, target_pos)
            
            if segment_points:
                # Skip first point to avoid duplicates with previous segment end
                detailed_path.extend(segment_points[1:])
                rospy.loginfo(f"Added A* segment to boundary point {i+1}/{len(ordered_boundary_points)} via service: {len(segment_points)} points")
            else:
                rospy.logwarn(f"Failed to create A* path to boundary point {i+1}/{len(ordered_boundary_points)} via service")
            
            # Update current position
            current_pos = target_pos
        
        # NOTE: Contours are kept OPEN - no loop closing
        # Each contour is a path segment that will be connected to other contours via _connect_contour_paths
        
        rospy.loginfo(f"Generated detailed A* contour path with {len(detailed_path)} total points")
        return detailed_path

    def _create_detailed_contour_path(self, ordered_waypoints, all_boundary_points, planning_grid, planning_origin, planning_resolution):
        """Create detailed path following the boundary between ordered waypoints"""
        if not ordered_waypoints:
            return []
        
        detailed_path = []
        
        for i in range(len(ordered_waypoints)):
            current_wp = ordered_waypoints[i]
            next_wp = ordered_waypoints[(i + 1) % len(ordered_waypoints)]  # Wrap around for closed loop
            
            # Find boundary points between current and next waypoint
            segment_points = self._find_boundary_segment_between_waypoints(current_wp, next_wp, all_boundary_points)
            
            if i == 0:
                detailed_path.extend(segment_points)
            else:
                detailed_path.extend(segment_points[1:])  # Skip first point to avoid duplicates
        
        return detailed_path

    def _find_boundary_segment_between_waypoints(self, start_wp, end_wp, boundary_points):
        """Find boundary points that lie between two waypoints"""
        # Find closest boundary points to the waypoints
        boundary_array = np.array(boundary_points)
        
        start_distances = np.linalg.norm(boundary_array - np.array(start_wp), axis=1)
        end_distances = np.linalg.norm(boundary_array - np.array(end_wp), axis=1)
        
        start_idx = np.argmin(start_distances)
        end_idx = np.argmin(end_distances)
        
        # Get the segment between these indices
        if start_idx <= end_idx:
            segment_indices = list(range(start_idx, end_idx + 1))
        else:
            # Wrap around
            segment_indices = list(range(start_idx, len(boundary_points))) + list(range(0, end_idx + 1))
        
        segment = [boundary_points[i] for i in segment_indices]
        
        # If segment is too sparse, interpolate
        if len(segment) < 3:
            num_interp = max(3, int(np.linalg.norm(np.array(end_wp) - np.array(start_wp)) / self.boundary_resolution))
            t_values = np.linspace(0, 1, num_interp)
            segment = [np.array(start_wp) + t * (np.array(end_wp) - np.array(start_wp)) for t in t_values]
        
        return segment

    def _connect_contour_paths(self, contour_paths):
        """Connect multiple contour paths in sequence with A* transitions"""
        if not contour_paths:
            return []
        
        robot_pos = np.array([self.robot_x, self.robot_y])
        final_path = []
        current_position = robot_pos
        
        for i, contour_data in enumerate(contour_paths):
            contour_path = contour_data['path']
            contour_type = contour_data['type']
            
            if not contour_path:
                continue
            
            rospy.loginfo(f"Connecting to {contour_type} contour {i} with {len(contour_path)} points")
            
            # For first contour in processing order, it's already ordered from robot position
            # For subsequent contours, find the closest point to current position and reorder
            original_order = contour_data.get('original_order', i)
            if original_order == 0:
                # First contour in processing order is already ordered to start from nearest point to robot
                reordered_contour = contour_path
                rospy.loginfo(f"First contour in sequence (contour {contour_data['contour_id']}): using existing order (already starts from robot-nearest point)")
            else:
                # Find the closest point in this contour to current position (end of previous contour)
                distances = [np.linalg.norm(np.array(point) - current_position) for point in contour_path]
                closest_idx = np.argmin(distances)
                
                # Reorder contour path to start from closest point to previous contour's end
                reordered_contour = contour_path[closest_idx:] + contour_path[:closest_idx]
                rospy.loginfo(f"Subsequent contour (contour {contour_data['contour_id']}): reordered to start from point closest to previous contour end")
            
            # If this is not the first contour in processing order, create A* transition from current position to contour start using A* service
            if original_order > 0:
                transition_points = self._get_astar_path_via_service(current_position, np.array(reordered_contour[0]))
                
                if transition_points:
                    final_path.extend(transition_points[1:])  # Skip first point to avoid duplicate
                    rospy.loginfo(f"Added A* transition to contour {i} via service: {len(transition_points)} points")
                else:
                    rospy.logwarn(f"Failed to create A* transition to contour {i} via service - contours may be disconnected")
                    # Fallback: add direct connection (but warn user)
                    final_path.append(reordered_contour[0])
                    rospy.logwarn("Using direct connection - path may go through non-traversable areas")
            
            # Add the reordered contour path
            if original_order == 0:
                # For first contour, create A* path from robot to first contour point using A* service
                robot_to_contour_points = self._get_astar_path_via_service(current_position, np.array(reordered_contour[0]))
                
                if robot_to_contour_points:
                    final_path.extend(robot_to_contour_points)
                    rospy.loginfo(f"Added A* path from robot to first contour via service: {len(robot_to_contour_points)} points")
                else:
                    rospy.logwarn("Failed to create A* path from robot to first contour via service")
                    # Fallback: add direct connection
                    final_path.append([current_position[0], current_position[1]])
                
                # Add the contour path (skip first point to avoid duplicate with A* path end)
                final_path.extend(reordered_contour[1:])
            else:
                final_path.extend(reordered_contour[1:])  # Skip first point to avoid duplicate
            
            # Update current position to end of this contour
            current_position = np.array(reordered_contour[-1])
        
        rospy.loginfo(f"Connected {len(contour_paths)} contours into final path with {len(final_path)} points")
        return final_path

    def _order_path_from_robot(self, all_boundary_points, planning_grid, planning_origin, planning_resolution):
        """
        Order boundary points by first solving TSP on a subset of waypoints
        using traversable A* distances, then generating detailed paths.
        """
        if not all_boundary_points:
            return []
        
        robot_pos = np.array([self.robot_x, self.robot_y])
        
        # Subsample waypoints for TSP
        MAX_TSP_WAYPOINTS = 12
        
        if len(all_boundary_points) <= MAX_TSP_WAYPOINTS:
            waypoints = all_boundary_points
        else:
            step = len(all_boundary_points) // MAX_TSP_WAYPOINTS
            waypoints = [all_boundary_points[i] for i in range(0, len(all_boundary_points), step)]
            if len(waypoints) > MAX_TSP_WAYPOINTS:
                waypoints = waypoints[:MAX_TSP_WAYPOINTS]
            
        rospy.loginfo(f"Using {len(waypoints)} waypoints for TSP from {len(all_boundary_points)} total boundary points")
        
        # Solve TSP with A* distances
        ordered_waypoints = self._solve_tsp_for_waypoints(waypoints, robot_pos, planning_grid, planning_origin, planning_resolution)
        
        if not ordered_waypoints:
            rospy.logwarn("TSP failed, using nearest neighbor ordering instead.")
            return self._nearest_neighbor_ordering(all_boundary_points, robot_pos)
        
        # Now, create a detailed path by connecting waypoints with A* paths
        detailed_path_points = []
        nodes = [robot_pos] + ordered_waypoints
        
        for i in range(len(nodes) - 1):
            start_point = nodes[i]
            end_point = nodes[i+1]
            
            segment_path = self._get_detailed_astar_path(
                start_point, end_point, planning_grid, planning_origin,
                planning_resolution, planning_grid.shape[0], planning_grid.shape[1], Path().header
            )
            
            # Convert poses back to points and add to detailed path
            segment_points = [np.array([p.pose.position.x, p.pose.position.y]) for p in segment_path]
            
            if detailed_path_points and segment_points:
                detailed_path_points.extend(segment_points[1:])
            else:
                detailed_path_points.extend(segment_points)
        
        # If the path is a loop, connect the last waypoint to the first
        if len(ordered_waypoints) > 1:
            start_point = ordered_waypoints[-1]
            end_point = ordered_waypoints[0]
            
            segment_path = self._get_detailed_astar_path(
                start_point, end_point, planning_grid, planning_origin,
                planning_resolution, planning_grid.shape[0], planning_grid.shape[1], Path().header
            )
            segment_points = [np.array([p.pose.position.x, p.pose.position.y]) for p in segment_path]
            detailed_path_points.extend(segment_points[1:])
        
        return detailed_path_points

    def _solve_tsp_for_waypoints(self, waypoints, robot_pos, planning_grid, planning_origin, planning_resolution):
        """Solve TSP for waypoint ordering using traversable path distances"""
        nodes = [robot_pos] + waypoints
        num_nodes = len(nodes)
        
        rospy.loginfo(f"Creating traversable distance matrix for {num_nodes} waypoints...")
        
        planner = AStarPlanner(planning_grid, planning_resolution)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    start_world = nodes[i]
                    end_world = nodes[j]
                    
                    start_px = int((start_world[0] - planning_origin[0]) / planning_resolution)
                    start_py = int((start_world[1] - planning_origin[1]) / planning_resolution)
                    end_px = int((end_world[0] - planning_origin[0]) / planning_resolution)
                    end_py = int((end_world[1] - planning_origin[1]) / planning_resolution)
                    
                    start_pixel = (start_px, planning_grid.shape[0] - 1 - start_py)
                    end_pixel = (end_px, planning_grid.shape[0] - 1 - end_py)
                    
                    _, path_length = planner.plan(start_pixel, end_pixel)
                    
                    if path_length == float('inf'):
                        euclidean_dist = np.linalg.norm(np.array(start_world) - np.array(end_world))
                        distance_matrix[i, j] = euclidean_dist * 100
                    else:
                        distance_matrix[i, j] = path_length
                else:
                    distance_matrix[i, j] = 0
        
        try:
            permutation, total_distance = solve_tsp_dynamic_programming(distance_matrix)
            rospy.loginfo(f"TSP solved with traversable paths! Total distance: {total_distance:.2f}m")
            
            ordered_waypoints = []
            robot_idx_in_permutation = permutation.index(0)
            
            for i in range(1, len(permutation)):
                perm_idx = (robot_idx_in_permutation + i) % len(permutation)
                node_idx = permutation[perm_idx]
                if node_idx > 0:
                    ordered_waypoints.append(waypoints[node_idx - 1])
            
            return ordered_waypoints
            
        except Exception as e:
            rospy.logerr(f"TSP solver failed: {e}")
            return None
    
    def _get_astar_path_via_service(self, start_world, end_world):
        """Get A* path using the external A* service"""
        try:
            req = AstarPathRequest()
            req.start_x = start_world[0]
            req.start_y = start_world[1]
            req.end_x = end_world[0]
            req.end_y = end_world[1]
            
            response = self.astar_service(req)
            
            if response.path_length < 0:
                rospy.logwarn(f"A* service could not find path from ({start_world[0]:.2f}, {start_world[1]:.2f}) to ({end_world[0]:.2f}, {end_world[1]:.2f})")
                return []
            
            # Convert Path message to list of points
            path_points = [[pose.pose.position.x, pose.pose.position.y] for pose in response.path.poses]
            return path_points
            
        except rospy.ServiceException as e:
            rospy.logerr(f"A* service call failed: {e}")
            return []

    def _nearest_neighbor_ordering(self, boundary_points, robot_pos):
        """Fallback ordering using nearest neighbor algorithm"""
        if not boundary_points:
            return []
            
        unvisited = list(range(len(boundary_points)))
        ordered_points = []
        current_pos = robot_pos
        
        while unvisited:
            distances = [np.linalg.norm(np.array(boundary_points[i]) - current_pos) for i in unvisited]
            nearest_idx_in_unvisited = np.argmin(distances)
            nearest_idx = unvisited[nearest_idx_in_unvisited]
            
            ordered_points.append(boundary_points[nearest_idx])
            current_pos = np.array(boundary_points[nearest_idx])
            del unvisited[nearest_idx_in_unvisited]
        
        rospy.loginfo(f"Nearest neighbor ordering: {len(ordered_points)} points")
        return ordered_points

    def _publish_boundary_markers(self, boundary_loops, header):
        """Publish boundary visualization markers"""
        marker_array = MarkerArray()
        
        for i, loop in enumerate(boundary_loops):
            marker = Marker()
            marker.header = header
            marker.ns = "boundary_loops"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            
            if i == 0:
                marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
            else:
                marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
            
            for point in loop:
                p = Point()
                p.x, p.y, p.z = point[0], point[1], 0.1
                marker.points.append(p)
                
            if len(loop) > 0:
                p = Point()
                p.x, p.y, p.z = loop[0][0], loop[0][1], 0.1
                marker.points.append(p)
                
            marker_array.markers.append(marker)
        
        self.boundary_markers_pub.publish(marker_array)

    def smooth_path_with_bspline(self,path_in, num_points_multiplier=2, s=0.5):
        x_coords = [pose.pose.position.x for pose in path_in.poses]
        y_coords = [pose.pose.position.y for pose in path_in.poses]

        if len(x_coords) < 4:
            rospy.logwarn("경로 점 개수가 부족하여 B-Spline 스무딩을 건너뜁니다.")
            return path_in.poses

        tck, u = splprep([x_coords, y_coords], s=s, k=3)

        if tck is None:
            rospy.logerr("B-Spline 계산에 실패했습니다.")
            return path_in.poses

        new_num_points = int(len(x_coords) * num_points_multiplier)
        u_new = np.linspace(u.min(), u.max(), new_num_points)
        x_new, y_new = splev(u_new, tck)

        smoothed_poses = []
        for i in range(len(x_new)):
            pose = PoseStamped()
            pose.header = path_in.header
            pose.pose.position.x = x_new[i]
            pose.pose.position.y = y_new[i]
            pose.pose.orientation.w = 1.0
            smoothed_poses.append(pose)
            
        return smoothed_poses

    def smooth_path_with_moving_average(self,path_in, window_size=30):
        if window_size < 3:
            return path_in.poses 

        x = np.array([p.pose.position.x for p in path_in.poses])
        y = np.array([p.pose.position.y for p in path_in.poses])

        x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='same')
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        
        x_smooth[:window_size//2] = x[:window_size//2]
        y_smooth[:window_size//2] = y[:window_size//2]
        x_smooth[-window_size//2:] = x[-window_size//2:]
        y_smooth[-window_size//2:] = y[-window_size//2:]

        smoothed_poses = []
        for i in range(len(x_smooth)):
            pose = PoseStamped()
            pose.header = path_in.header
            pose.pose.position.x = x_smooth[i]
            pose.pose.position.y = y_smooth[i]
            pose.pose.orientation.w = 1.0
            smoothed_poses.append(pose)
            
        return smoothed_poses

if __name__ == '__main__':
    try:
        rospy.init_node('coverage_path_planner_detailed_path', anonymous=True)
        CoveragePathPlanner()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
























# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy
# import numpy as np
# import math
# import heapq
# import cv2

# from scipy.spatial import ConvexHull, cKDTree
# from typing import Tuple, List
# from scipy.interpolate import splprep, splev 
# from nav_msgs.msg import Path, OccupancyGrid, Odometry
# from sensor_msgs.msg import PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point, PoseStamped
# from std_msgs.msg import ColorRGBA
# from python_tsp.exact import solve_tsp_dynamic_programming
# from sklearn.cluster import DBSCAN




# class AStarPlanner:
#     def __init__(self, grid_map, resolution):
#         self.grid_map = grid_map
#         self.resolution = resolution
#         self.height, self.width = grid_map.shape
#         self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

#     def _heuristic(self, a, b):
#         return math.hypot(a[0] - b[0], a[1] - b[1])

#     def plan(self, start_pixel, end_pixel):
#         if not (0 <= start_pixel[0] < self.width and 0 <= start_pixel[1] < self.height and
#                 0 <= end_pixel[0] < self.width and 0 <= end_pixel[1] < self.height):
#             return None, float('inf')
#         if self.grid_map[start_pixel[1], start_pixel[0]] == 100 or \
#            self.grid_map[end_pixel[1], end_pixel[0]] == 100:
#             return None, float('inf')

#         open_set = []
#         heapq.heappush(open_set, (0, start_pixel))
#         came_from = {}
#         g_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
#         g_score[start_pixel] = 0
#         f_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
#         f_score[start_pixel] = self._heuristic(start_pixel, end_pixel)

#         while open_set:
#             _, current = heapq.heappop(open_set)
#             if current == end_pixel:
#                 return self._reconstruct_path(came_from, current)
#             for dx, dy in self.neighbors:
#                 neighbor = (current[0] + dx, current[1] + dy)
#                 if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
#                     continue
#                 if self.grid_map[neighbor[1], neighbor[0]] == 100:
#                     continue
#                 move_cost = math.hypot(dx, dy)
#                 tentative_g_score = g_score[current] + move_cost
#                 if tentative_g_score < g_score[neighbor]:
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g_score
#                     f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_pixel)
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
#         return None, float('inf')

#     def _reconstruct_path(self, came_from, current):
#         path_length = 0.0
#         path_pixels = [current]
#         while current in came_from:
#             prev = came_from[current]
#             dist = math.hypot(current[0] - prev[0], current[1] - prev[1])
#             path_length += dist
#             current = prev
#             path_pixels.append(current)
#         path_pixels.reverse()
#         return path_pixels, path_length * self.resolution

# class CoveragePathPlanner:
#     def __init__(self):
#         rospy.loginfo("Boundary-Following Coverage Path Planner 초기화 중...")

#         # Boundary following parameters
#         self.safety_margin = rospy.get_param('~safety_margin', 0.3)  # 30cm safety margin
#         self.boundary_resolution = rospy.get_param('~boundary_resolution', 0.5)  # 50cm between boundary points
#         self.min_boundary_length = rospy.get_param('~min_boundary_length', 2.0)  # Minimum 2m boundary to consider
#         self.local_grid_resolution = rospy.get_param('~local_grid_resolution', 0.05)
        
#         # Path smoothing parameters
#         self.enable_smoothing = rospy.get_param('~enable_smoothing', True)
#         self.bspline_smoothness = rospy.get_param('~bspline_smoothness', 2.0)  # Higher = smoother (good for boundaries)
#         self.moving_avg_window = rospy.get_param('~moving_avg_window', 7)  # Points to average (odd number works better)
        
#         # Robot state
#         self.robot_x = 0.0
#         self.robot_y = 0.0
#         self.robot_pose_received = False

#         self.traversable_points_3d = None
#         self.kdtree = None

#         # Publishers
#         self.boundary_markers_pub = rospy.Publisher("/boundary_markers", MarkerArray, queue_size=2)
#         self.boundary_path_pub = rospy.Publisher("/coverage_tsp_path", Path, queue_size=2)
#         self.local_map_pub = rospy.Publisher("/local_traversable_map", OccupancyGrid, queue_size=2)

#         # Subscribers
#         self.points_sub = rospy.Subscriber("/traversable_area_filtered", PointCloud2, self._pointcloud_callback, queue_size=1)
#         self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=1)
#         self.occupancy_sub = rospy.Subscriber("/occupancy_map", OccupancyGrid, self._occupancy_callback, queue_size=1)
        
#         # Store occupancy map for boundary detection
#         self.occupancy_map = None
#         self.occupancy_info = None
        
#         # Performance optimization variables
#         self.last_processing_time = 0
#         self.processing_interval = 10.0  # Only process every 5 seconds
#         self.last_boundary_path = None
#         self.boundary_cache_timeout = 20.0  # Cache boundaries for 10 seconds
        
#         rospy.loginfo("Boundary coverage planner ready. Waiting for robot pose and traversable area...")
#         rospy.loginfo(f"Performance settings: Processing every {self.processing_interval}s, cache timeout {self.boundary_cache_timeout}s")
#         rospy.loginfo(f"Smoothing settings: enabled={self.enable_smoothing}, B-spline s={self.bspline_smoothness}, moving avg window={self.moving_avg_window}")
#         rospy.loginfo(f"Boundary settings: safety_margin={self.safety_margin}m, resolution={self.boundary_resolution}m, min_length={self.min_boundary_length}m")

#     def _odom_callback(self, msg: Odometry):
#         """Update robot position for path planning"""
#         self.robot_x = msg.pose.pose.position.x
#         self.robot_y = msg.pose.pose.position.y
#         self.robot_pose_received = True

#     def _occupancy_callback(self, msg: OccupancyGrid):
#         """Store occupancy map for reference (not used for boundary generation)"""
#         self.occupancy_map = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
#         self.occupancy_info = msg.info
#         # Note: We use global traversable area for boundary generation instead

#     def _pointcloud_callback(self, msg: PointCloud2):
#         """Process global traversable area and generate clean boundaries"""
#         if not self.robot_pose_received:
#             rospy.logwarn("Robot pose not received yet. Skipping boundary generation.")
#             return
        
#         # Performance throttling - only process every few seconds
#         current_time = rospy.Time.now().to_sec()
#         if current_time - self.last_processing_time < self.processing_interval:
#             return
            
#         # Check if we have a recent cached path
#         if (self.last_boundary_path and 
#             current_time - self.last_processing_time < self.boundary_cache_timeout):
#             rospy.loginfo("Using cached boundary path")
#             self.boundary_path_pub.publish(self.last_boundary_path)
#             return
            
#         rospy.loginfo("Processing global traversable area for clean boundary detection...")
#         points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

#         if len(points_list) < 10:
#             rospy.logwarn("Insufficient points for boundary detection.")
#             return
            
#         rospy.loginfo(f"Processing {len(points_list)} traversable points...")

#         points_2d = np.array([p[:2] for p in points_list])
        
#         # Create global occupancy grid from traversable area
#         min_x, min_y = np.min(points_2d, axis=0)
#         max_x, max_y = np.max(points_2d, axis=0)
        
#         padding = 2.0  # Larger padding for global map
#         global_origin = (min_x - padding, min_y - padding)
#         global_width = int((max_x - min_x + 2 * padding) / self.local_grid_resolution)
#         global_height = int((max_y - min_y + 2 * padding) / self.local_grid_resolution)

#         # Limit grid size to prevent memory issues
#         MAX_GRID_SIZE = 2000  # 2000x2000 max
#         if global_width > MAX_GRID_SIZE or global_height > MAX_GRID_SIZE:
#             rospy.logwarn(f"Grid too large ({global_width}x{global_height}). Reducing resolution...")
#             scale_factor = max(global_width / MAX_GRID_SIZE, global_height / MAX_GRID_SIZE)
#             new_resolution = self.local_grid_resolution * scale_factor
#             global_width = int((max_x - min_x + 2 * padding) / new_resolution)
#             global_height = int((max_y - min_y + 2 * padding) / new_resolution)
#             rospy.loginfo(f"Using reduced resolution: {new_resolution:.3f}, new size: {global_width}x{global_height}")
#         else:
#             new_resolution = self.local_grid_resolution

#         rospy.loginfo(f"Creating global occupancy grid: {global_width}x{global_height}, resolution: {new_resolution:.3f}")

#         # Create global binary occupancy grid - start with obstacles everywhere
#         global_grid = np.full((global_height, global_width), 100, dtype=np.uint8)
        
#         # Mark traversable areas as free space
#         for point in points_2d:
#             px = int((point[0] - global_origin[0]) / new_resolution)
#             py = int((point[1] - global_origin[1]) / new_resolution)
#             if 0 <= px < global_width and 0 <= py < global_height:
#                 global_grid[global_height - 1 - py, px] = 0

#         # Apply clean boundary detection logic (similar to occupancy map processing)
#         boundary_loops = self._extract_clean_boundaries_from_grid(global_grid, global_origin, new_resolution, msg.header)
        
#         if boundary_loops:
#             boundary_path = self._generate_boundary_following_path(boundary_loops, msg.header)
#             if boundary_path and boundary_path.poses:
#                 # Cache the path and update timing
#                 self.last_boundary_path = boundary_path
#                 self.last_processing_time = current_time
                
#                 self.boundary_path_pub.publish(boundary_path)
#                 rospy.loginfo(f"Published global boundary path with {len(boundary_path.poses)} poses")
#             else:
#                 rospy.logwarn("Failed to generate valid boundary path")
#         else:
#             rospy.logwarn("No boundaries detected in global traversable area")

#     def _extract_clean_boundaries_from_grid(self, grid, origin, resolution, header):
#         """Extract clean boundaries from occupancy grid using morphological operations"""
#         rospy.loginfo("Extracting clean boundaries from global grid...")
        
#         # Create clean binary map for boundary detection
#         # Grid: 0 = free space, 100 = obstacle
#         clean_map = np.zeros_like(grid, dtype=np.uint8)
#         clean_map[grid == 0] = 255    # Free space = white (interior)
#         clean_map[grid == 100] = 0    # Obstacles = black (boundaries)
        
#         # Apply morphological operations to clean up noise
#         kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
#         # Remove small noise in free space
#         clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_OPEN, kernel_small)
        
#         # Fill small holes in free space 
#         clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_CLOSE, kernel_medium)
        
#         # Additional cleanup: remove very small connected components
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_map, connectivity=8)
        
#         # Filter out small components (less than 500 pixels)
#         min_area = 500
#         filtered_map = np.zeros_like(clean_map)
#         for i in range(1, num_labels):  # Skip background (label 0)
#             if stats[i, cv2.CC_STAT_AREA] >= min_area:
#                 filtered_map[labels == i] = 255
        
#         # Find contours on the cleaned map
#         contours, _ = cv2.findContours(filtered_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         boundary_loops = []
        
#         for contour in contours:
#             # Filter by area and perimeter
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
            
#             # Convert perimeter threshold from meters to pixels
#             min_perimeter_pixels = self.min_boundary_length / resolution
            
#             if area > 200 and perimeter > min_perimeter_pixels:
#                 # Smooth the contour to reduce noise
#                 epsilon = 0.02 * perimeter  # 2% of perimeter
#                 smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
#                 # Convert pixel coordinates to world coordinates
#                 world_contour = []
#                 for point in smoothed_contour.reshape(-1, 2):
#                     px, py = point[0], point[1]
#                     # Convert from grid coordinates to world coordinates
#                     wx = origin[0] + (px + 0.5) * resolution
#                     wy = origin[1] + (grid.shape[0] - 1 - py + 0.5) * resolution
#                     world_contour.append([wx, wy])
                
#                 if len(world_contour) > 3:
#                     boundary_loops.append(np.array(world_contour))
#                     rospy.loginfo(f"Added boundary loop with {len(world_contour)} points, area: {area:.0f} pixels")
        
#         rospy.loginfo(f"Extracted {len(boundary_loops)} clean boundary loops from global grid")
        
#         # Publish the cleaned grid for debugging
#         self._publish_debug_grid(filtered_map, origin, resolution, header)
        
#         return boundary_loops

#     def _publish_debug_grid(self, grid, origin, resolution, header):
#         """Publish cleaned grid for debugging visualization"""
#         msg = OccupancyGrid()
#         msg.header = header
#         msg.header.stamp = rospy.Time.now()
#         msg.info.resolution = resolution
#         msg.info.width = grid.shape[1]
#         msg.info.height = grid.shape[0]
#         msg.info.origin.position.x = origin[0]
#         msg.info.origin.position.y = origin[1]
#         msg.info.origin.orientation.w = 1.0
        
#         # Convert to occupancy grid format (0-100 scale)
#         occupancy_data = np.zeros_like(grid, dtype=np.int8)
#         occupancy_data[grid == 0] = 100    # Obstacles
#         occupancy_data[grid == 255] = 0    # Free space
        
#         # Flip vertically for ROS convention
#         flipped_data = np.flipud(occupancy_data)
#         msg.data = flipped_data.flatten().tolist()
        
#         self.local_map_pub.publish(msg)

#     def _generate_boundaries_from_occupancy(self, header):
#         """Generate clean boundaries from occupancy map"""
#         if self.occupancy_map is None or self.occupancy_info is None:
#             return
            
#         rospy.loginfo("Generating boundaries from occupancy map...")
        
#         # Create clean binary map: 0 = free space, 255 = obstacle/unknown
#         clean_map = np.zeros_like(self.occupancy_map, dtype=np.uint8)
#         clean_map[self.occupancy_map == 0] = 255  # Free space = white
#         clean_map[self.occupancy_map >= 50] = 0   # Obstacles = black  
#         clean_map[self.occupancy_map == -1] = 0   # Unknown = black
        
#         # Apply morphological operations to clean up the map
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_CLOSE, kernel)  # Fill small holes
#         clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_OPEN, kernel)   # Remove small noise
        
#         # Find contours on the cleaned map
#         contours, _ = cv2.findContours(clean_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         boundary_loops = []
        
#         for contour in contours:
#             # Filter by area and perimeter
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
            
#             if area > 100 and perimeter > self.min_boundary_length / self.occupancy_info.resolution:
#                 # Convert pixel coordinates to world coordinates
#                 world_contour = []
#                 for point in contour.reshape(-1, 2):
#                     px, py = point[0], point[1]
#                     # Convert from occupancy grid coordinates to world coordinates
#                     wx = self.occupancy_info.origin.position.x + (px + 0.5) * self.occupancy_info.resolution
#                     wy = self.occupancy_info.origin.position.y + (py + 0.5) * self.occupancy_info.resolution
#                     world_contour.append([wx, wy])
                
#                 if len(world_contour) > 3:
#                     boundary_loops.append(np.array(world_contour))
        
#         rospy.loginfo(f"Detected {len(boundary_loops)} clean boundary loops from occupancy map")
        
#         if boundary_loops:
#             boundary_path = self._generate_boundary_following_path(boundary_loops, header)
#             if boundary_path and boundary_path.poses:
#                 self.boundary_path_pub.publish(boundary_path)
#                 rospy.loginfo(f"Published boundary path with {len(boundary_path.poses)} poses")
#             else:
#                 rospy.logwarn("Failed to generate valid boundary path")
#         else:
#             rospy.logwarn("No boundaries detected in occupancy map")
            
#     def _detect_boundary_loops(self, local_grid, local_origin, resolution):
#         """Detect boundary loops from occupancy grid using edge detection"""
#         # Find edges using Canny edge detection
#         edges = cv2.Canny(local_grid, 50, 150)
        
#         # Find contours from edges
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         boundary_loops = []
        
#         for contour in contours:
#             # Convert pixel coordinates to world coordinates
#             world_contour = []
#             for point in contour.reshape(-1, 2):
#                 px, py = point[0], point[1]
#                 # Convert from flipped grid coordinates to world coordinates
#                 py_world = local_grid.shape[0] - 1 - py
#                 wx = local_origin[0] + (px + 0.5) * resolution
#                 wy = local_origin[1] + (py_world + 0.5) * resolution
#                 world_contour.append([wx, wy])
            
#             world_contour = np.array(world_contour)
            
#             # Calculate contour length
#             if len(world_contour) > 3:
#                 perimeter = cv2.arcLength(contour, True) * resolution
#                 if perimeter > self.min_boundary_length:
#                     boundary_loops.append(world_contour)
        
#         rospy.loginfo(f"Detected {len(boundary_loops)} boundary loops")
#         return boundary_loops

#     def _generate_boundary_following_path(self, boundary_loops, header):
#         """Generate path following boundaries with safety margin"""
#         if not boundary_loops:
#             return None
            
#         # Sort boundaries by size (outer boundary first)
#         boundary_loops.sort(key=lambda loop: self._calculate_loop_area(loop), reverse=True)
        
#         all_boundary_points = []
        
#         # Process each boundary loop
#         for i, loop in enumerate(boundary_loops):
#             # Apply safety margin by shrinking the boundary inward
#             shrunk_boundary = self._apply_safety_margin(loop, self.safety_margin)
            
#             if shrunk_boundary is not None and len(shrunk_boundary) > 2:
#                 # Resample boundary points at desired resolution
#                 resampled_boundary = self._resample_boundary(shrunk_boundary, self.boundary_resolution)
#                 all_boundary_points.extend(resampled_boundary)
                
#                 rospy.loginfo(f"Boundary {i}: {len(loop)} raw -> {len(shrunk_boundary)} shrunk -> {len(resampled_boundary)} resampled points")
        
#         if not all_boundary_points:
#             rospy.logwarn("No valid boundary points after processing")
#             return None
        
#         # Create path starting from robot position
#         ordered_path = self._order_path_from_robot(all_boundary_points)
        
#         # Convert to ROS Path message
#         path_msg = Path()
#         path_msg.header = header
#         path_msg.header.stamp = rospy.Time.now()
        
#         for point in ordered_path:
#             pose = PoseStamped()
#             pose.header = path_msg.header
#             pose.pose.position.x = point[0]
#             pose.pose.position.y = point[1]
#             pose.pose.position.z = 0.0
#             pose.pose.orientation.w = 1.0
#             path_msg.poses.append(pose)
        
#         # Apply smoothing to the boundary path
#         if self.enable_smoothing and len(path_msg.poses) > 10:
#             rospy.loginfo(f"Applying smoothing to boundary path with {len(path_msg.poses)} points")
            
#             try:
#                 # Apply B-spline smoothing for gentle curves
#                 smoothed_poses = self.smooth_path_with_bspline(
#                     path_msg, 
#                     num_points_multiplier=1.2,  # Slight increase in density
#                     s=self.bspline_smoothness
#                 )
                
#                 # Validate B-spline result
#                 if len(smoothed_poses) < len(path_msg.poses) * 0.7:
#                     raise ValueError("B-spline reduced path points too much")
                
#                 path_msg.poses = smoothed_poses
#                 rospy.loginfo(f"B-spline smoothing: {len(path_msg.poses)} points (s={self.bspline_smoothness})")
                
#             except Exception as e:
#                 rospy.logwarn(f"B-spline smoothing failed ({e}), using moving average instead")
                
#                 # Fallback to moving average smoothing
#                 smoothed_poses = self.smooth_path_with_moving_average(
#                     path_msg, 
#                     window_size=self.moving_avg_window
#                 )
#                 path_msg.poses = smoothed_poses
#                 rospy.loginfo(f"Moving average smoothing: {len(path_msg.poses)} points (window={self.moving_avg_window})")
                
#         elif not self.enable_smoothing:
#             rospy.loginfo("Path smoothing disabled - using raw boundary path")
#         else:
#             rospy.loginfo("Path too short for smoothing, using original boundary path")
        
#         # Publish boundary markers for visualization
#         self._publish_boundary_markers(boundary_loops, header)
        
#         return path_msg

#     def _calculate_loop_area(self, loop):
#         """Calculate area of a boundary loop using shoelace formula"""
#         if len(loop) < 3:
#             return 0
#         x = loop[:, 0]
#         y = loop[:, 1]
#         return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

#     def _apply_safety_margin(self, boundary, margin):
#         """Apply safety margin by shrinking boundary inward"""
#         if len(boundary) < 3:
#             return None
            
#         try:
#             # Calculate centroid
#             centroid = np.mean(boundary, axis=0)
            
#             # Move each point toward centroid by safety margin
#             shrunk_points = []
#             for point in boundary:
#                 direction = centroid - point
#                 distance = np.linalg.norm(direction)
#                 if distance > margin:
#                     # Move inward by safety margin
#                     unit_direction = direction / distance
#                     new_point = point + unit_direction * margin
#                     shrunk_points.append(new_point)
                    
#             return np.array(shrunk_points) if len(shrunk_points) > 2 else None
            
#         except Exception as e:
#             rospy.logwarn(f"Failed to apply safety margin: {e}")
#             return boundary

#     def _resample_boundary(self, boundary, resolution):
#         """Resample boundary points at specified resolution"""
#         if len(boundary) < 2:
#             return boundary.tolist()
            
#         resampled = [boundary[0]]  # Start with first point
        
#         for i in range(1, len(boundary)):
#             prev_point = resampled[-1]
#             curr_point = boundary[i]
            
#             # Calculate distance between points
#             distance = np.linalg.norm(curr_point - prev_point)
            
#             # Add intermediate points if needed
#             if distance > resolution:
#                 num_points = int(distance / resolution)
#                 for j in range(1, num_points + 1):
#                     t = j / num_points
#                     interp_point = prev_point + t * (curr_point - prev_point)
#                     resampled.append(interp_point)
#             else:
#                 resampled.append(curr_point)
                
#         return resampled

#     def _order_path_from_robot(self, boundary_points):
#         """Order boundary points using TSP for waypoint ordering, then create smooth boundary-following path"""
#         if not boundary_points:
#             return []
        
#         robot_pos = np.array([self.robot_x, self.robot_y])
        
#         # Strategy: Use TSP on subsampled waypoints, then fill in detailed boundary path
#         MAX_TSP_WAYPOINTS = 12  # Reduced for faster TSP solving
        
#         if len(boundary_points) <= MAX_TSP_WAYPOINTS:
#             # Small enough - use all points for TSP
#             waypoints = boundary_points
#             all_boundary_points = boundary_points
#         else:
#             # Subsample waypoints for TSP ordering, keep all points for path generation
#             step = len(boundary_points) // MAX_TSP_WAYPOINTS
#             waypoints = [boundary_points[i] for i in range(0, len(boundary_points), step)]
#             if len(waypoints) > MAX_TSP_WAYPOINTS:
#                 waypoints = waypoints[:MAX_TSP_WAYPOINTS]
#             all_boundary_points = boundary_points
            
#         rospy.loginfo(f"Using {len(waypoints)} waypoints for TSP from {len(boundary_points)} total boundary points")
        
#         # Solve TSP on waypoints only
#         waypoint_order = self._solve_tsp_for_waypoints(waypoints, robot_pos)
        
#         if not waypoint_order:
#             rospy.logwarn("TSP failed, using nearest neighbor on all points")
#             return self._nearest_neighbor_ordering(boundary_points, robot_pos)
        
#         # Now create detailed path following boundary between ordered waypoints
#         if len(boundary_points) <= MAX_TSP_WAYPOINTS:
#             # Small case - waypoints are the actual boundary points
#             return waypoint_order
#         else:
#             # Large case - create smooth path following boundary between waypoints
#             return self._create_boundary_following_path(waypoint_order, all_boundary_points)
    
#     def _solve_tsp_for_waypoints(self, waypoints, robot_pos):
#         """Solve TSP for waypoint ordering using traversable path distances"""
#         # Create nodes: robot position + waypoints
#         nodes = [robot_pos] + waypoints
#         num_nodes = len(nodes)
        
#         rospy.loginfo(f"Creating traversable distance matrix for {num_nodes} waypoints...")
        
#         # Create distance matrix using A* pathfinding
#         distance_matrix = np.zeros((num_nodes, num_nodes))
        
#         # We need to create a simple occupancy grid for A* planning
#         # Use the current global traversable area to create a planning grid
#         planning_grid, planning_origin, planning_resolution = self._create_planning_grid_for_waypoints(nodes)
        
#         if planning_grid is None:
#             rospy.logwarn("Failed to create planning grid, falling back to Euclidean distances")
#             return self._solve_tsp_euclidean(waypoints, robot_pos)
        
#         # Create A* planner for the waypoint region
#         planner = AStarPlanner(planning_grid, planning_resolution)
        
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:
#                     # Convert world coordinates to grid coordinates
#                     start_world = nodes[i]
#                     end_world = nodes[j]
                    
#                     start_px = int((start_world[0] - planning_origin[0]) / planning_resolution)
#                     start_py = int((start_world[1] - planning_origin[1]) / planning_resolution)
#                     end_px = int((end_world[0] - planning_origin[0]) / planning_resolution)
#                     end_py = int((end_world[1] - planning_origin[1]) / planning_resolution)
                    
#                     # Flip Y coordinate for grid convention
#                     start_pixel = (start_px, planning_grid.shape[0] - 1 - start_py)
#                     end_pixel = (end_px, planning_grid.shape[0] - 1 - end_py)
                    
#                     # Plan path using A*
#                     _, path_length = planner.plan(start_pixel, end_pixel)
                    
#                     if path_length == float('inf'):
#                         # If no path found, use large penalty but not infinite
#                         rospy.logwarn(f"No traversable path between waypoints {i} and {j}, using penalty distance")
#                         euclidean_dist = np.linalg.norm(np.array(start_world) - np.array(end_world))
#                         distance_matrix[i, j] = euclidean_dist * 10  # Large penalty
#                     else:
#                         distance_matrix[i, j] = path_length
#                 else:
#                     distance_matrix[i, j] = 0
        
#         try:
#             # Solve TSP with traversable distances
#             permutation, total_distance = solve_tsp_dynamic_programming(distance_matrix)
#             rospy.loginfo(f"TSP solved with traversable paths! Total distance: {total_distance:.2f}m")
            
#             # Extract ordered waypoints (skip robot position at index 0)
#             ordered_waypoints = []
#             robot_idx_in_permutation = permutation.index(0)
            
#             # Start from robot position and follow TSP order
#             for i in range(1, len(permutation)):
#                 perm_idx = (robot_idx_in_permutation + i) % len(permutation)
#                 node_idx = permutation[perm_idx]
#                 if node_idx > 0:  # Skip robot position (index 0)
#                     ordered_waypoints.append(waypoints[node_idx - 1])
            
#             return ordered_waypoints
            
#         except Exception as e:
#             rospy.logerr(f"TSP solver failed: {e}")
#             return None
    
#     def _solve_tsp_euclidean(self, waypoints, robot_pos):
#         """Fallback TSP using Euclidean distances"""
#         rospy.loginfo("Using fallback Euclidean TSP")
#         nodes = [robot_pos] + waypoints
#         num_nodes = len(nodes)
        
#         distance_matrix = np.zeros((num_nodes, num_nodes))
        
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:
#                     node_i = np.array(nodes[i])
#                     node_j = np.array(nodes[j])
#                     distance = np.linalg.norm(node_i - node_j)
#                     distance_matrix[i, j] = distance
        
#         try:
#             permutation, total_distance = solve_tsp_dynamic_programming(distance_matrix)
#             rospy.loginfo(f"Fallback TSP solved! Total distance: {total_distance:.2f}m")
            
#             ordered_waypoints = []
#             robot_idx_in_permutation = permutation.index(0)
            
#             for i in range(1, len(permutation)):
#                 perm_idx = (robot_idx_in_permutation + i) % len(permutation)
#                 node_idx = permutation[perm_idx]
#                 if node_idx > 0:
#                     ordered_waypoints.append(waypoints[node_idx - 1])
            
#             return ordered_waypoints
            
#         except Exception as e:
#             rospy.logerr(f"Fallback TSP also failed: {e}")
#             return None
    
#     def _create_planning_grid_for_waypoints(self, waypoint_nodes):
#         """Create a focused planning grid around the waypoints for A* pathfinding"""
#         if not waypoint_nodes:
#             return None, None, None
            
#         # Find bounding box of all waypoints
#         waypoints_array = np.array(waypoint_nodes)
#         min_x, min_y = np.min(waypoints_array, axis=0)
#         max_x, max_y = np.max(waypoints_array, axis=0)
        
#         # Add padding around waypoints
#         padding = 3.0  # 3 meter padding
#         planning_origin = (min_x - padding, min_y - padding)
#         planning_resolution = 0.1  # 10cm resolution for planning
        
#         planning_width = int((max_x - min_x + 2 * padding) / planning_resolution)
#         planning_height = int((max_y - min_y + 2 * padding) / planning_resolution)
        
#         # Limit planning grid size
#         max_planning_size = 500  # 500x500 max for fast A*
#         if planning_width > max_planning_size or planning_height > max_planning_size:
#             scale_factor = max(planning_width / max_planning_size, planning_height / max_planning_size)
#             planning_resolution *= scale_factor
#             planning_width = int((max_x - min_x + 2 * padding) / planning_resolution)
#             planning_height = int((max_y - min_y + 2 * padding) / planning_resolution)
#             rospy.loginfo(f"Scaled planning grid to {planning_width}x{planning_height} with resolution {planning_resolution:.3f}")
        
#         # Create planning grid (start with all obstacles)
#         planning_grid = np.full((planning_height, planning_width), 100, dtype=np.uint8)
        
#         # We need access to the current traversable points to mark free space
#         # This is a bit tricky since we don't store the global traversable area
#         # For now, we'll use a simple approach: mark areas around waypoints as free
#         # and assume they're connected (since they're on the boundary)
        
#         # Mark areas around each waypoint as free
#         free_radius_cells = int(1.0 / planning_resolution)  # 1 meter radius around each waypoint
        
#         for waypoint in waypoint_nodes:
#             wx, wy = waypoint[0], waypoint[1]
#             center_px = int((wx - planning_origin[0]) / planning_resolution)
#             center_py = int((wy - planning_origin[1]) / planning_resolution)
            
#             # Mark circular area around waypoint as free
#             for dy in range(-free_radius_cells, free_radius_cells + 1):
#                 for dx in range(-free_radius_cells, free_radius_cells + 1):
#                     if dx*dx + dy*dy <= free_radius_cells*free_radius_cells:
#                         px = center_px + dx
#                         py = center_py + dy
#                         if 0 <= px < planning_width and 0 <= py < planning_height:
#                             planning_grid[planning_height - 1 - py, px] = 0
        
#         # Connect waypoints with corridors (simplified connectivity)
#         self._connect_waypoints_in_grid(waypoint_nodes, planning_grid, planning_origin, planning_resolution)
        
#         rospy.loginfo(f"Created planning grid {planning_width}x{planning_height} for waypoint TSP")
#         return planning_grid, planning_origin, planning_resolution
    
#     def _connect_waypoints_in_grid(self, waypoint_nodes, grid, origin, resolution):
#         """Create simple connections between nearby waypoints in the planning grid"""
#         corridor_width_cells = int(0.6 / resolution)  # 60cm wide corridors
        
#         for i, wp1 in enumerate(waypoint_nodes):
#             for j, wp2 in enumerate(waypoint_nodes[i+1:], i+1):
#                 # Only connect if waypoints are reasonably close
#                 dist = np.linalg.norm(np.array(wp1) - np.array(wp2))
#                 if dist < 5.0:  # Only connect if less than 5 meters apart
                    
#                     # Convert to grid coordinates
#                     x1 = int((wp1[0] - origin[0]) / resolution)
#                     y1 = int((wp1[1] - origin[1]) / resolution)
#                     x2 = int((wp2[0] - origin[0]) / resolution)
#                     y2 = int((wp2[1] - origin[1]) / resolution)
                    
#                     # Draw line between waypoints
#                     line_points = self._bresenham_line(x1, y1, x2, y2)
                    
#                     for px, py in line_points:
#                         # Draw corridor around line
#                         for dy in range(-corridor_width_cells//2, corridor_width_cells//2 + 1):
#                             for dx in range(-corridor_width_cells//2, corridor_width_cells//2 + 1):
#                                 grid_x = px + dx
#                                 grid_y = grid.shape[0] - 1 - (py + dy)  # Flip Y
#                                 if 0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]:
#                                     grid[grid_y, grid_x] = 0
    
#     def _bresenham_line(self, x0, y0, x1, y1):
#         """Bresenham's line algorithm to get points between two coordinates"""
#         points = []
#         dx = abs(x1 - x0)
#         dy = abs(y1 - y0)
#         x, y = x0, y0
#         sx = -1 if x0 > x1 else 1
#         sy = -1 if y0 > y1 else 1
        
#         if dx > dy:
#             err = dx / 2.0
#             while x != x1:
#                 points.append((x, y))
#                 err -= dy
#                 if err < 0:
#                     y += sy
#                     err += dx
#                 x += sx
#         else:
#             err = dy / 2.0
#             while y != y1:
#                 points.append((x, y))
#                 err -= dx
#                 if err < 0:
#                     x += sx
#                     err += dy
#                 y += sy
#         points.append((x, y))
#         return points

#     def _create_boundary_following_path(self, ordered_waypoints, all_boundary_points):
#         """Create detailed path that follows boundary between waypoints"""
#         if not ordered_waypoints or not all_boundary_points:
#             return []
        
#         rospy.loginfo(f"Creating boundary-following path between {len(ordered_waypoints)} waypoints using {len(all_boundary_points)} boundary points")
        
#         # Convert to numpy arrays for easier processing
#         waypoints = np.array(ordered_waypoints)
#         boundary_array = np.array(all_boundary_points)
        
#         detailed_path = []
        
#         for i in range(len(waypoints)):
#             current_waypoint = waypoints[i]
#             next_waypoint = waypoints[(i + 1) % len(waypoints)]  # Wrap around for last -> first
            
#             # Find boundary segment between current and next waypoint
#             segment = self._find_boundary_segment(current_waypoint, next_waypoint, boundary_array)
            
#             # Add segment to path (avoid duplicate points)
#             if i == 0:
#                 detailed_path.extend(segment)
#             else:
#                 detailed_path.extend(segment[1:])  # Skip first point to avoid duplicate
        
#         rospy.loginfo(f"Generated detailed boundary path with {len(detailed_path)} points")
#         return detailed_path
    
#     def _find_boundary_segment(self, start_waypoint, end_waypoint, boundary_points):
#         """Find boundary points between two waypoints"""
#         # Find closest boundary points to waypoints
#         start_distances = np.linalg.norm(boundary_points - start_waypoint, axis=1)
#         end_distances = np.linalg.norm(boundary_points - end_waypoint, axis=1)
        
#         start_idx = np.argmin(start_distances)
#         end_idx = np.argmin(end_distances)
        
#         # Extract segment from boundary
#         if start_idx <= end_idx:
#             segment_indices = list(range(start_idx, end_idx + 1))
#         else:
#             # Wrap around the boundary
#             segment_indices = list(range(start_idx, len(boundary_points))) + list(range(0, end_idx + 1))
        
#         # Return boundary points for this segment
#         segment = [boundary_points[idx] for idx in segment_indices]
        
#         # If segment is too sparse, add intermediate points
#         if len(segment) < 3:
#             # Direct interpolation between waypoints
#             num_interp = max(3, int(np.linalg.norm(end_waypoint - start_waypoint) / self.boundary_resolution))
#             t_values = np.linspace(0, 1, num_interp)
#             segment = [start_waypoint + t * (end_waypoint - start_waypoint) for t in t_values]
        
#         return segment

#     def _nearest_neighbor_ordering(self, boundary_points, robot_pos):
#         """Fallback ordering using nearest neighbor algorithm"""
#         if not boundary_points:
#             return []
            
#         unvisited = list(range(len(boundary_points)))
#         ordered_points = []
#         current_pos = robot_pos
        
#         while unvisited:
#             # Find nearest unvisited point
#             distances = [np.linalg.norm(np.array(boundary_points[i]) - current_pos) for i in unvisited]
#             nearest_idx = unvisited[np.argmin(distances)]
            
#             # Add to path and update current position
#             ordered_points.append(boundary_points[nearest_idx])
#             current_pos = np.array(boundary_points[nearest_idx])
#             unvisited.remove(nearest_idx)
        
#         rospy.loginfo(f"Nearest neighbor ordering: {len(ordered_points)} points")
#         return ordered_points

#     def _publish_boundary_markers(self, boundary_loops, header):
#         """Publish boundary visualization markers"""
#         marker_array = MarkerArray()
        
#         for i, loop in enumerate(boundary_loops):
#             marker = Marker()
#             marker.header = header
#             marker.ns = "boundary_loops"
#             marker.id = i
#             marker.type = Marker.LINE_STRIP
#             marker.action = Marker.ADD
#             marker.pose.orientation.w = 1.0
#             marker.scale.x = 0.05
            
#             # Different colors for different boundaries
#             if i == 0:  # Outer boundary - red
#                 marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
#             else:  # Inner boundaries - blue
#                 marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
            
#             for point in loop:
#                 p = Point()
#                 p.x, p.y, p.z = point[0], point[1], 0.1
#                 marker.points.append(p)
                
#             # Close the loop
#             if len(loop) > 0:
#                 p = Point()
#                 p.x, p.y, p.z = loop[0][0], loop[0][1], 0.1
#                 marker.points.append(p)
                
#             marker_array.markers.append(marker)
        
#         self.boundary_markers_pub.publish(marker_array)

#     def _shrink_rectangle(self, corners: np.ndarray, margin: float) -> np.ndarray:
#         """주어진 사각형을 margin 만큼 안쪽으로 축소시킵니다."""
#         origin_point = corners[0]
#         vec_m = corners[1] - origin_point
#         vec_n = corners[3] - origin_point

#         len_m = np.linalg.norm(vec_m)
#         len_n = np.linalg.norm(vec_n)

#         # 마진이 사각형 너비/높이의 절반보다 크면 축소할 수 없으므로 원본 반환
#         if 2 * margin >= len_m or 2 * margin >= len_n:
#             rospy.logwarn("마진이 너무 커서 그리드 영역을 축소할 수 없습니다. 원본 영역을 사용합니다.")
#             return corners

#         norm_vec_m = vec_m / len_m
#         norm_vec_n = vec_n / len_n

#         # 새로운 시작점을 (마진, 마진) 만큼 안쪽으로 이동
#         new_origin = origin_point + norm_vec_m * margin + norm_vec_n * margin

#         # 새로운 너비와 높이 벡터 계산
#         new_vec_m = vec_m - 2 * norm_vec_m * margin
#         new_vec_n = vec_n - 2 * norm_vec_n * margin

#         # 축소된 사각형의 4개 모서리 계산
#         p0 = new_origin
#         p1 = new_origin + new_vec_m
#         p2 = new_origin + new_vec_m + new_vec_n
#         p3 = new_origin + new_vec_n

#         return np.array([p0, p1, p2, p3])
    
#     def _get_detailed_astar_path(self, start_world, end_world, local_grid, local_origin, resolution, height, width, header) -> List[PoseStamped]:
#         start_px = int((start_world[0] - local_origin[0]) / resolution)
#         start_py = int((start_world[1] - local_origin[1]) / resolution)
#         end_px = int((end_world[0] - local_origin[0]) / resolution)
#         end_py = int((end_world[1] - local_origin[1]) / resolution)

#         start_pixel = (start_px, height - 1 - start_py)
#         end_pixel = (end_px, height - 1 - end_py)
        
#         planner = AStarPlanner(local_grid, resolution)
#         pixel_path, _ = planner.plan(start_pixel, end_pixel)
        
#         path_poses = []
#         if pixel_path:
#             for px, py_flipped in pixel_path:
#                 py_raw = height - 1 - py_flipped
#                 wx = (px + 0.5) * resolution + local_origin[0]
#                 wy = (py_raw + 0.5) * resolution + local_origin[1]
                
#                 pose = PoseStamped(header=header)
#                 pose.pose.position.x = wx
#                 pose.pose.position.y = wy
#                 pose.pose.orientation.w = 1.0
#                 path_poses.append(pose)
        
#         return path_poses

#     def _solve_and_publish_tsp_path(self, grid_centers, header, local_grid, local_origin, resolution, height, width):
#         rospy.loginfo("ATSP 계산 시작...")

#         filtered_centers = []
#         if self.origin_filter_distance > 0:
#             for center in grid_centers:
#                 dist_to_origin = math.hypot(center[0], center[1])
#                 if dist_to_origin > self.origin_filter_distance:
#                     filtered_centers.append(center)
#             rospy.loginfo(
#                 f"원점({self.origin_filter_distance}m) 필터링: "
#                 f"{len(grid_centers)} -> {len(filtered_centers)}개 경유점"
#             )
#         else:
#             filtered_centers = grid_centers

#         if not filtered_centers:
#             rospy.logwarn("필터링 후 남은 경유점이 없어 TSP를 실행할 수 없습니다.")
#             return

#         origin_node = np.array([0.0, 0.0, 0.0])
#         # [수정] 필터링된 경유점 리스트를 사용
#         nodes = [origin_node] + filtered_centers
#         num_nodes = len(nodes)

#         temp_planner = AStarPlanner(local_grid, resolution)
#         distance_matrix = np.zeros((num_nodes, num_nodes))
#         rospy.loginfo(f"{num_nodes}개 노드에 대해 거리 행렬 계산 중...")

#         for i in range(num_nodes):
#             for j in range(i + 1, num_nodes): # <-- 여기가 핵심!
                
#                 start_px = int((nodes[i][0] - local_origin[0]) / resolution)
#                 start_py = int((nodes[i][1] - local_origin[1]) / resolution)
#                 end_px = int((nodes[j][0] - local_origin[0]) / resolution)
#                 end_py = int((nodes[j][1] - local_origin[1]) / resolution)
#                 start_pixel = (start_px, height - 1 - start_py)
#                 end_pixel = (end_px, height - 1 - end_py)
                
#                 _, path_length = temp_planner.plan(start_pixel, end_pixel)
#                 cost = path_length if path_length != float('inf') else 1e9
#                 distance_matrix[i, j] = cost
#                 distance_matrix[j, i] = cost

#         rospy.loginfo("거리 행렬 계산 완료. TSP 문제 해결 중...")
#         try:
#             permutation, total_cost = solve_tsp_dynamic_programming(distance_matrix)
#         except Exception as e:
#             rospy.logerr(f"TSP 솔버 실행 중 오류 발생: {e}")
#             return
            
#         if total_cost >= 1e9:
#             rospy.logwarn("TSP 경로를 찾을 수 없습니다.")
#             return

#         rospy.loginfo(f"TSP 해결! 최단 경로 길이: {total_cost:.2f}")

#         final_path = Path(header=header)
#         final_path.header.stamp = rospy.Time.now()
        
#         rospy.loginfo("상세 경로 생성 중...")
#         for i in range(len(permutation) - 1):
#             start_node_idx = permutation[i]
#             end_node_idx = permutation[i+1]
            
#             start_node = nodes[start_node_idx]
#             end_node = nodes[end_node_idx]
            
#             segment_poses = self._get_detailed_astar_path(
#                 start_node[:2], end_node[:2], local_grid, local_origin, 
#                 resolution, height, width, final_path.header
#             )
            
#             if final_path.poses and segment_poses:
#                 final_path.poses.extend(segment_poses[1:])
#             else:
#                 final_path.poses.extend(segment_poses)
#         final_path.poses = self.smooth_path_with_bspline(final_path, num_points_multiplier=2, s=0.5)

#         self.tsp_path_pub.publish(final_path)
#         rospy.loginfo("계산된 전체 상세 경로를 '/coverage_tsp_path' 토픽으로 발행했습니다.")

#     def _publish_local_map(self, grid_data, origin, resolution, header):
#         msg = OccupancyGrid(header=header)
#         msg.header.stamp = rospy.Time.now()
#         msg.info.resolution = resolution
#         msg.info.width = grid_data.shape[1]
#         msg.info.height = grid_data.shape[0]
#         msg.info.origin.position.x = origin[0]
#         msg.info.origin.position.y = origin[1]
#         msg.info.origin.orientation.w = 1.0
#         flipped_data = np.flipud(grid_data)
#         msg.data = flipped_data.flatten().tolist()
#         self.local_map_pub.publish(msg)

#     def _calculate_and_publish_grid_centers(self, rect_corners: np.ndarray, header) -> list:
#         origin_point = rect_corners[0]
#         vec_m = rect_corners[1] - origin_point
#         vec_n = rect_corners[3] - origin_point
#         step_m = vec_m / self.GRID_M
#         step_n = vec_n / self.GRID_N
#         projected_centers = []
#         for i in range(self.GRID_N):
#             for j in range(self.GRID_M):
#                 center_2d = origin_point + (j * step_m) + (i * step_n) + (0.5 * step_m) + (0.5 * step_n)
#                 _, index = self.kdtree.query(center_2d)
#                 actual_point_3d = self.traversable_points_3d[index]
#                 projected_centers.append(actual_point_3d)
#         marker_array = MarkerArray()
#         for i, center_3d in enumerate(projected_centers):
#             marker = Marker(header=header, ns="grid_centers", id=i, type=Marker.SPHERE, action=Marker.ADD)
#             marker.pose.position = Point(x=center_3d[0], y=center_3d[1], z=center_3d[2] + 0.1)
#             marker.pose.orientation.w = 1.0
#             marker.scale.x = marker.scale.y = marker.scale.z = self.GRID_MARKER_SCALE
#             marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
#             marker_array.markers.append(marker)
#         self.grid_marker_pub.publish(marker_array)
#         return projected_centers

#     @staticmethod
#     def find_minimum_bounding_rectangle(points: np.ndarray) -> Tuple[float, np.ndarray]:
#         try:
#             hull = ConvexHull(points)
#         except Exception as e:
#             rospy.logerr(f"Convex Hull 계산 중 오류 발생: {e}")
#             return float('inf'), None
#         hull_points = points[hull.vertices]
#         min_area, best_rectangle_corners = float('inf'), None
#         for i in range(len(hull_points)):
#             p1, p2 = hull_points[i], hull_points[(i + 1) % len(hull_points)]
#             edge_vec = p2 - p1
#             angle = np.arctan2(edge_vec[1], edge_vec[0])
#             rot_mat = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
#             rot_points = hull_points @ rot_mat.T
#             min_x, max_x = np.min(rot_points[:, 0]), np.max(rot_points[:, 0])
#             min_y, max_y = np.min(rot_points[:, 1]), np.max(rot_points[:, 1])
#             area = (max_x - min_x) * (max_y - min_y)
#             if area < min_area:
#                 min_area = area
#                 corners_rot = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
#                 inv_rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#                 best_rectangle_corners = corners_rot @ inv_rot_mat.T
#         return min_area, best_rectangle_corners

#     # def _publish_rectangle_marker(self, corners_2d: np.ndarray, header):
#     #     marker = Marker(header=header, ns="coverage_rectangle", id=0, type=Marker.LINE_STRIP, action=Marker.ADD)
#     #     marker.pose.orientation.w, marker.scale.x = 1.0, 0.05
#     #     marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
#     #     for corner in corners_2d:
#     #         marker.points.append(Point(x=corner[0], y=corner[1], z=0.0))
#     #     marker.points.append(Point(x=corners_2d[0, 0], y=corners_2d[0, 1], z=0.0))
#     #     self.rect_marker_pub.publish(marker)
#     def _publish_rectangle_marker(self, corners_2d: np.ndarray, header, ns="coverage_rectangle", id=0, color=ColorRGBA(1.0, 0.0, 0.0, 1.0)):
#         marker = Marker(header=header, ns=ns, id=id, type=Marker.LINE_STRIP, action=Marker.ADD)
#         marker.pose.orientation.w, marker.scale.x = 1.0, 0.05
#         marker.color = color
#         for corner in corners_2d:
#             marker.points.append(Point(x=corner[0], y=corner[1], z=0.0))
#         marker.points.append(Point(x=corners_2d[0, 0], y=corners_2d[0, 1], z=0.0))
#         self.rect_marker_pub.publish(marker)
    
#     def smooth_path_with_bspline(self,path_in, num_points_multiplier=2, s=0.5):
#         """
#         B-Spline을 사용하여 경로를 부드럽게 만듭니다.
#         """
#         x_coords = [pose.pose.position.x for pose in path_in.poses]
#         y_coords = [pose.pose.position.y for pose in path_in.poses]

#         if len(x_coords) < 4:
#             rospy.logwarn("경로 점 개수가 부족하여 B-Spline 스무딩을 건너뜁니다.")
#             return path_in.poses

#         tck, u = splprep([x_coords, y_coords], s=s, k=3)

#         if tck is None:
#             rospy.logerr("B-Spline 계산에 실패했습니다.")
#             return path_in.poses

#         new_num_points = int(len(x_coords) * num_points_multiplier)
#         u_new = np.linspace(u.min(), u.max(), new_num_points)
#         x_new, y_new = splev(u_new, tck)

#         smoothed_poses = []
#         for i in range(len(x_new)):
#             pose = PoseStamped()
#             pose.header = path_in.header
#             pose.pose.position.x = x_new[i]
#             pose.pose.position.y = y_new[i]
#             pose.pose.orientation.w = 1.0
#             smoothed_poses.append(pose)
            
#         return smoothed_poses

#     def smooth_path_with_moving_average(self,path_in, window_size=30):
#         """
#         이동 평균 필터로 경로를 부드럽게 만듭니다.
#         """
#         if window_size < 3:
#             # Path 객체가 아닌 poses 리스트를 반환하도록 수정!
#             return path_in.poses 

#         x = np.array([p.pose.position.x for p in path_in.poses])
#         y = np.array([p.pose.position.y for p in path_in.poses])

#         x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='same')
#         y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        
#         x_smooth[:window_size//2] = x[:window_size//2]
#         y_smooth[:window_size//2] = y[:window_size//2]
#         x_smooth[-window_size//2:] = x[-window_size//2:]
#         y_smooth[-window_size//2:] = y[-window_size//2:]

#         smoothed_poses = [] # path_out 객체 대신 바로 리스트 생성
#         for i in range(len(x_smooth)):
#             pose = PoseStamped()
#             pose.header = path_in.header
#             pose.pose.position.x = x_smooth[i]
#             pose.pose.position.y = y_smooth[i]
#             pose.pose.orientation.w = 1.0
#             smoothed_poses.append(pose)
            
#         return smoothed_poses

# if __name__ == '__main__':
#     try:
#         rospy.init_node('coverage_path_planner_detailed_path', anonymous=True)
#         CoveragePathPlanner()
        
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass





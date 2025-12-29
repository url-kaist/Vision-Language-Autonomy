#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf.transformations as tft
from threading import Lock
import struct
from collections import deque

# ROS Messages
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String


class GridMapper:
    def __init__(self):
        # --- Parameters ---
        self.fov_deg = rospy.get_param("~fov_deg", 79.0)
        self.max_dist = rospy.get_param("~max_dist", 4.0)
        self.obs_height = rospy.get_param("~obs_height", 0.2)
        self.resolution = rospy.get_param("~resolution", 0.1)
        
        # --- Frontier & Noise Filter Parameters ---
        self.cluster_min_size = rospy.get_param("~frontier/cluster_min", 20) 
        self.cluster_size_xy = rospy.get_param("~frontier/cluster_size_xy", 2.0)
        self.wall_thickness = rospy.get_param("~wall_thickness", 1)

        self.fov_rad = np.deg2rad(self.fov_deg)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.pose_init = False
        
        self.global_grid = None
        self.local_grid = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        self.map_lock = Lock()
        
        # Subscribers
        rospy.Subscriber("/global_points", PointCloud2, self._pcd_callback)
        rospy.Subscriber("/Odometry", Odometry, self._odom_callback)

        # --- Publishers ---
        self.map_pub = rospy.Publisher("/explored_map", OccupancyGrid, queue_size=1, latch=True)
        self.wall_cloud_pub = rospy.Publisher("/map_points/wall", PointCloud2, queue_size=1)
        self.free_cloud_pub = rospy.Publisher("/map_points/free", PointCloud2, queue_size=1)
        self.frontier_cloud_pub = rospy.Publisher("/map_points/frontier", PointCloud2, queue_size=1)

        self.instruction_following_pub = rospy.Publisher("/instruction_following_exp_status", String, queue_size=1, latch=False)
        
        self.fov_pub = rospy.Publisher("/fov_marker", Marker, queue_size=1)
        self.frontier_marker_pub = rospy.Publisher("/frontier_markers", MarkerArray, queue_size=1)

        rospy.Timer(rospy.Duration(0.5), self._update_loop)

        rospy.loginfo(f"Node Initialized. Wall Thickness: {self.wall_thickness}")

    def _pcd_callback(self, msg: PointCloud2):
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data))
        
        if points.size == 0:
            return

        with self.map_lock:
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            
            width = int(np.ceil((max_x - min_x) / self.resolution)) + 10
            height = int(np.ceil((max_y - min_y) / self.resolution)) + 10
            
            self.origin_x = min_x
            self.origin_y = min_y
            
            new_global_grid = np.zeros((height, width), dtype=np.int8)

            xs = ((points[:, 0] - self.origin_x) / self.resolution).astype(int)
            ys = ((points[:, 1] - self.origin_y) / self.resolution).astype(int)
            zs = points[:, 2]

            valid_mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs, ys, zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]

            obstacle_mask = zs > self.obs_height
            new_global_grid[ys[obstacle_mask], xs[obstacle_mask]] = 100
            
            # Wall Inflation
            if self.wall_thickness > 0:
                current_obstacles = (new_global_grid == 100)
                inflated_obstacles = self._inflate_mask(current_obstacles, self.wall_thickness)
                new_global_grid[inflated_obstacles] = 100

            self.global_grid = new_global_grid

            if self.local_grid is None:
                self.local_grid = np.full((height, width), -1, dtype=np.int8)
            elif self.local_grid.shape != (height, width):
                self.local_grid = np.full((height, width), -1, dtype=np.int8)

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_init = True

    def _update_loop(self, event):
        if not self.pose_init or self.global_grid is None or self.local_grid is None:
            return

        with self.map_lock:
            h, w = self.global_grid.shape
            start_angle = -self.fov_rad / 2.0
            end_angle = self.fov_rad / 2.0
            rays = np.arange(start_angle, end_angle, np.deg2rad(1.0))

            for ray_a in rays:
                global_a = self.robot_yaw + ray_a
                dx = math.cos(global_a)
                dy = math.sin(global_a)

                for dist in np.arange(0.0, self.max_dist, self.resolution):
                    wx = self.robot_x + dist * dx
                    wy = self.robot_y + dist * dy
                    
                    c = int((wx - self.origin_x) / self.resolution)
                    r = int((wy - self.origin_y) / self.resolution)
                    
                    if not (0 <= r < h and 0 <= c < w): break
                    
                    val = self.global_grid[r, c]
                    self.local_grid[r, c] = val
                    if val == 100: break
            
            # Find Frontiers (Returns: Centroids, All Points)
            frontier_centroids, frontier_points = self._find_and_cluster_frontiers()

            if len(frontier_points) == 0:
                self.instruction_following_pub.publish("no_frontier")
            else:
                print("no_frontier")
            
            self._publish_occupancy_grid()
            self._publish_fov_viz()
            self._publish_split_clouds(frontier_points)
            self._publish_frontier_markers(frontier_centroids)

    def _find_and_cluster_frontiers(self):
        if self.local_grid is None:
            return [], []

        grid = self.local_grid
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        
        u_up    = np.pad(unknown_mask[1:, :], ((0,1),(0,0)), mode='constant')
        u_down  = np.pad(unknown_mask[:-1, :], ((1,0),(0,0)), mode='constant')
        u_left  = np.pad(unknown_mask[:, 1:], ((0,0),(0,1)), mode='constant')
        u_right = np.pad(unknown_mask[:, :-1], ((0,0),(1,0)), mode='constant')
        
        has_unknown_neighbor = u_up | u_down | u_left | u_right
        frontier_mask = free_mask & has_unknown_neighbor
        
        frontier_indices = np.argwhere(frontier_mask)
        
        if len(frontier_indices) == 0:
            return [], []

        # BFS Clustering
        candidate_cells = set(map(tuple, frontier_indices))
        clusters = []
        
        while candidate_cells:
            seed = candidate_cells.pop()
            cluster = [seed]
            queue = deque([seed])
            
            while queue:
                r, c = queue.popleft()
                neighbors = [(r+1,c), (r-1,c), (r,c+1), (r,c-1),
                             (r+1,c+1), (r-1,c-1), (r+1,c-1), (r-1,c+1)]
                for nr, nc in neighbors:
                    if (nr, nc) in candidate_cells:
                        candidate_cells.remove((nr, nc))
                        cluster.append((nr, nc))
                        queue.append((nr, nc))
            
            if len(cluster) >= self.cluster_min_size:
                world_cluster = []
                for (r, c) in cluster:
                    wx = self.origin_x + (c + 0.5) * self.resolution
                    wy = self.origin_y + (r + 0.5) * self.resolution
                    world_cluster.append([wx, wy])
                clusters.append(np.array(world_cluster))

        # PCA Split & Representative Point Selection
        final_centroids = []
        for cluster in clusters:
            self._recursive_pca_split(cluster, final_centroids)
            
        all_frontier_points = []
        blue_rgb = self._pack_rgb(0, 0, 255)
        for cluster in clusters:
            for pt in cluster:
                all_frontier_points.append([pt[0], pt[1], 0.1, blue_rgb])

        return final_centroids, all_frontier_points

    def _recursive_pca_split(self, cluster_points, result_list):
        """
        Recursively splits cluster.
        [개선] 단순 평균(mean) 대신, 평균과 가장 가까운 실제 포인트를 반환합니다.
        """
        if len(cluster_points) < self.cluster_min_size:
            return

        mean = np.mean(cluster_points, axis=0)
        diffs = cluster_points - mean
        dists = np.linalg.norm(diffs, axis=1)
        
        # PCA 분할이 더 이상 필요 없는 경우 (크기가 작음)
        if np.max(dists) <= self.cluster_size_xy:
            # --- [핵심 개선] ---
            # 평균값(mean)은 허공일 수 있으므로, 평균과 가장 가까운 점(Medoid)을 찾습니다.
            min_dist_idx = np.argmin(dists)
            representative_point = cluster_points[min_dist_idx]
            result_list.append(representative_point)
            return

        # PCA 계산
        cov = np.cov(diffs.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        max_idx = np.argmax(eig_vals)
        first_pc = eig_vecs[:, max_idx]
        
        projections = np.dot(diffs, first_pc)
        
        mask1 = projections >= 0
        mask2 = projections < 0
        
        self._recursive_pca_split(cluster_points[mask1], result_list)
        self._recursive_pca_split(cluster_points[mask2], result_list)

    def _publish_split_clouds(self, frontier_points):
        if self.local_grid is None: return

        # 1. Wall Cloud
        wall_indices = np.where(self.local_grid == 100)
        wall_points = []
        red_rgb = self._pack_rgb(0, 0, 0)
        wall_z = self.obs_height
        
        for r, c in zip(wall_indices[0], wall_indices[1]):
            wx = self.origin_x + (c + 0.5) * self.resolution
            wy = self.origin_y + (r + 0.5) * self.resolution
            wall_points.append([wx, wy, wall_z, red_rgb])
            
        self._publish_pc2(self.wall_cloud_pub, wall_points)

        # 2. Free Cloud
        free_indices = np.where(self.local_grid == 0)
        free_points = []
        green_rgb = self._pack_rgb(0, 255, 0)
        
        for r, c in zip(free_indices[0], free_indices[1]):
            wx = self.origin_x + (c + 0.5) * self.resolution
            wy = self.origin_y + (r + 0.5) * self.resolution
            free_points.append([wx, wy, 0.0, green_rgb])
            
        self._publish_pc2(self.free_cloud_pub, free_points)
        
        # 3. Frontier Cloud
        self._publish_pc2(self.frontier_cloud_pub, frontier_points)
        #frontier_points를 geometry_msgs/PoseStamped: "/move_base_simple/goal"로 publish하면 됨


    def _publish_pc2(self, publisher, points_list):
        header = Header(stamp=rospy.Time.now(), frame_id="world")
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1)
        ]
        
        # [수정됨] 잔상 방지: 빈 리스트라도 발행해야 RViz에서 사라짐
        if not points_list:
            pc_msg = pc2.create_cloud(header, fields, [])
            publisher.publish(pc_msg)
            return

        pc_msg = pc2.create_cloud(header, fields, points_list)
        publisher.publish(pc_msg)

    def _pack_rgb(self, r, g, b):
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack('f', struct.pack('I', rgb))[0]

    def _inflate_mask(self, mask, radius):
        if radius <= 0: return mask
        inflated = mask.copy()
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0: continue
                shifted = np.roll(mask, shift=(i, j), axis=(0, 1))
                if i > 0: shifted[:i, :] = False
                elif i < 0: shifted[i:, :] = False
                if j > 0: shifted[:, :j] = False
                elif j < 0: shifted[:, j:] = False
                inflated |= shifted
        return inflated

    def _publish_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.info.resolution = self.resolution
        msg.info.width = self.local_grid.shape[1]
        msg.info.height = self.local_grid.shape[0]
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.local_grid.flatten().tolist()
        self.map_pub.publish(msg)

    def _publish_fov_viz(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "fov"; marker.id = 0; marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
        marker.scale.x = 0.05; marker.color.r = 1.0; marker.color.a = 1.0
        p0 = Point(self.robot_x, self.robot_y, 0.5)
        la = self.robot_yaw - self.fov_rad/2
        ra = self.robot_yaw + self.fov_rad/2
        p1 = Point(self.robot_x + self.max_dist*math.cos(la), self.robot_y + self.max_dist*math.sin(la), 0.5)
        p2 = Point(self.robot_x + self.max_dist*math.cos(ra), self.robot_y + self.max_dist*math.sin(ra), 0.5)
        marker.points = [p0, p1, p2, p0]
        self.fov_pub.publish(marker)

    def _publish_frontier_markers(self, frontier_centroids):
        marker_array = MarkerArray()
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)
        
        for i, center in enumerate(frontier_centroids):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
            marker_array.markers.append(marker)
        self.frontier_marker_pub.publish(marker_array)

if __name__ == "__main__":
    rospy.init_node("GridMapper")
    GridMapper()
    rospy.spin()
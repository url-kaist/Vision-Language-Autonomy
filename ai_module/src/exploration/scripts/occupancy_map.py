#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial import cKDTree, ConvexHull
from matplotlib.path import Path
import tf.transformations as tft
from collections import deque
import time 


class TerrainMapBuilder:
    def __init__(self):
        rospy.loginfo("Initializing TerrainMapBuilder...")

        # --- 파라미터 로드 ---
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.default_observation_radius = rospy.get_param("~observation_radius", 5.0)
        self.instruction_following_observation_radius = 10.0
        self.observation_radius = self.default_observation_radius  # Start with default
        self.check_radius = self.observation_radius + rospy.get_param("~check_radius_buffer", 0.2)
        
        # Question type tracking
        self.question_type = None
        
        # 맵의 전체 크기 (미터 단위)
        self.map_width_meters = rospy.get_param("~map_width_meters", 50.0)
        self.map_height_meters = rospy.get_param("~map_height_meters", 50.0)

        # 맵의 픽셀 단위 크기 계산
        self.map_width_pixels = int(self.map_width_meters / self.resolution)
        self.map_height_pixels = int(self.map_height_meters / self.resolution)

        # --- 수정된 부분 ---
        # 맵의 원점은 첫 오도메트리 수신 후 설정
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_initialized = False # 맵 초기화 플래그
        # -------------------

        terrain_topic = rospy.get_param("~terrain_topic", "/traversable_area")
        odom_topic = rospy.get_param("~odom_topic", "/state_estimation")
        self.map_frame = rospy.get_param("~map_frame", "map")

        # --- 전역 변수 초기화 ---
        self.terrain_points = None
        self.terrain_kdtree = None
        self.global_map_data = np.full((self.map_height_pixels, self.map_width_pixels), -1, dtype=np.int8)
        self.robot_pose = None

        # --- 발행자 및 구독자 설정 ---
        self.map_pub = rospy.Publisher("/occupancy_map", OccupancyGrid, queue_size=1, latch=True)
        
        # 지형 맵 로드
        self._load_terrain_map(terrain_topic)

        rospy.Subscriber(odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber("/question_type", String, self._question_type_callback, queue_size=1)

        rospy.loginfo("TerrainMapBuilder initialized. Waiting for first odometry to set map origin...")
        rospy.loginfo(f"Map Size: {self.map_width_pixels}x{self.map_height_pixels} pixels")
        rospy.loginfo(f"Resolution: {self.resolution} m/pixel")


    def _load_terrain_map(self, topic):
        rospy.loginfo(f"Waiting for terrain map from topic: {topic}...")
        try:
            cloud_msg = rospy.wait_for_message(topic, PointCloud2, timeout=30.0)
            points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y"), skip_nans=True)), dtype=np.float32)
            
            if points.shape[0] == 0:
                rospy.logerr("Received empty terrain point cloud!")
                rospy.signal_shutdown("Empty terrain map")
                return

            self.terrain_points = points
            self.terrain_kdtree = cKDTree(self.terrain_points)
            rospy.loginfo(f"Terrain map loaded with {len(self.terrain_points)} points.")
        except rospy.ROSException:
            rospy.logerr(f"Timeout while waiting for terrain map on topic {topic}.")
            rospy.signal_shutdown("Failed to load terrain map")

    def _question_type_callback(self, msg: String):
        """Update observation radius based on question type"""
        new_question_type = msg.data if msg.data else None
        
        if new_question_type != self.question_type:
            self.question_type = new_question_type
            
            if self.question_type == "instruction_following":
                new_observation_radius = self.instruction_following_observation_radius
                rospy.loginfo(f"Question type: {self.question_type} - Setting observation radius to {new_observation_radius}m")
            else:
                new_observation_radius = self.default_observation_radius
                rospy.loginfo(f"Question type: {self.question_type} - Setting observation radius to {new_observation_radius}m")
            
            # Update observation radius and check radius
            self.observation_radius = new_observation_radius
            self.check_radius = self.observation_radius + 0.2  # Using the buffer from parameter
            
            rospy.loginfo(f"Observation radius updated to {self.observation_radius}m")

    def _odom_callback(self, msg: Odometry):
        if not self.map_initialized:
            start_x = msg.pose.pose.position.x
            start_y = msg.pose.pose.position.y
            
            self.map_origin_x = start_x - self.map_width_meters / 2.0
            self.map_origin_y = start_y - self.map_height_meters / 2.0
            
            self.map_initialized = True
            rospy.loginfo(f"Map origin set to ({self.map_origin_x:.2f}, {self.map_origin_y:.2f}) based on initial odometry.")
        # -------------------

        self.robot_pose = msg.pose.pose
        self._update_map()

    def _world_to_map(self, wx, wy):
        """월드 좌표를 맵 픽셀 좌표로 변환"""
        if not (self.map_origin_x <= wx < self.map_origin_x + self.map_width_pixels * self.resolution and
                self.map_origin_y <= wy < self.map_origin_y + self.map_height_pixels * self.resolution):
            return None, None # 맵 범위를 벗어남

        mx = int((wx - self.map_origin_x) / self.resolution)
        my = int((wy - self.map_origin_y) / self.resolution)
        return mx, my

    # def _update_map(self):
    #     if self.robot_pose is None or self.terrain_kdtree is None:
    #         return

    #     robot_x = self.robot_pose.position.x
    #     robot_y = self.robot_pose.position.y
        
    #     local_point_indices = self.terrain_kdtree.query_ball_point([robot_x, robot_y], self.check_radius)
    #     if not local_point_indices:
    #         self._publish_map()
    #         return
        
    #     local_points = self.terrain_points[local_point_indices]

    #     # 1. 현재 관측된 모든 free cell 목록 생성
    #     free_cells = set()
    #     points_in_obs_radius_list = []
    #     for px, py in local_points:
    #         dist_sq = (px - robot_x)**2 + (py - robot_y)**2
    #         if dist_sq <= self.observation_radius**2:
    #             points_in_obs_radius_list.append([px, py])
    #             mx, my = self._world_to_map(px, py)
    #             if mx is not None:
    #                 free_cells.add((mx, my))
        
    #     # 2. 팽창(Dilation)
    #     dilated_free_cells = set()
    #     for mx, my in free_cells:
    #         for dx in [-1, 0, 1]:
    #             for dy in [-1, 0, 1]:
    #                 dilated_free_cells.add((mx + dx, my + dy))

    #     # 3. 팽창된 free_cells 목록을 기반으로 전역 맵 업데이트
    #     for mx, my in dilated_free_cells:
    #         if 0 <= my < self.map_height_pixels and 0 <= mx < self.map_width_pixels:
    #             self.global_map_data[my, mx] = 0
            
    #     # 4. 장애물 판별
    #     points_in_obs_radius = np.array(points_in_obs_radius_list)
    #     if points_in_obs_radius.shape[0] < 3:
    #         self._publish_map()
    #         return
    #     try:
    #         hull = ConvexHull(points_in_obs_radius, qhull_options='QJ') 
    #         hull_path = Path(points_in_obs_radius[hull.vertices])
    #         min_x, min_y = np.min(points_in_obs_radius, axis=0)
    #         max_x, max_y = np.max(points_in_obs_radius, axis=0)
    #         min_mx, min_my = self._world_to_map(min_x, min_y)
    #         max_mx, max_my = self._world_to_map(max_x, max_y)
            
    #         if min_mx is not None:
    #             for my in range(min_my, max_my + 1):
    #                 for mx in range(min_mx, max_mx + 1):
    #                     if (mx, my) in dilated_free_cells: continue
    #                     px = self.map_origin_x + (mx + 0.5) * self.resolution
    #                     py = self.map_origin_y + (my + 0.5) * self.resolution
    #                     if (px - robot_x)**2 + (py - robot_y)**2 > self.observation_radius**2: continue
    #                     if hull_path.contains_point((px, py)):
    #                         if self.global_map_data[my, mx] == -1:
    #                             self.global_map_data[my, mx] = 100
    #     except Exception as e:
    #         rospy.logwarn_throttle(5, f"Convex Hull calculation failed: {e}")

    #     # 5. Flood Fill로 벽 너머 영역 제거
    #     robot_mx, robot_my = self._world_to_map(robot_x, robot_y)
    #     if robot_mx is None or (robot_mx, robot_my) not in dilated_free_cells:
    #         self._publish_map()
    #         return

    #     q = deque([(robot_mx, robot_my)])
    #     reachable_cells = set([(robot_mx, robot_my)])
    #     while q:
    #         mx, my = q.popleft()
    #         for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
    #             nx, ny = mx + dx, my + dy
    #             if (nx, ny) in reachable_cells: continue
    #             if (nx, ny) in dilated_free_cells:
    #                 reachable_cells.add((nx, ny))
    #                 q.append((nx, ny))

    #     # 6. 도달할 수 없는 free cell들을 다시 unknown으로 변경
    #     occluded_cells = dilated_free_cells - reachable_cells
    #     for mx, my in occluded_cells:
    #         if 0 <= my < self.map_height_pixels and 0 <= mx < self.map_width_pixels:
    #             self.global_map_data[my, mx] = -1
        
    #     # 7. 최종 맵 발행
    #     self._publish_map()

    def _update_map(self):
        start_time = time.time()
        if self.robot_pose is None or self.terrain_kdtree is None or not self.map_initialized:
            return

        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        
        robot_mx, robot_my = self._world_to_map(robot_x, robot_y)
        if robot_mx is None:
            rospy.logwarn_throttle(5.0, "Robot is outside of the predefined map bounds!")
            return
            
        # 1. 로봇 주변의 이동 가능 지형 포인트들을 한번만 가져옵니다.
        local_point_indices = self.terrain_kdtree.query_ball_point([robot_x, robot_y], self.observation_radius)
        if not local_point_indices:
            self._publish_map()
            return
        
        local_terrain_points = self.terrain_points[local_point_indices]

        # 2. 모든 이동 가능 지형 포인트를 미리 픽셀 좌표로 변환하여 'set'에 저장합니다.
        #    'set'은 특정 요소가 포함되어 있는지 매우 빠르게 확인할 수 있습니다.
        free_pixel_set = set()
        for point in local_terrain_points:
            mx, my = self._world_to_map(point[0], point[1])
            if mx is not None:
                free_pixel_set.add((mx, my))

        # 관측 반경을 픽셀 단위로 변환
        obs_radius_pixels = int(self.observation_radius / self.resolution)

        # 3. Raycasting을 수행합니다.
        for angle in np.linspace(0, 2 * np.pi, 1080): # 광선 수를 조절하여 성능/정밀도 조절 가능 (예: 180)
            end_mx = robot_mx + int(obs_radius_pixels * np.cos(angle))
            end_my = robot_my + int(obs_radius_pixels * np.sin(angle))
            
            ray_path = self._bresenham_line(robot_mx, robot_my, end_mx, end_my)
            
            for mx, my in ray_path:
                if not (0 <= my < self.map_height_pixels and 0 <= mx < self.map_width_pixels):
                    break
                
                # 4. KDTree 검색 대신, 미리 계산된 'set'에서 빠르게 확인합니다.
                if (mx, my) in free_pixel_set:
                    self.global_map_data[my, mx] = 0  # 0: 자유 공간
                else:
                    self.global_map_data[my, mx] = 100 # 100: 장애물
                    break # 장애물을 만나면 광선 중단
        end_time = time.time()
        update_time = end_time - start_time
        # print(f"update time for cycle {update_time:.4f}" )
        self._publish_map()

    def _publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = rospy.Time.now()
        map_msg.header.frame_id = self.map_frame
        
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_pixels
        map_msg.info.height = self.map_height_pixels
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.orientation.w = 1.0

        map_msg.data = self.global_map_data.flatten().tolist()
        
        self.map_pub.publish(map_msg)

    def _bresenham_line(self, x0, y0, x1, y1):
        """두 점 사이의 모든 픽셀 좌표를 반환하는 브레즈네햄 라인 알고리즘."""
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
    
if __name__ == "__main__":
    rospy.init_node("terrain_map_builder")
    builder = TerrainMapBuilder()
    rospy.spin()

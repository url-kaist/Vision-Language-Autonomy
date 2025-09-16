#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external")
import rospy
import numpy as np
import math
import threading
import struct
from collections import deque
from typing import List

# ROS 메시지 타입
from std_msgs.msg import Float32MultiArray, ColorRGBA, Bool, String, Int16
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

# 시각화 및 변환 라이브러리
import matplotlib.cm as cm
import tf.transformations as tft

from ai_module.src.utils.logger import Logger

KNOWLEDGE_BASE = {
    "toilet": {
        # 화장실에 있을 확률이 압도적으로 높음 (엔트로피 낮음 -> w_room 높음)
        "room_probs": {
            "bathroom": 0.978, 
            "bedroom": 0.01, 
            "living_room": 0.01, 
            "kitchen": 0.001,
            "garage": 0.001
        },
        # 연관 객체: 세면대, 샤워기와의 상관관계가 매우 높음
        "context_objects": {
            "sink": 0.95, 
            "shower": 0.85, 
            "towel": 0.6
        }
    },
    "chair": {
        # 다양한 방에 나타날 수 있음 (엔트로피 높음 -> w_object 높음)
        "room_probs": {
            "living_room": 0.398, 
            "kitchen": 0.3, 
            "bedroom": 0.3, 
            "bathroom": 0.001,
            "garage": 0.001
        },
        # 테이블, 책상과의 상관관계가 매우 높음
        "context_objects": {
            "table": 0.95, 
            "desk": 0.9, 
            "couch": 0.6,
            "tv": 0.4
        }
    },
    "piano": {
        # 주로 거실에만 존재 (엔트로피 매우 낮음 -> w_room 매우 높음)
        "room_probs": {
            "living_room": 0.967,
            "bedroom": 0.03,
            "kitchen": 0.001,
            "bathroom": 0.001,
            "garage": 0.001
        },
        # 피아노 의자도 'chair'의 일종으로 간주
        "context_objects": {
            "chair": 0.9,
            "couch": 0.5,
            "potted plant": 0.3
        }
    },
    "table": {
        # 의자와 유사하게 다양한 방에 나타남 (엔트로피 높음 -> w_object 높음)
        "room_probs": {
            "kitchen": 0.498,
            "living_room": 0.4,
            "bedroom": 0.1,
            "bathroom": 0.001,
            "garage": 0.001
        },
        "context_objects": {
            "chair": 0.95,
            "couch": 0.7, # 거실의 커피 테이블
            "refrigerator": 0.5 # 주방의 식탁
        }
    }
}

class Frontier:
    def __init__(self, xyz: Point, score: float):
        self.xyz   = xyz
        self.score = score  


class VLMValueMap:
        # --- 클래스 변수 (설정값) ---
    TEXT_LABELS        = ["A bed, blanket and pillow in a warm room"]
    TARGET_LABEL       = "A bed, blanket and pillow in a warm room"
    NUM_SEGMENTS       = 4
    CAMERA_HFOV_DEG    = 360.0
    CAMERA_HFOV_RAD    = math.pi * CAMERA_HFOV_DEG / 180.0
    MARKER_SCALE_RATIO = 2.0

    def __init__(self):
        # --- 로거 및 파라미터 초기화 ---
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(quiet=quiet, prefix='VLMValueMap')
        self.min_cluster_size = rospy.get_param("~min_cluster_size", 10)
        
        # Dynamic depth parameters based on question type
        self.default_depth_max_dist = rospy.get_param("~map_depth_max", 5.0)
        self.instruction_following_depth_max_dist = 10.0
        self.depth_max_dist = self.default_depth_max_dist  # Start with default
        
        self.ray_resolution   = np.deg2rad(rospy.get_param("~map_resolution_step", 1.0))
        self.dup_R = rospy.get_param("~frontier_duplicate_radius", 1.5)
        self.visited_R = rospy.get_param("~frontier_visited_radius", 1.0)

        # Question type tracking
        self.question_type = None

        # --- 변수 초기화 ---
        self.grid_info: MapMetaData | None = None
        
        # Multi-task semantic maps
        self.num_tasks = 0
        self.task_semantic_values = []  # List of semantic value arrays, one per task
        self.task_confidences = []      # List of confidence arrays, one per task
        self.current_task_idx = 0       # Index of currently active task
        
        # Current active maps (for compatibility)
        self.semantic_value: np.ndarray | None = None
        self.confidence:     np.ndarray | None = None
        
        self.occupancy_data: np.ndarray | None = None
        self.knowledge_base = KNOWLEDGE_BASE
        self.segment_scores = [0.0] * self.NUM_SEGMENTS
        self.all_task_scores = []       # Scores for all tasks [task][segment]
        self.target_idx     = self.TEXT_LABELS.index(self.TARGET_LABEL)

        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.pose_init = False
        
        self.frontiers: List[Frontier] = []
        self.closed_frontiers: list[Point] = []
        self.map_lock = threading.Lock()
        self.map_header = None

        # --- 발행자 설정 ---
        self.value_cloud_pub = rospy.Publisher("/semantic_value_cloud", PointCloud2, queue_size=2)
        self.frontiers_viz_pub = rospy.Publisher("/frontiers_viz", MarkerArray, queue_size=2)
        self.frontiers_data_pub = rospy.Publisher("/frontiers_data", PoseArray, queue_size=2)
        self.best_marker_pub = rospy.Publisher("/best_frontier_marker", MarkerArray, queue_size=2)
        self.orientation_viz_pub = rospy.Publisher("/debug/orientation_viz", MarkerArray, queue_size=1)

        # --- 구독자 설정 ---
        rospy.Subscriber("/clip/segment_scores", Float32MultiArray, self._score_callback, queue_size=5, buff_size=2**24)
        rospy.Subscriber("/clip/all_task_scores", Float32MultiArray, self._all_task_scores_callback, queue_size=5, buff_size=2**24)
        rospy.Subscriber("/clip/reset_values", Bool, self._task_switch_callback, queue_size=1)
        rospy.Subscriber("/steps", String, self._steps_callback, queue_size=1)
        rospy.Subscriber("/current_step_idx", Int16, self._step_idx_callback, queue_size=1)
        rospy.Subscriber("/occupancy_map", OccupancyGrid, self._map_callback, queue_size=5)
        rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=5)
        rospy.Subscriber("/question_type", String, self._question_type_callback, queue_size=1)
        
        rospy.Timer(rospy.Duration(1.0), self._frontier_update_loop)
        
        self.logger.loginfo("FrontierSelector initialized.")

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_init = True
        self._publish_orientation_viz()

    def _question_type_callback(self, msg: String):
        """Update depth_max_dist based on question type"""
        new_question_type = msg.data if msg.data else None
        
        if new_question_type != self.question_type:
            self.question_type = new_question_type
            
            if self.question_type == "instruction_following":
                new_depth_max_dist = self.instruction_following_depth_max_dist
                self.logger.loginfo(f"Question type: {self.question_type} - Setting depth max distance to {new_depth_max_dist}m")
            else:
                new_depth_max_dist = self.default_depth_max_dist
                self.logger.loginfo(f"Question type: {self.question_type} - Setting depth max distance to {new_depth_max_dist}m")
            
            # Update depth max distance
            self.depth_max_dist = new_depth_max_dist
            
            self.logger.loginfo(f"Depth max distance updated to {self.depth_max_dist}m")

    def _score_callback(self, msg: Float32MultiArray):
        """빠른 콜백: CLIP 점수 수신 시 값 누적 및 시각화 즉시 실행"""
        if len(msg.data) == self.NUM_SEGMENTS:
            self.segment_scores = msg.data
            self.logger.loginfo(f"Segment Scores: {[f'{s:.3f}' for s in self.segment_scores]}")

        if self.pose_init and self.grid_info is not None:
            self._update_value_layer(self.robot_x, self.robot_y, self.robot_yaw)
            self._publish_value_cloud()

    def _map_callback(self, msg: OccupancyGrid):
        """중간 속도 콜백: 지도 메시지 수신 시 버퍼 관리 및 점유 데이터 업데이트"""
        self.map_header = msg.header
        with self.map_lock:
            new_h, new_w = msg.info.height, msg.info.width
            need_realloc = (self.grid_info is None or 
                           (self.semantic_value is not None and 
                            (new_w > self.semantic_value.shape[1] or new_h > self.semantic_value.shape[0])))
            
            if need_realloc:
                buffer_w, buffer_h = new_w + 100, new_h + 100
                old_grid_info = self.grid_info
                old_semantic_value, old_confidence = self.semantic_value, self.confidence
                old_task_semantic_values = self.task_semantic_values.copy()
                old_task_confidences = self.task_confidences.copy()
                
                self.grid_info = msg.info
                
                # Reallocate single maps
                self.semantic_value = np.zeros((buffer_h, buffer_w), dtype=np.float32)
                self.confidence = np.zeros((buffer_h, buffer_w), dtype=np.float32)
                
                # Reallocate all task maps
                self.task_semantic_values = []
                self.task_confidences = []
                for task_idx in range(self.num_tasks):
                    self.task_semantic_values.append(np.zeros((buffer_h, buffer_w), dtype=np.float32))
                    self.task_confidences.append(np.zeros((buffer_h, buffer_w), dtype=np.float32))
                
                self.logger.loginfo(f"Map buffer re-allocated to ({buffer_h}x{buffer_w}) for {self.num_tasks} tasks.")

                # Copy old data if it exists
                if old_grid_info is not None and old_semantic_value is not None:
                    dx = int(round((old_grid_info.origin.position.x - self.grid_info.origin.position.x) / self.grid_info.resolution))
                    dy = int(round((old_grid_info.origin.position.y - self.grid_info.origin.position.y) / self.grid_info.resolution))
                    old_h, old_w = old_semantic_value.shape
                    from_x_start, from_y_start = max(0, -dx), max(0, -dy)
                    to_x_start, to_y_start = max(0, dx), max(0, dy)
                    from_x_end, from_y_end = min(old_w, buffer_w - dx), min(old_h, buffer_h - dy)
                    to_x_end, to_y_end = min(buffer_w, dx + old_w), min(buffer_h, dy + old_h)
                    
                    if (from_x_end > from_x_start) and (from_y_end > from_y_start):
                        # Copy single maps
                        self.semantic_value[to_y_start:to_y_end, to_x_start:to_x_end] = old_semantic_value[from_y_start:from_y_end, from_x_start:from_x_end]
                        self.confidence[to_y_start:to_y_end, to_x_start:to_x_end] = old_confidence[from_y_start:from_y_end, from_x_start:from_x_end]
                        
                        # Copy all task maps
                        for task_idx in range(min(len(old_task_semantic_values), self.num_tasks)):
                            if task_idx < len(self.task_semantic_values):
                                self.task_semantic_values[task_idx][to_y_start:to_y_end, to_x_start:to_x_end] = old_task_semantic_values[task_idx][from_y_start:from_y_end, from_x_start:from_x_end]
                                self.task_confidences[task_idx][to_y_start:to_y_end, to_x_start:to_x_end] = old_task_confidences[task_idx][from_y_start:from_y_end, from_x_start:from_x_end]
                
                # Update active maps reference
                self._switch_active_maps()
            elif self.num_tasks > 0 and len(self.task_semantic_values) == 0:
                # Initialize task maps if we have tasks but no maps yet
                self._initialize_task_maps()
            
            self.occupancy_data = np.array(msg.data, dtype=np.int8).reshape((new_h, new_w))

    def _frontier_update_loop(self, event):
        """느린 콜백: 2초에 한 번씩 프론티어 탐색만 수행"""
        # map_header가 아직 수신되지 않았으면 아무 작업도 하지 않음
        if self.map_header is None:  # <<< 안전장치 추가
            return
            
        with self.map_lock:
            if self.occupancy_data is None or self.grid_info is None:
                return
            occupancy_data = self.occupancy_data
            grid_info = self.grid_info
        
        # ▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼
        # OccupancyGrid 메시지를 새로 만들 필요가 없어졌습니다.
        # msg = OccupancyGrid() 
        # msg.info = grid_info

        bnds = self._find_frontier(occupancy_data)
        # self.logger.loginfo(f"Found {len(bnds)} raw frontier cells")
        
        clusters = self._cluster_points(bnds)
        # self.logger.loginfo(f"Clustered into {len(clusters)} groups")
        
        clusters = self._filter_by_size(clusters, self.min_cluster_size)
        # self.logger.loginfo(f"After size filtering: {len(clusters)} clusters (min_size={self.min_cluster_size})")
        
        waypoints = self._clusters_to_waypoints(clusters, grid_info.resolution, grid_info.origin.position.x, grid_info.origin.position.y)
        # self.logger.loginfo(f"Generated {len(waypoints)} waypoints")
        
        frontiers_before = len(self.frontiers)
        self._update_frontiers(waypoints)
        # self.logger.loginfo(f"Frontiers: {frontiers_before} -> {len(self.frontiers)} after update")
        
        self._prune_visited_frontiers()
        # self.logger.loginfo(f"After pruning visited: {len(self.frontiers)} frontiers")
        
        self._prune_invalid_frontiers()
        # self.logger.loginfo(f"After pruning invalid: {len(self.frontiers)} frontiers")
        
        valid_waypoints = [f.xyz for f in self.frontiers]
        scores = np.array([f.score for f in self.frontiers], dtype=np.float32)

        # 저장해 둔 self.map_header를 직접 전달합니다.
        self._publish_frontiers_as_pose_array(valid_waypoints, scores, self.map_header)
        self._publish_frontiers_for_viz(valid_waypoints, scores, self.map_header)

        if len(valid_waypoints) > 0:
            best_i = int(np.argmax(scores)) if scores.size > 0 else 0
            if best_i < len(valid_waypoints):
                # 저장해 둔 self.map_header를 직접 전달합니다.
                self._publish_best_frontier(valid_waypoints[best_i], self.map_header)

    def _update_value_layer(self, cam_x: float, cam_y: float, yaw: float):
        if self.grid_info is None or self.occupancy_data is None: return
        
        with self.map_lock:
            val, conf = self.semantic_value, self.confidence
            res, ox, oy = self.grid_info.resolution, self.grid_info.origin.position.x, self.grid_info.origin.position.y
            buffer_h, buffer_w = val.shape
            map_h, map_w = self.occupancy_data.shape
            
            seg_w = self.CAMERA_HFOV_RAD / self.NUM_SEGMENTS
            rays = np.arange(-self.CAMERA_HFOV_RAD / 2.0, self.CAMERA_HFOV_RAD / 2.0, self.ray_resolution)
            segment_fov_half = (self.CAMERA_HFOV_RAD / self.NUM_SEGMENTS) / 2.0
            for a_local in rays:
                angle_deg = math.degrees(a_local)
                
                # 1. 광선(ray)이 속한 세그먼트의 인덱스와 중심 각도를 결정합니다.
                if -45 <= angle_deg < 45:
                    seg_idx = 1
                    center_angle = 0.0
                elif 45 <= angle_deg < 135:
                    seg_idx = 2
                    center_angle = math.pi / 2.0
                elif -135 <= angle_deg < -45:
                    seg_idx = 0
                    center_angle = -math.pi / 2.0
                else:
                    seg_idx = 3
                    center_angle = math.pi
                
                # 2. VLFM 신뢰도 공식 적용
                # theta: 현재 광선과 세그먼트 중심 사이의 각도 차이
                theta = a_local - center_angle
                # 후방 세그먼트의 각도 التفاف 처리 (-pi 와 pi는 동일)
                if abs(theta) > math.pi:
                    theta = (2 * math.pi) - abs(theta)

                # c_curr: 현재 신뢰도. 세그먼트 중앙에서 1, 가장자리에서 0
                # cos^2( (theta / 45도) * (pi / 2) )
                cos_arg = (theta / segment_fov_half) * (math.pi / 2.0)
                c_curr = math.cos(cos_arg)**2

                # 3. v_curr (현재 값)은 기존처럼 세그먼트의 대표 점수를 사용
                v_curr = self.segment_scores[seg_idx]
                
                # 4. 가중 평균 (이하 로직은 동일)
                a_global = yaw + a_local
                dv = np.array([np.cos(a_global), np.sin(a_global)])

                for dist in np.arange(0.0, self.depth_max_dist, res):
                    gx, gy = cam_x + dist * dv[0], cam_y + dist * dv[1]
                    gi, gj = int((gy - oy) / res), int((gx - ox) / res)

                    if not (0 <= gi < buffer_h and 0 <= gj < buffer_w): break
                    if not (0 <= gi < map_h and 0 <= gj < map_w): continue
                    
                    if self.occupancy_data[gi, gj] == 0:
                        v_prev, c_prev = val[gi, gj], conf[gi, gj]
                        if c_prev == 0.0: 
                            v_new, c_new = v_curr, c_curr
                        else:
                            denominator = c_curr + c_prev
                            if denominator > 1e-6:
                                # 논문의 가치 및 신뢰도 업데이트 공식
                                v_new = (c_curr * v_curr + c_prev * v_prev) / denominator
                                c_new = (c_curr**2 + c_prev**2) / denominator
                            else: 
                                v_new, c_new = v_prev, c_prev
                        val[gi, gj], conf[gi, gj] = v_new, c_new
                    elif self.occupancy_data[gi, gj] == 100:
                        break

    def _publish_value_cloud(self):
        if self.grid_info is None or self.occupancy_data is None: return
        with self.map_lock:
            h, w = self.occupancy_data.shape
            semantic_value_view = self.semantic_value[:h, :w]
            occupancy_data_copy = self.occupancy_data
        
        free_mask = (occupancy_data_copy == 0)
        value_mask = (semantic_value_view > 0.01)
        semantic_rows, semantic_cols = np.where(free_mask & value_mask)
        obstacle_rows, obstacle_cols = np.where(occupancy_data_copy == 100)
        
        if semantic_rows.size == 0 and obstacle_rows.size == 0: return

        points = []
        res = self.grid_info.resolution
        ox, oy = self.grid_info.origin.position.x, self.grid_info.origin.position.y
        
        if semantic_rows.size > 0:
            values = semantic_value_view[semantic_rows, semantic_cols]
            v_max = np.max(values) if np.any(values > 0) else 1.0
            colormap = cm.get_cmap('inferno')

            for i, (r, c) in enumerate(zip(semantic_rows, semantic_cols)):
                x, y, z = ox + (c + 0.5) * res, oy + (r + 0.5) * res, 0.0
                color = colormap(values[i] / v_max)
                r_int, g_int, b_int = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                rgb = struct.unpack('I', struct.pack('BBBB', b_int, g_int, r_int, 255))[0]
                packed_rgb = struct.unpack('f', struct.pack('I', rgb))[0]
                points.append([x, y, z, packed_rgb])
        
        if obstacle_rows.size > 0:
            black_rgb = struct.unpack('I', struct.pack('BBBB', 0, 0, 0, 255))[0]
            black_packed_rgb = struct.unpack('f', struct.pack('I', black_rgb))[0]
            for r, c in zip(obstacle_rows, obstacle_cols):
                x, y, z = ox + (c + 0.5) * res, oy + (r + 0.5) * res, 0.0
                points.append([x, y, z, black_packed_rgb])
            
        header = Header(stamp=rospy.Time.now(), frame_id='map')
        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), PointField('rgb', 12, PointField.FLOAT32, 1)]
        cloud_msg = pc2.create_cloud(header, fields, points)
        self.value_cloud_pub.publish(cloud_msg)

    def _update_frontiers(self, new_waypoints: List[Point]):
        for f in self.frontiers:
            f.score = self._cell_value(f.xyz)

        for waypoint in new_waypoints:
            is_duplicate = any(self._points_close(f.xyz, waypoint, self.dup_R) for f in self.frontiers)
            if is_duplicate:
                continue
            
            is_closed = any(self._points_close(waypoint, cf, self.dup_R) for cf in self.closed_frontiers)
            if is_closed:
                continue

            self.frontiers.append(Frontier(waypoint, self._cell_value(waypoint)))
        self.frontiers.sort(key=lambda f: f.score, reverse=True)

    def _publish_frontiers_as_pose_array(self, waypoints, scores, header):
        pose_array_msg = PoseArray(header=header)
        for waypoint, score in zip(waypoints, scores):
            pose = Pose()
            pose.position.x = waypoint.x
            pose.position.y = waypoint.y
            pose.position.z = score + 3
            pose.orientation.w = 1.0
            pose_array_msg.poses.append(pose)
        self.frontiers_data_pub.publish(pose_array_msg)

    def _publish_frontiers_for_viz(self, waypoints, scores, header):
        marker_array = self._make_markers(waypoints, scores,
                                           header,
                                           self.grid_info.resolution * self.MARKER_SCALE_RATIO,
                                           ColorRGBA(1, 1, 0, 0.6))
        
        delete_all_marker = Marker(action=Marker.DELETEALL, header=header)
        self.frontiers_viz_pub.publish(MarkerArray(markers=[delete_all_marker]))
        self.frontiers_viz_pub.publish(marker_array)

    def _prune_visited_frontiers(self):
        still_open = []
        for f in self.frontiers:
            if self._points_close_xyz(f.xyz, self.robot_x, self.robot_y, self.visited_R):
                self.closed_frontiers.append(f.xyz)
            else:
                still_open.append(f)
        self.frontiers = still_open

    def _points_close_xyz(self,pt: Point, x: float, y: float, thresh: float) -> bool:
        return math.hypot(pt.x - x, pt.y - y) < thresh

    def _cell_value(self, pt: Point) -> float:
        if self.semantic_value is None:
            return 0.0
        res, ox, oy = self.grid_info.resolution, self.grid_info.origin.position.x, self.grid_info.origin.position.y
        gi = int((pt.y - oy) / res)
        gj = int((pt.x - ox) / res)
        if 0 <= gi < self.semantic_value.shape[0] and 0 <= gj < self.semantic_value.shape[1]:
            return float(self.semantic_value[gi, gj])
        return 0.0

    def _points_close(self,p1: Point, p2: Point, thresh: float) -> bool:
        return math.hypot(p1.x - p2.x, p1.y - p2.y) < thresh

    def _find_frontier(self, arr: np.ndarray):
        h, w = arr.shape
        bnds = []
        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0: continue
                
                is_frontier = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] == -1:
                        is_frontier = True
                        break
                
                if is_frontier:
                    bnds.append((i, j))
        return bnds

    def _cluster_points(self, points):
        pts = set(points)
        visited = set()
        clusters = []
        for p in points:
            if p in visited: continue
            
            q = deque([p])
            visited.add(p)
            current_cluster = []
            while q:
                ci, cj = q.popleft()
                current_cluster.append((ci, cj))
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0: continue
                        ni, nj = ci + di, cj + dj
                        if (ni, nj) in pts and (ni, nj) not in visited:
                            visited.add((ni, nj))
                            q.append((ni, nj))
            clusters.append(current_cluster)
        return clusters

    def _filter_by_size(self, clusters, min_sz):
        return [c for c in clusters if len(c) >= min_sz]

    def _clusters_to_waypoints(self, clusters, res, ox, oy):
        waypoints = []
        for c in clusters:
            # Use PCA to find representative frontier points instead of centroid
            pca_waypoints = self._pca_frontier_selection(c, res, ox, oy)
            waypoints.extend(pca_waypoints)
        return waypoints
    
    def _pca_frontier_selection(self, cluster, res, ox, oy):
        """
        Use PCA to select representative frontier points that are guaranteed to be actual frontier points.
        Returns 1-3 representative points per cluster based on cluster size and shape.
        """
        if len(cluster) < 2:
            # For very small clusters, just convert the single point
            ci, cj = cluster[0]
            x = ox + (cj + 0.5) * res
            y = oy + (ci + 0.5) * res
            return [Point(x, y, 0.0)]
        
        # Convert cluster points to numpy array for PCA
        points = np.array(cluster, dtype=np.float32)
        
        # Calculate mean and center the data
        mean_point = np.mean(points, axis=0)
        centered_points = points - mean_point
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Handle edge case where all points are identical (covariance is zero)
        if np.allclose(cov_matrix, 0):
            # All points are the same, just return one
            ci, cj = cluster[0]
            x = ox + (cj + 0.5) * res
            y = oy + (ci + 0.5) * res
            return [Point(x, y, 0.0)]
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending) to get principal components
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # First principal component (direction of maximum variation)
        pc1 = eigenvectors[:, 0]
        
        # Project all points onto the first principal component
        projections = np.dot(centered_points, pc1)
        
        # Select representative points based on cluster size
        representative_points = []
        
        if len(cluster) <= 15:
            # Small cluster: select just one point (closest to center along PC1)
            median_projection = np.median(projections)
            median_idx = np.argmin(np.abs(projections - median_projection))
            representative_points = [cluster[median_idx]]
                
        elif len(cluster) <= 30:
            # Medium cluster: select just the middle point
            median_projection = np.median(projections)
            median_idx = np.argmin(np.abs(projections - median_projection))
            representative_points = [cluster[median_idx]]
            
        else:
            # Large cluster: select quartile points for better coverage
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)
            
            # Find quartile points
            q1_projection = np.percentile(projections, 25)
            q3_projection = np.percentile(projections, 75)
            
            q1_idx = np.argmin(np.abs(projections - q1_projection))
            q3_idx = np.argmin(np.abs(projections - q3_projection))
            
            # Collect unique indices
            indices = list(set([min_idx, q1_idx, q3_idx, max_idx]))
            representative_points = [cluster[idx] for idx in indices]
        
        # Convert grid coordinates to world coordinates
        waypoints = []
        for ci, cj in representative_points:
            x = ox + (cj + 0.5) * res
            y = oy + (ci + 0.5) * res
            waypoints.append(Point(x, y, 0.0))
        
        return waypoints

    def _publish_best_frontier(self, best_wp, header):
        delete_marker = Marker(header=header, action=Marker.DELETEALL)
        self.best_marker_pub.publish(MarkerArray(markers=[delete_marker]))

        scores = np.array([self._cell_value(best_wp)], dtype=np.float32)
        marker_array = self._make_markers([best_wp], scores, header,
                                          self.grid_info.resolution,
                                          ColorRGBA(1,0,0,1))
        self.best_marker_pub.publish(marker_array)

    def _publish_orientation_viz(self):
        """
        로봇의 방향(화살표)과 360도 세그먼트별 점수를 색상으로 표현하는 원형 시야각을 시각화합니다.
        """
        if not self.pose_init or self.map_header is None:
            return

        marker_array = MarkerArray()

        # --- 1. 로봇 방향을 나타내는 파란색 화살표 (이전과 동일) ---
        arrow = Marker()
        arrow.header = self.map_header
        arrow.ns = "debug_orientation"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        # ... (이전 답변의 화살표 설정 코드와 동일하게 유지) ...
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.0, 0.5, 1.0, 0.9
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.8, 0.1, 0.15
        start_point = Point(x=self.robot_x, y=self.robot_y, z=0.1)
        end_point = Point(x=self.robot_x + arrow.scale.x * math.cos(self.robot_yaw),
                        y=self.robot_y + arrow.scale.x * math.sin(self.robot_yaw), z=0.1)
        arrow.points.extend([start_point, end_point])
        marker_array.markers.append(arrow)

        # --- 2. 360도 분할 시야각 (TRIANGLE_LIST) ---
        fov_segments = Marker()
        fov_segments.header = self.map_header
        fov_segments.ns = "debug_fov_segments"
        fov_segments.id = 0
        fov_segments.type = Marker.TRIANGLE_LIST
        fov_segments.action = Marker.ADD
        fov_segments.scale.x = 1.0
        fov_segments.scale.y = 1.0
        fov_segments.scale.z = 1.0

        # 색상 맵 ('viridis'는 점수가 낮으면 보라색, 높으면 노란색으로 표현)
        colormap = cm.get_cmap('viridis')
        
        fov_range = 1.5  # 원의 반지름
        segment_angle = self.CAMERA_HFOV_RAD / self.NUM_SEGMENTS
        
        # 로봇의 정면(yaw)을 기준으로 360도를 NUM_SEGMENTS 만큼 나눔
        for i in range(self.NUM_SEGMENTS):
            # 각 세그먼트의 시작 각도와 끝 각도 계산
            # VLMValueMap의 점수 계산 로직에 맞춰, 로봇 후방(-180도)부터 시작
            start_angle = self.robot_yaw - (self.CAMERA_HFOV_RAD / 2.0) + (i * segment_angle)
            end_angle = start_angle + segment_angle

            # 삼각형의 세 꼭짓점
            p1 = Point(x=self.robot_x, y=self.robot_y, z=0.05)
            p2 = Point(x=self.robot_x + fov_range * math.cos(start_angle),
                    y=self.robot_y + fov_range * math.sin(start_angle), z=0.05)
            p3 = Point(x=self.robot_x + fov_range * math.cos(end_angle),
                    y=self.robot_y + fov_range * math.sin(end_angle), z=0.05)
            
            fov_segments.points.extend([p1, p2, p3])

            # 현재 세그먼트의 점수를 가져와 0~1 사이로 정규화 (점수 범위가 다르다면 정규화 방식 수정 필요)
            score = self.segment_scores[i] if self.segment_scores else 0.0
            
            # 점수를 색상으로 변환 (RGBA)
            color = colormap(score)
            marker_color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.5) # 반투명
            
            # 세 꼭짓점에 동일한 색상 적용
            fov_segments.colors.extend([marker_color, marker_color, marker_color])

        marker_array.markers.append(fov_segments)
        
        self.orientation_viz_pub.publish(marker_array)

    def _make_markers(self, pts, scores, header, base_scale, color):
        ma = MarkerArray()
        if scores.size == 0: return ma
        
        max_score = np.max(scores) if np.any(scores > 0) else 1.0
        for i, p in enumerate(pts):
            # --- 원기둥(CYLINDER) 마커 ---
            m = Marker(header=header, ns="frontier", id=i, type=Marker.CYLINDER, action=Marker.ADD)
            m.pose.position = p
            
            # 원기둥의 높이를 1.0으로 설정하고, 그 절반만큼 Z 위치를 올려 바닥에 붙도록 함
            cylinder_height = 0.3
            m.pose.position.z = cylinder_height / 2.0 
            m.pose.orientation = Quaternion(0,0,0,1)
            
            # X, Y는 원기둥의 지름, Z는 높이가 됩니다.
            score_scale_factor = 0.5 + (scores[i] / max_score if max_score > 0 else 1.0)
            m.scale.x = base_scale * score_scale_factor
            m.scale.y = base_scale * score_scale_factor
            m.scale.z = cylinder_height
            
            m.color = color
            m.lifetime = rospy.Duration(0)
            ma.markers.append(m)

            # --- 텍스트 마커 ---
            t = Marker(header=header, ns="frontier_text", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position = Point(p.x, p.y, cylinder_height + 0.2)
            t.pose.orientation.w = 1.0
            t.scale.z = base_scale * 0.8
            t.color = ColorRGBA(1,1,1,1)
            t.text = f"{scores[i]:.2f}"
            t.lifetime = rospy.Duration(0)
            ma.markers.append(t)
        return ma

    def _prune_invalid_frontiers(self):
        """
        현재 OccupancyGrid 맵을 기준으로 더 이상 유효하지 않은 프론티어를 제거합니다.
        좌표 변환 오차에 강인하도록 '점'이 아닌 '영역'을 검사합니다.
        """
        if not self.frontiers:
            return

        valid_frontiers = []
        res = self.grid_info.resolution
        ox, oy = self.grid_info.origin.position.x, self.grid_info.origin.position.y
        map_h, map_w = self.occupancy_data.shape

        # 프론티어 좌표 주변을 탐색할 반경 (셀 단위). 1이면 3x3, 2면 5x5 영역을 탐색.
        search_radius = 2

        for f in self.frontiers:
            # 프론티어의 월드 좌표를 기준 그리드 인덱스로 변환
            center_gi = int((f.xyz.y - oy) / res)
            center_gj = int((f.xyz.x - ox) / res)
            
            is_still_valid = False
            # 중심점 주변의 작은 영역을 탐색
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    gi, gj = center_gi + di, center_gj + dj

                    # 맵 범위를 벗어나는지 확인
                    if not (0 <= gi < map_h and 0 <= gj < map_w):
                        continue

                    # 1. 해당 셀이 '자유 공간(0)'인지 확인
                    if self.occupancy_data[gi, gj] == 0:
                        # 2. 주변 8방향에 '미지 공간(-1)'이 있는지 확인
                        has_unknown_neighbor = False
                        for ni_offset in range(-1, 2):
                            for nj_offset in range(-1, 2):
                                if ni_offset == 0 and nj_offset == 0: continue
                                
                                ni, nj = gi + ni_offset, gj + nj_offset
                                if 0 <= ni < map_h and 0 <= nj < map_w and self.occupancy_data[ni, nj] == -1:
                                    has_unknown_neighbor = True
                                    break
                            if has_unknown_neighbor: break
                        
                        # 두 조건을 모두 만족하는 셀을 찾으면, 이 프론티어는 유효함
                        if has_unknown_neighbor:
                            is_still_valid = True
                            break

                    # 1. 해당 셀이 '미지 공간(-1)'인지 확인
                    if self.occupancy_data[gi, gj] == -1:
                        # 2. 주변 8방향에 '자유 공간(0)'이 있는지 확인
                        has_free_neighbor = False
                        for ni_offset in range(-1, 2):
                            for nj_offset in range(-1, 2):
                                if ni_offset == 0 and nj_offset == 0: continue
                                
                                ni, nj = gi + ni_offset, gj + nj_offset
                                if 0 <= ni < map_h and 0 <= nj < map_w and self.occupancy_data[ni, nj] == 0:
                                    has_free_neighbor = True
                                    break
                            if has_free_neighbor: break
                        
                        # 두 조건을 모두 만족하는 셀을 찾으면, 이 프론티어는 유효함
                        if has_free_neighbor:
                            is_still_valid = True
                            break

                if is_still_valid:
                    break
            
            # 유효성이 확인된 프론티어만 목록에 추가
            if is_still_valid:
                valid_frontiers.append(f)

        num_pruned = len(self.frontiers) - len(valid_frontiers)
        if num_pruned > 0:
            self.logger.loginfo(f"Pruned {num_pruned} invalid (phantom) frontiers.")
        
        self.frontiers = valid_frontiers

    def _steps_callback(self, msg: String):
        """Initialize multiple task maps when steps are received"""
        if not msg.data:
            return
            
        steps = msg.data.split("/")
        new_num_tasks = len(steps)
        
        if new_num_tasks != self.num_tasks:
            self.logger.loginfo(f"Initializing {new_num_tasks} task semantic maps")
            self.num_tasks = new_num_tasks
            
            # Initialize task-specific maps if we have grid info
            if self.grid_info is not None:
                self._initialize_task_maps()
            else:
                # If no grid info yet, just prepare the task count
                self.logger.loginfo(f"Grid info not available yet - will initialize {new_num_tasks} task maps when occupancy grid is received")
                # Clear existing maps to force re-initialization
                self.task_semantic_values = []
                self.task_confidences = []
            
            self.logger.loginfo(f"Task maps will be initialized for: {steps}")

    def _step_idx_callback(self, msg: Int16):
        """Switch to the specified task map"""
        new_idx = msg.data
        
        self.logger.loginfo(f"=== STEP INDEX CALLBACK ===")
        self.logger.loginfo(f"Received step index: {new_idx}")
        self.logger.loginfo(f"Current task index: {self.current_task_idx}")
        self.logger.loginfo(f"Number of tasks: {self.num_tasks}")
        self.logger.loginfo(f"Task maps initialized: {len(self.task_semantic_values)}")
        
        if 0 <= new_idx < self.num_tasks:
            if new_idx != self.current_task_idx:
                self.logger.loginfo(f"Switching from task {self.current_task_idx} to task {new_idx}")
                self.current_task_idx = new_idx
                self._switch_active_maps()
            else:
                self.logger.loginfo(f"Already on task {new_idx} - no switch needed")
        else:
            self.logger.logwarn(f"Invalid task index {new_idx} for {self.num_tasks} tasks")
            
        self.logger.loginfo(f"=== END STEP INDEX CALLBACK ===")

    def _task_switch_callback(self, msg: Bool):
        """Handle task switching (replaces reset callback)"""
        # msg.data = False means switch task (don't reset)
        # This is triggered from CLIP node when step_idx changes
        self.logger.loginfo(f"Task switch signal received: {msg.data}")
        self.logger.loginfo(f"Current task index: {self.current_task_idx}")
        self.logger.loginfo(f"Active semantic value shape: {self.semantic_value.shape if self.semantic_value is not None else 'None'}")
        self.logger.loginfo(f"Task maps available: {len(self.task_semantic_values)}")
        
    def _all_task_scores_callback(self, msg: Float32MultiArray):
        """Receive scores for all tasks simultaneously"""
        if not msg.layout.dim or len(msg.layout.dim) < 2:
            self.logger.logwarn("Invalid scores message layout")
            return
            
        num_tasks = msg.layout.dim[0].size
        num_segments = msg.layout.dim[1].size
        
        self.logger.loginfo(f"=== RECEIVED ALL TASK SCORES ===")
        self.logger.loginfo(f"Message contains {num_tasks} tasks, {num_segments} segments")
        self.logger.loginfo(f"Expected tasks: {self.num_tasks}")
        self.logger.loginfo(f"Data length: {len(msg.data)}")
        
        if num_tasks != self.num_tasks:
            self.logger.logwarn(f"Received scores for {num_tasks} tasks but have {self.num_tasks} initialized")
            return
            
        # Parse the flattened data back to [task][segment] format
        self.all_task_scores = []
        for task_idx in range(num_tasks):
            task_scores = []
            for seg_idx in range(num_segments):
                data_idx = task_idx * num_segments + seg_idx
                if data_idx < len(msg.data):
                    task_scores.append(msg.data[data_idx])
                else:
                    task_scores.append(0.0)
            self.all_task_scores.append(task_scores)
            self.logger.loginfo(f"Task {task_idx} scores: {task_scores}")
        
        self.logger.loginfo(f"=== UPDATING ALL TASK VALUE LAYERS ===")
        # Update all task maps simultaneously
        self._update_all_task_value_layers()
        self.logger.loginfo(f"=== FINISHED UPDATING ALL TASK VALUE LAYERS ===")
        
        # Also update the single-task scores for backward compatibility
        if self.current_task_idx < len(self.all_task_scores):
            self.segment_scores = self.all_task_scores[self.current_task_idx]
            self.logger.loginfo(f"Updated current task ({self.current_task_idx}) scores: {self.segment_scores}")
        
        self.logger.loginfo(f"=== END ALL TASK SCORES PROCESSING ===")
        
        # Note: _update_all_task_value_layers() already handles updating the semantic maps
        # No need to call _update_value_layer() again

    def _reset_values_callback(self, msg: Bool):
        """Legacy callback - kept for compatibility"""
        self._task_switch_callback(msg)
        
    def _initialize_task_maps(self):
        """Initialize semantic value and confidence maps for all tasks"""
        if self.grid_info is None:
            self.logger.logwarn("Cannot initialize task maps - grid_info is None")
            return
            
        if self.num_tasks == 0:
            self.logger.logwarn("Cannot initialize task maps - num_tasks is 0")
            return
            
        # Determine buffer size from existing semantic_value or use default
        if self.semantic_value is not None:
            buffer_h, buffer_w = self.semantic_value.shape
        else:
            # Use grid size + buffer if no existing semantic_value
            buffer_h = self.grid_info.height + 100
            buffer_w = self.grid_info.width + 100
            # Also initialize the main semantic maps
            self.semantic_value = np.zeros((buffer_h, buffer_w), dtype=np.float32)
            self.confidence = np.zeros((buffer_h, buffer_w), dtype=np.float32)
        
        # Initialize arrays for each task
        self.task_semantic_values = []
        self.task_confidences = []
        
        for task_idx in range(self.num_tasks):
            semantic_map = np.zeros((buffer_h, buffer_w), dtype=np.float32)
            confidence_map = np.zeros((buffer_h, buffer_w), dtype=np.float32)
            
            self.task_semantic_values.append(semantic_map)
            self.task_confidences.append(confidence_map)
        
        # Set the active maps to the current task
        self._switch_active_maps()
        
        self.logger.loginfo(f"Successfully initialized {self.num_tasks} task maps with size ({buffer_h}x{buffer_w})")
        self.logger.loginfo(f"Current task index: {self.current_task_idx}")
        self.logger.loginfo(f"Task semantic values length: {len(self.task_semantic_values)}")
        self.logger.loginfo(f"Task confidences length: {len(self.task_confidences)}")
    
    def _switch_active_maps(self):
        """Switch the active semantic_value and confidence to the current task"""
        self.logger.loginfo(f"=== SWITCHING ACTIVE MAPS ===")
        self.logger.loginfo(f"Current task index: {self.current_task_idx}")
        self.logger.loginfo(f"Available task semantic values: {len(self.task_semantic_values)}")
        self.logger.loginfo(f"Available task confidences: {len(self.task_confidences)}")
        
        if (self.current_task_idx < len(self.task_semantic_values) and 
            self.current_task_idx < len(self.task_confidences)):
            
            old_semantic_shape = self.semantic_value.shape if self.semantic_value is not None else None
            old_confidence_shape = self.confidence.shape if self.confidence is not None else None
            
            self.semantic_value = self.task_semantic_values[self.current_task_idx]
            self.confidence = self.task_confidences[self.current_task_idx]
            
            self.logger.loginfo(f"Successfully switched to task {self.current_task_idx} semantic map")
            self.logger.loginfo(f"Old semantic shape: {old_semantic_shape}")
            self.logger.loginfo(f"New semantic shape: {self.semantic_value.shape}")
            self.logger.loginfo(f"New confidence shape: {self.confidence.shape}")
        else:
            self.logger.logwarn(f"Cannot switch to task {self.current_task_idx} - maps not initialized")
            self.logger.logwarn(f"Available semantic maps: {len(self.task_semantic_values)}")
            self.logger.logwarn(f"Available confidence maps: {len(self.task_confidences)}")
            
        self.logger.loginfo(f"=== END SWITCHING ACTIVE MAPS ===")
    
    def _update_all_task_value_layers(self):
        """Update value layers for all tasks simultaneously"""
        self.logger.loginfo(f"=== _UPDATE_ALL_TASK_VALUE_LAYERS CALLED ===")
        self.logger.loginfo(f"Grid info: {self.grid_info is not None}")
        self.logger.loginfo(f"Occupancy data: {self.occupancy_data is not None}")
        self.logger.loginfo(f"All task scores length: {len(self.all_task_scores)}")
        self.logger.loginfo(f"Number of tasks: {self.num_tasks}")
        self.logger.loginfo(f"Pose initialized: {self.pose_init}")
        
        if (self.grid_info is None or self.occupancy_data is None or 
            len(self.all_task_scores) != self.num_tasks):
            self.logger.logwarn("Blocking condition met - cannot update task value layers:")
            self.logger.logwarn(f"  - grid_info is None: {self.grid_info is None}")
            self.logger.logwarn(f"  - occupancy_data is None: {self.occupancy_data is None}")
            self.logger.logwarn(f"  - all_task_scores length ({len(self.all_task_scores)}) != num_tasks ({self.num_tasks})")
            return
            
        if not self.pose_init:
            self.logger.logwarn("Pose not initialized - cannot update task value layers")
            return
            
        self.logger.loginfo("All conditions met - proceeding with task value layer update")
        
        cam_x, cam_y, yaw = self.robot_x, self.robot_y, self.robot_yaw
        
        with self.map_lock:
            res, ox, oy = self.grid_info.resolution, self.grid_info.origin.position.x, self.grid_info.origin.position.y
            map_h, map_w = self.occupancy_data.shape
            
            seg_w = self.CAMERA_HFOV_RAD / self.NUM_SEGMENTS
            rays = np.arange(-self.CAMERA_HFOV_RAD / 2.0, self.CAMERA_HFOV_RAD / 2.0, self.ray_resolution)
            segment_fov_half = (self.CAMERA_HFOV_RAD / self.NUM_SEGMENTS) / 2.0
            
            for a_local in rays:
                angle_deg = math.degrees(a_local)
                
                # Determine segment index and center angle
                if -45 <= angle_deg < 45:
                    seg_idx = 1
                    center_angle = 0.0
                elif 45 <= angle_deg < 135:
                    seg_idx = 2
                    center_angle = math.pi / 2.0
                elif -135 <= angle_deg < -45:
                    seg_idx = 0
                    center_angle = -math.pi / 2.0
                else:
                    seg_idx = 3
                    center_angle = math.pi
                
                # Calculate confidence
                theta = a_local - center_angle
                if abs(theta) > math.pi:
                    theta = (2 * math.pi) - abs(theta)
                
                cos_arg = (theta / segment_fov_half) * (math.pi / 2.0)
                c_curr = math.cos(cos_arg)**2
                
                # Cast ray and update all task maps
                a_global = yaw + a_local
                dv = np.array([np.cos(a_global), np.sin(a_global)])

                for dist in np.arange(0.0, self.depth_max_dist, res):
                    gx, gy = cam_x + dist * dv[0], cam_y + dist * dv[1]
                    gi, gj = int((gy - oy) / res), int((gx - ox) / res)

                    if not (0 <= gi < map_h and 0 <= gj < map_w): break
                    if not (0 <= gi < self.task_semantic_values[0].shape[0] and 
                            0 <= gj < self.task_semantic_values[0].shape[1]): continue
                    
                    if self.occupancy_data[gi, gj] == 0:
                        # Update all task maps simultaneously
                        for task_idx in range(self.num_tasks):
                            if task_idx >= len(self.all_task_scores): continue
                            if seg_idx >= len(self.all_task_scores[task_idx]): continue
                            
                            v_curr = self.all_task_scores[task_idx][seg_idx]
                            val = self.task_semantic_values[task_idx]
                            conf = self.task_confidences[task_idx]
                            
                            v_prev, c_prev = val[gi, gj], conf[gi, gj]
                            if c_prev == 0.0: 
                                v_new, c_new = v_curr, c_curr
                            else:
                                denominator = c_curr + c_prev
                                if denominator > 1e-6:
                                    v_new = (c_curr * v_curr + c_prev * v_prev) / denominator
                                    c_new = (c_curr**2 + c_prev**2) / denominator
                                else: 
                                    v_new, c_new = v_prev, c_prev
                            val[gi, gj], conf[gi, gj] = v_new, c_new
                    
                    elif self.occupancy_data[gi, gj] == 100:
                        break


if __name__ == "__main__":
    rospy.init_node("VLMValueMap")
    VLMValueMap()
    rospy.spin()

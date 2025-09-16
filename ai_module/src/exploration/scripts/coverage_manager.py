#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import message_filters
import math

from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped

class PathSplitter:
    def __init__(self):
        rospy.loginfo("경로 분할 및 자동 순차 발행 노드 초기화 중...")
        # --- 멤버 변수 초기화 ---
        self.segmented_paths_container = []
        self.current_robot_pose = None
        self.current_segment_index = 0
        self.is_mission_active = False
        self.progress_threshold = rospy.get_param('~progress_threshold', 0.80)
        self.publish_rate_hz = rospy.get_param('~publish_rate_hz', 10.0)
        self.mission_timer = None

        # --- 퍼블리셔 및 서브스크라이버 설정 ---
        self.segment_pub = rospy.Publisher('/segmented_path_current', Path, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.odometry_callback, queue_size=1)
        path_sub = message_filters.Subscriber('/coverage_tsp_path', Path)
        waypoints_sub = message_filters.Subscriber('/grid_center_markers', MarkerArray)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [path_sub, waypoints_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.synchronized_callback)
        rospy.loginfo(f"노드가 준비되었습니다. 경로 발행 주기는 {self.publish_rate_hz}Hz 입니다.")
        rospy.loginfo("'/coverage_tsp_path'와 '/grid_center_markers' 토픽을 기다립니다...")

    def _timer_callback(self, event):
        """타이머에 의해 주기적으로 호출되어 경로를 발행하는 함수"""
        if not self.is_mission_active or self.current_segment_index >= len(self.segmented_paths_container):
            return
        
        segment_poses = self.segmented_paths_container[self.current_segment_index]
        path_segment = Path()
        path_segment.header.stamp = rospy.Time.now()
        path_segment.header.frame_id = "map"
        path_segment.poses = segment_poses
        # print(f"total poses: {len(segment_poses)}") # 디버깅용으로 남겨두셔도 좋습니다.
        self.segment_pub.publish(path_segment)

    def odometry_callback(self, msg: Odometry):
        if not self.is_mission_active: return
        self.current_robot_pose = msg.pose.pose
        if self.current_robot_pose is None or self.current_segment_index >= len(self.segmented_paths_container): return
        current_segment = self.segmented_paths_container[self.current_segment_index]
        if len(current_segment) < 2: return
        closest_point_index = self.find_closest_pose_index(current_segment, self.current_robot_pose.position)
        progress = closest_point_index / (len(current_segment) - 1.0)
        if progress >= self.progress_threshold:
            rospy.loginfo(f"구간 {self.current_segment_index + 1} 도착 완료! (진행률: {progress * 100:.1f}%)")
            self.current_segment_index += 1
            if self.current_segment_index >= len(self.segmented_paths_container):
                rospy.loginfo("모든 경로 구간을 완료했습니다! 미션을 종료합니다.")
                self.is_mission_active = False
                if self.mission_timer:
                    self.mission_timer.shutdown()
                final_path = Path()
                final_path.header.stamp = rospy.Time.now()
                final_path.header.frame_id = "map"
                self.segment_pub.publish(final_path)

    def synchronized_callback(self, path_msg: Path, markers_msg: MarkerArray):
        if self.is_mission_active: return
        rospy.loginfo("새로운 전체 경로와 경유점을 수신하여 미션을 시작합니다.")
        full_path_poses, waypoints = path_msg.poses, [m.pose.position for m in markers_msg.markers]
        if not full_path_poses or not waypoints: return
        
        waypoint_indices = sorted([(self.find_closest_pose_index(full_path_poses, wp), wp) for wp in waypoints])
        
        segmented_paths, last_idx = [], 0
        for idx, _ in waypoint_indices:
            # 두 경유점의 인덱스가 같거나 바로 옆이라면 슬라이싱 결과가 1 또는 2가 될 수 있음
            segment = full_path_poses[last_idx:idx+1]
            segmented_paths.append(segment)
            last_idx = idx
        
        if last_idx < len(full_path_poses) -1:
            segmented_paths.append(full_path_poses[last_idx:])
        
        # ###############################################################
        # ### 핵심 수정! ###
        # 길이가 2 미만인 (점이 1개 이하인) 무의미한 경로 조각들을 제거합니다.
        final_segmented_paths = [segment for segment in segmented_paths if len(segment) >= 2]
        self.segmented_paths_container = final_segmented_paths
        # ###############################################################
        
        if self.segmented_paths_container:
            self.current_segment_index = 0
            self.is_mission_active = True
            if self.mission_timer:
                self.mission_timer.shutdown()
            self.mission_timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate_hz), self._timer_callback)
            rospy.loginfo(f"경로 분할 완료! 유효한 구간 {len(self.segmented_paths_container)}개로 미션을 시작합니다. ({self.publish_rate_hz}Hz로 경로 발행 시작)")
        else:
            rospy.logwarn("유효한 경로 구간이 없어 미션을 시작할 수 없습니다.")

    def find_closest_pose_index(self, path_poses: list, target_point: Point) -> int:
        closest_index, min_dist_sq = -1, float('inf')
        for i, pose in enumerate(path_poses):
            dx = pose.pose.position.x - target_point.x; dy = pose.pose.position.y - target_point.y
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq: min_dist_sq, closest_index = dist_sq, i
        return closest_index

if __name__ == '__main__':
    try:
        rospy.init_node('path_splitter_node', anonymous=True)
        PathSplitter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
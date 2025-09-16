#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from math import hypot
from typing import List

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import tf.transformations as tft

class KeyframePublisher:
    def __init__(self):
        # 파라미터 설정
        self.distance_threshold = rospy.get_param("~distance_threshold", 1.0)
        self.time_interval_sec = rospy.get_param("~time_interval", 0.01)
        self.time_interval = rospy.Duration(self.time_interval_sec)
        self.marker_scale = rospy.get_param("~marker_scale", 0.5)
        self.frame_id = rospy.get_param("~frame_id", "map")

        # 키프레임 데이터
        self.keyframes: List[Pose] = []
        self.last_keyframe_pose: Pose | None = None
        self.last_keyframe_time: rospy.Time | None = None
        self.current_pose: Pose | None = None

        # ROS 인터페이스
        rospy.Subscriber("/state_estimation_at_scan", Odometry, self._odom_callback, queue_size=1)
        self.keyframe_pub = rospy.Publisher("/keyframes", PoseArray, queue_size=10)
        self.marker_pub = rospy.Publisher("/keyframe_markers", MarkerArray, queue_size=10)

        rospy.loginfo("Keyframe Publisher node initialized.")
        rospy.loginfo(f"Distance Threshold: {self.distance_threshold} meters")
        rospy.loginfo(f"Time Interval: {self.time_interval.to_sec()} seconds")
        rospy.loginfo(f"Marker Scale: {self.marker_scale}")
        rospy.loginfo(f"Frame ID: {self.frame_id}")


    def _odom_callback(self, msg: Odometry):
        """Odometry 메시지를 수신하고 키프레임 로직을 처리합니다."""
        self.current_pose = msg.pose.pose
        current_time = msg.header.stamp

        if self._should_add_keyframe(self.current_pose, current_time):
            self._add_keyframe(self.current_pose, current_time)
            self._publish_keyframes(msg.header)
            self._publish_markers(msg.header)

    def _calculate_distance(self, pose1: Pose, pose2: Pose) -> float:
        """두 Pose 간의 유클리드 거리를 계산합니다."""
        p1 = pose1.position
        p2 = pose2.position
        return hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _should_add_keyframe(self, current_pose: Pose, current_time: rospy.Time) -> bool:
        """새 키프레임을 추가해야 하는지 여부를 결정합니다."""
        if self.last_keyframe_pose is None or self.last_keyframe_time is None:
            return True

        distance_moved = self._calculate_distance(self.last_keyframe_pose, current_pose)
        time_elapsed = current_time - self.last_keyframe_time

        if distance_moved >= self.distance_threshold:
            rospy.loginfo(f"Adding keyframe due to distance: {distance_moved:.2f} m")
            return True

        if time_elapsed >= self.time_interval:
            rospy.loginfo(f"Adding keyframe due to time: {time_elapsed.to_sec():.2f} s")
            return True

        return False

    def _add_keyframe(self, pose: Pose, time: rospy.Time):
        """키프레임 목록에 새 포즈를 추가합니다."""
        self.keyframes.append(pose)
        self.last_keyframe_pose = pose
        self.last_keyframe_time = time
        rospy.loginfo(f"Added keyframe #{len(self.keyframes)}. Position: ({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})")

    def _publish_keyframes(self, header):
        """현재 키프레임 목록을 PoseArray로 게시합니다."""
        pa = PoseArray()
        pa.header = header
        pa.header.frame_id = self.frame_id
        pa.poses = self.keyframes
        self.keyframe_pub.publish(pa)

    def _create_marker(self, pose: Pose, marker_id: int, color: ColorRGBA, scale: float, header) -> Marker:
        """단일 마커를 생성합니다."""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.frame_id
        marker.ns = "keyframes"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        marker.lifetime = rospy.Duration(0) # 영구적으로 표시
        return marker

    def _publish_markers(self, header):
        """키프레임을 시각화하기 위한 마커 배열을 게시합니다."""
        marker_array = MarkerArray()
        
        color_previous = ColorRGBA(0.0, 1.0, 1.0, 0.6) # 청록색 (이전 키프레임)
        color_latest = ColorRGBA(1.0, 0.0, 0.0, 1.0) # 빨간색 (최신 키프레임)

        for i, pose in enumerate(self.keyframes):
            is_latest = (i == len(self.keyframes) - 1)
            color = color_latest if is_latest else color_previous
            scale = self.marker_scale * 1.5 if is_latest else self.marker_scale # 최신 마커를 약간 크게

            marker = self._create_marker(pose, i, color, scale, header)
            marker_array.markers.append(marker)
            
        # 오래된 마커 삭제 (선택 사항: ID가 더 이상 목록에 없으면 삭제)
        # 현재는 ADD 액션만 사용하므로, 이전 마커가 남아있게 됩니다.
        # 만약 목록에서 키프레임이 제거될 경우, 삭제 로직이 필요합니다.

        self.marker_pub.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("keyframe_publisher_with_markers")
    KeyframePublisher()
    rospy.spin()
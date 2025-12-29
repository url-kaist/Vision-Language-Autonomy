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
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String


class Test:
    def __init__(self):
        try:
            self.frontier_sub = rospy.Subscriber("/frontier_markers", MarkerArray, self._frontier_callback, queue_size=1)
            self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

            self.exploration_strategy = "geometric_frontier"
            self.exploration_strategy_sub = rospy.Subscriber("/exploration_strategy", String, self._strategy_cmd_callback, queue_size=1)
            self.is_vg_first = False

            self.current_pose = None
            self.current_index = -1
            self.odom_sub = rospy.Subscriber("/Odometry", Odometry, self._odom_callback, queue_size=1)
            self.indices_sub = rospy.Subscriber('/tsp_path_indices', Int32MultiArray, self.indices_callback)
            
            # vg_first not -> frontier 
            # vg_first ->ATSP (node number is not matter.) -> ATSP node order ->/move_base_simple/goal publish by order.
        except Exception as e:
            rospy.logerr(f"<Test.__init__> Error occurs: {e}")


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

    def _strategy_cmd_callback(self, msg: String) -> None:
        new_strategy = msg.data
        if new_strategy in ["semantic_frontier", "geometric_frontier", 
                            "semantic_atsp", "geometric_atsp"]:
            self.is_vg_first = False

        elif new_strategy in ['vg_first', 'vg_first_inference']:
            self.is_vg_first = True
        elif new_strategy == "coverage_planning":
            self.is_vg_first = False
        else:
            self.logger.logwarn(f"Received unknown exploration strategy command: {new_strategy}. Keeping current strategy: {self.exploration_strategy}")

    def indices_callback(self, msg):
        # 3. 데이터 처리
        # msg.data에 리스트 형태(tuple)로 데이터가 들어있습니다.
        indices_list = list(msg.data) # 튜플을 리스트로 변환 (필요시)
        
        rospy.loginfo(f"Received indices: {indices_list}")
        
        # 예: 첫 번째 인덱스 접근
        if indices_list:
            first_index = indices_list[0]
            rospy.loginfo(f"First target index: {first_index}")

        # 여기에 원하는 로직 추가 (예: 로봇 이동 명령 등))

    def _frontier_callback(self, msg):
        # short path first # maybe..?
        try:
            rospy.loginfo(f"<_frontier_callback.0>")
            if len(msg.markers) == 0:
                return
            if self.is_vg_first:
                return

            for m in msg.markers:
                pose = PoseStamped(header=m.header)
                pose.pose.position.x = m.pose.position.x
                pose.pose.position.y = m.pose.position.y
                pose.pose.position.z = m.pose.position.z
                pose.pose.orientation.x = m.pose.orientation.x
                pose.pose.orientation.y = m.pose.orientation.y
                pose.pose.orientation.z = m.pose.orientation.z
                pose.pose.orientation.w = m.pose.orientation.w

                self.goal_pub.publish(pose)
            rospy.loginfo(f"<_frontier_callback.1> Publish: ({pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z})")
        except Exception as e:
            rospy.logerr(f"<_frontier_callback> Error occurs: {e}")

if __name__ == "__main__":
    rospy.init_node("Test")
    Test()
    rospy.spin()
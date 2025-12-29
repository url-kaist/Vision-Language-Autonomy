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
from std_msgs.msg import Bool


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
            self.active_waypoints_sub = rospy.Subscriber("/active_waypoints", MarkerArray, self._active_waypoints_callback, queue_size=10)
            self.arrival_status_sub = rospy.Publisher("/path_follower/arrived", Bool, self._arrival_status_callback, queue_size=1)
            self.running = False
            self.already_pub_goal = False

            # vg_first not -> frontier 
            # vg_first ->ATSP (node number is not matter.) -> ATSP node order ->/move_base_simple/goal publish by order.
        except Exception as e:
            rospy.logerr(f"<Test.__init__> Error occurs: {e}")

    def _arrival_status_callback(self, msg):
        self.path_follower_ready = msg.data

    def _active_waypoints_callback(self, msg):
        waypoints = []
        for m in msg.markers:
            waypoint_xyz = np.array([m.pose.position.x, m.pose.position.y, 0.0])
            waypoint_ori = np.array([m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w])
            header = m.header
            waypoints.append((header, waypoint_xyz, waypoint_ori))

        # Update waypoints
        self.waypoints = waypoints

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
        if self.path_follower_ready:
            if self.running:
                return
            if self.already_pub_goal:
                self.goal_pub.publish(self.current_pose)
                return
            self.running = True

            position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

            target_waypoint, target_index = None, None
            min_dist = np.inf
            for _index, (header, xyz, ori) in enumerate(self.waypoints):
                dist = np.linalg.norm(xyz[:2] - position[:2])
                if dist < min_dist:
                    target_waypoint = (header, xyz, ori)
                    target_index = _index

            if target_waypoint is not None:
                (header, xyz, ori) = target_waypoint
                pose = PoseStamped(header=header)
                pose.pose.position.x = xyz[0]
                pose.pose.position.y = xyz[1]
                pose.pose.position.z = xyz[2]
                pose.pose.orientation.x = ori[0]
                pose.pose.orientation.y = ori[1]
                pose.pose.orientation.z = ori[2]
                pose.pose.orientation.w = ori[3]
                self.current_pose = pose
                del self.waypoints[target_index]
                self.goal_pub.publish(self.current_pose)
                self.already_pub_goal = True

            self.running = False
        else:
            self.already_pub_goal = False

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
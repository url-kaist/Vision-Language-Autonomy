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
            self.path_follower_ready = False
            self.frontier_sub = rospy.Subscriber("/frontier_markers", MarkerArray, self._frontier_callback, queue_size=1)
            self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
            self.test_target_pub = rospy.Publisher("/test/target", String, queue_size=1)

            self.exploration_strategy = "geometric_frontier"
            self.exploration_strategy_sub = rospy.Subscriber("/exploration_strategy", String, self._strategy_cmd_callback, queue_size=1)
            
            # --- [Added] Instruction Status Subscriber ---
            # Listens for "no_frontier" to switch modes
            self.instruction_status_sub = rospy.Subscriber("/instruction_following_exp_status", String, self._instruction_status_callback, queue_size=1)
            # ---------------------------------------------

            self.is_vg_first = False

            self.current_pose = None
            self.current_index = -1
            
            # --- [Added] Initialization ---
            self.waypoints = [] 
            # -------------------

            self.odom_sub = rospy.Subscriber("/Odometry", Odometry, self._odom_callback, queue_size=1)
            self.active_waypoints_sub = rospy.Subscriber("/active_waypoints", MarkerArray, self._active_waypoints_callback, queue_size=10)
            self.arrival_status_sub = rospy.Subscriber("/path_follower/arrived", Bool, self._arrival_status_callback, queue_size=1)
            self.running = False
            self.already_pub_goal = False

            # --- [Added] Stuck detection variables ---
            self.stuck_timeout = rospy.Duration(rospy.get_param("~stuck_timeout", 10.0))
            self.stuck_dist_thresh = rospy.get_param("~stuck_dist_thresh", 0.2)
            
            self.last_stuck_check_time = rospy.Time.now()
            self.last_stuck_check_pos = None
            self.current_frontier_target = None 
            self.blacklisted_frontiers = []     
            # ------------------------------------------

        except Exception as e:
            rospy.logerr(f"<Test.__init__> Error occurs: {e}")

    # --- [Added] New Callback for Instruction Status ---
    def _instruction_status_callback(self, msg: String):
        """
        Handles the instruction status. 
        If 'no_frontier' is received, we switch to a mode that ignores frontiers 
        and follows active waypoints (equivalent to setting is_vg_first = True).
        """
        if msg.data == "no_frontier":
            rospy.loginfo("[Test] Received 'no_frontier'. Switching to Waypoint Focus Mode (ignoring frontiers).")
            self.is_vg_first = True
    # ---------------------------------------------------

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
        # Current robot position
        current_pos_np = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # --- Stuck Detection Logic ---
        if self.last_stuck_check_pos is None:
            self.last_stuck_check_pos = current_pos_np
            self.last_stuck_check_time = rospy.Time.now()

        # Only check stuck if NOT in waypoint mode (is_vg_first is False) and we have a target
        if not self.is_vg_first and self.current_frontier_target is not None:
            time_elapsed = rospy.Time.now() - self.last_stuck_check_time
            if time_elapsed > self.stuck_timeout:
                dist_moved = np.linalg.norm(current_pos_np - self.last_stuck_check_pos)
                if dist_moved < self.stuck_dist_thresh:
                    rospy.logwarn(f"[Test] Stuck detected. Blacklisting: {self.current_frontier_target}")
                    self.blacklisted_frontiers.append(self.current_frontier_target)
                    self.current_frontier_target = None
                
                self.last_stuck_check_pos = current_pos_np
                self.last_stuck_check_time = rospy.Time.now()
        # -----------------------------

        # If is_vg_first is False, we stop here (ignoring waypoints)
        if not self.is_vg_first:
            return
            
        # --- Waypoint Following Logic (Active when is_vg_first == True) ---
        rospy.loginfo(f"[VG First] self.path_follower_ready: {self.path_follower_ready}")
        if self.path_follower_ready:
            if self.running:
                return
            if self.already_pub_goal:
                if self.current_pose:
                    self.goal_pub.publish(self.current_pose)
                    self.test_target_pub.publish(String(data="vg_first"))
                    rospy.loginfo(f"[VG First] Publish {self.current_pose}")
                return
            self.running = True

            position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

            target_waypoint, target_index = None, None
            min_dist = np.inf
            
            if self.waypoints:
                for _index, (header, xyz, ori) in enumerate(self.waypoints):
                    dist = np.linalg.norm(xyz[:2] - position[:2])
                    if dist < min_dist:
                        min_dist = dist
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
                self.test_target_pub.publish(String(data="vg_first"))
                rospy.loginfo(f"[VG First] Publish {self.current_pose}")
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
            rospy.logwarn(f"Received unknown exploration strategy command: {new_strategy}.")

    def _frontier_callback(self, msg):
        try:
            # If is_vg_first is True (which happens if we received "no_frontier"), 
            # we return immediately, ignoring all frontier markers.
            if self.is_vg_first:
                return

            if len(msg.markers) == 0:
                return

            # --- Blacklist Logic ---
            filtered_markers = []
            for m in msg.markers:
                is_blacklisted = False
                m_pos = np.array([m.pose.position.x, m.pose.position.y])
                for bp in self.blacklisted_frontiers:
                    if np.linalg.norm(m_pos - np.array(bp)) < 0.5:
                        is_blacklisted = True
                        break
                if not is_blacklisted:
                    filtered_markers.append(m)

            if not filtered_markers:
                rospy.loginfo("All frontiers are blacklisted or none available.")
                return

            for m in filtered_markers:
                pose = PoseStamped(header=m.header)
                pose.pose.position.x = m.pose.position.x
                pose.pose.position.y = m.pose.position.y
                pose.pose.position.z = m.pose.position.z
                pose.pose.orientation.x = m.pose.orientation.x
                pose.pose.orientation.y = m.pose.orientation.y
                pose.pose.orientation.z = m.pose.orientation.z
                pose.pose.orientation.w = m.pose.orientation.w

                self.goal_pub.publish(pose)
                self.test_target_pub.publish(String(data="frontier"))
                self.current_frontier_target = (m.pose.position.x, m.pose.position.y)

            rospy.loginfo(f"<_frontier_callback> Publish goal. Current target: {self.current_frontier_target}")

        except Exception as e:
            rospy.logerr(f"<_frontier_callback> Error occurs: {e}")

if __name__ == "__main__":
    rospy.init_node("Test")
    Test()
    rospy.spin()
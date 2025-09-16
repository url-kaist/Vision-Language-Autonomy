#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from enum import Enum
import sys
import threading
import random
sys.path.append("/ws/external")

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Pose2D
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Int16, String

from ai_module.src.utils.logger import Logger

class Status(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"

class PathFollower:
    def __init__(self):
        """
        Initializes the Path Follower node.
        Input:
            None
        Output:
            None
        Parameter description:
            ~arrival_threshold (float): The distance in meters at which a waypoint is considered reached.
            ~loop_rate (float): The frequency in Hz for the main control loop.
        """        
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='PathFollower', log_path="/ws/external/log/exploration/path_follower.log")

        self.logger.loginfo("Initializing Path Follower node...")

        self.arrival_threshold = rospy.get_param('~arrival_threshold', 0.5)
        self.lookahead_dist = rospy.get_param('~lookahead_dist', 1.0)
        self.min_ld = rospy.get_param('~min_ld', 0.3)
        loop_rate = rospy.get_param('~loop_rate', 10.0)

        self.path_pub = rospy.Publisher("/jackal/path", Path, queue_size=1)
        self._target_path = None
        self.target_path1 = None
        self.target_path2 = None
        self.target_path3 = None
        self.prev_target_path = None
        self.current_waypoint_index = 0
        self.prev_waypoint_index = 0
        self.prev_assigned_index = 0   # Progress index which is used as a lower bound (to prevent regression)
        self.current_pose = None
        self.path_received = False
        self.path_follower_mode = 0
        self.ld_idx = 0.0
        self.prev_ld_idx = 0.0
        self.stuck = 0
        self.delay = 0
        self.is_stuck = False
        self.task_mode = None

        rospy.Subscriber("/planned_path", Path, self._planned_path_callback) ## 1
        rospy.Subscriber("/coverage_tsp_path", Path, self._coverage_path_callback) ## 2
        rospy.Subscriber("/waypoint_tsp_path", Path, self._waypoint_path_callback) ## 3
        rospy.Subscriber("/path_follower_mode", Int16, self._mode_callback) 
        rospy.Subscriber("/state_estimation", Odometry, self._odom_callback)
        rospy.Subscriber("/question_type", String, self._task_mode_callback)

        self.waypoint_pub = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=10)
        self.marker_pub = rospy.Publisher("/exploration/marker", Marker, queue_size=10)

        self.running = False
        self.active_signal_srv = rospy.Service("/path_follower/active_signal", Trigger, self._active_signal_callback)
        self.status_srv = rospy.Service("/path_follower/status", Trigger, self._status_callback)
        
        self._lock = threading.RLock()

        self.rate = rospy.Rate(loop_rate)
        self.logger.loginfo("Path Follower node is ready.")

    def _active_signal_callback(self, req):
        with self._lock:
            self.running = not self.running
            # if not self.running:
            #     self._initialize_path()
            
        self.logger.loginfo(f"State changed to {self.running}")
        
        return TriggerResponse(success=self.running, message="current state")

    def _status_callback(self, req):
        status = Status.WAITING            
        if self.path_received:
            status = Status.RUNNING
            
            # Check if reached the end of the path
            target_pose_stamped = self.target_path.poses[-1]

            robot_pos = self.current_pose.position
            waypoint_pos = target_pose_stamped.pose.position
            
            dist_to_goal = math.sqrt(
                (robot_pos.x - waypoint_pos.x)**2 +
                (robot_pos.y - waypoint_pos.y)**2
            )

            if dist_to_goal <= self.arrival_threshold and self.ld_idx >= (len(self.target_path.poses) - 5):
                self.logger.loginfo(f"[Status] dist_to_goal={dist_to_goal:.3f} m <= arrival_threshold={self.arrival_threshold:.3f} m")
                self.logger.loginfo(f"[Status] index_to_goal={self.ld_idx} <= arrival_threshold={len(self.target_path.poses) - 5}")
                status = Status.COMPLETED

            self.logger.loginfo(f"[Status] status = {status}")

        return TriggerResponse(success=1, message=f"{status}")
        
    def _closest_forward_index_on_new_path(self, robot_x, robot_y, new_path_xy, min_idx_threshold=0):
        if len(new_path_xy) == 0:
            return 0

        dist_to_path = np.linalg.norm(new_path_xy - np.array([robot_x, robot_y]), axis=1)

        cand_indices = np.where(np.arange(len(new_path_xy)) >= min_idx_threshold)[0]
        if len(cand_indices) == 0:
            return min_idx_threshold  # 또는 마지막 인덱스

        dists_for_cand = dist_to_path[cand_indices]
        local_closest_idx_in_cand = np.argmin(dists_for_cand)

        return cand_indices[local_closest_idx_in_cand]

    def _mode_callback(self, msg):
        self.logger.loginfo(f"--- Mode callback triggered! Received mode: {msg.data}, Previous mode: {self.path_follower_mode} ---")
        
        with self._lock:
            if self.path_follower_mode != msg.data:
                self.current_waypoint_index = 0
                self.prev_assigned_index = 0
                self.path_received = False
                self.logger.loginfo(f"Path follower mode CHANGED to {msg.data}. Path state has been reset.")
            
            self.path_follower_mode = msg.data

        
    def _initialize_path(self):
        self.current_waypoint_index = 0
        self.prev_assigned_index = 0
        self.path_received = False
        self.target_path = None
        self.current_pose = None
        self.running = False
        self.logger.loginfo("Path Follower state has been reset.")


    def _planned_path_callback(self, msg: Path):
        self.target_path1 = msg
        if self.path_follower_mode == 1:
            with self._lock:
                if not msg.poses:
                    self.logger.logwarn("Received path has no poses. Ignoring.")
                    return

                self.target_path = msg
                self.path_received = True

                if self.target_path != self.prev_target_path:
                    self.logger.loginfo(f"Path Initailize.")
                    self.current_waypoint_index = 0

                    min_dist = float('inf')
                    idx = 0
                    for target_p in self.target_path.poses[:]:
                        dist = math.hypot(target_p.pose.position.x - self.current_pose.position.x, target_p.pose.position.y - self.current_pose.position.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = idx + self.current_waypoint_index
                        idx += 1

                    self.current_waypoint_index = min_idx
                    self.prev_ld_idx = float(self.current_waypoint_index)

                self.prev_target_path = self.target_path


    @property
    def target_path(self):
        return self._target_path

    @target_path.setter
    def target_path(self, path):
        if path is not None:
            self.path_pub.publish(path)
            self._target_path = path
        else:
            # (선택) RViz에서 경로 지우고 싶다면 '빈 Path'를 보냄
            # empty = Path()
            # empty.header.stamp = rospy.Time.now()
            # empty.header.frame_id = "map"
            # self.path_pub.publish(empty)
            pass

    def _coverage_path_callback(self, msg: Path):
        """
        Resets the state upon receiving a new path.
        Input:
            msg (nav_msgs.msg.Path): The new path to be followed.
        Output:
            None
        Parameter description:
            self.target_path: Stores the received path.
            self.current_waypoint_index: Resets to 0 to start from the beginning of the new path.
            self.path_received: Flag set to True to indicate a valid path is available.
        """
        self.target_path2 = msg
        if self.path_follower_mode == 2:
            with self._lock:
                if not msg.poses:
                    self.logger.logwarn("Received path has no poses. Ignoring.")
                    return

                self.logger.loginfo(f"Received a new path with {len(msg.poses)} poses.")
                self.target_path = msg

                self.path_received = True

                if len(self.target_path.poses) != len(self.prev_target_path.poses):
                    self.logger.loginfo(f"Path Initailize.")
                    self.current_waypoint_index = 0

                    min_dist = float('inf')
                    idx = 0
                    for target_p in self.target_path.poses[:]:
                        dist = math.hypot(target_p.pose.position.x - self.current_pose.position.x, target_p.pose.position.y - self.current_pose.position.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = idx + self.current_waypoint_index
                        idx += 1

                    self.current_waypoint_index = min_idx
                    self.prev_ld_idx = float(self.current_waypoint_index)

                self.prev_target_path = self.target_path

                self.logger.loginfo(f"Received a new path with {len(msg.poses)} poses.")

    def _waypoint_path_callback(self, msg: Path):
        """
        Resets the state upon receiving a new path.
        Input:
            msg (nav_msgs.msg.Path): The new path to be followed.
        Output:
            None
        Parameter description:
            self.target_path: Stores the received path.
            self.current_waypoint_index: Resets to 0 to start from the beginning of the new path.
            self.path_received: Flag set to True to indicate a valid path is available.
        """
        self.target_path3 = msg
        if self.path_follower_mode == 3:
            with self._lock:
                if not msg.poses:
                    self.logger.logwarn("Received path has no poses. Ignoring.")
                    return

                self.logger.loginfo(f"Received a new path with {len(msg.poses)} poses.")
                self.target_path = msg
                self.path_received = True

                if self.target_path != self.prev_target_path:
                    self.logger.loginfo(f"Path Initailize.")
                    self.current_waypoint_index = 0

                    min_dist = float('inf')
                    idx = 0
                    for target_p in self.target_path.poses[:]:
                        dist = math.hypot(target_p.pose.position.x - self.current_pose.position.x, target_p.pose.position.y - self.current_pose.position.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = idx + self.current_waypoint_index
                        idx += 1

                    self.current_waypoint_index = min_idx
                    self.prev_ld_idx = float(self.current_waypoint_index)

                self.prev_target_path = self.target_path

                self.logger.loginfo(f"Received a new path with {len(msg.poses)} poses.")

    def _odom_callback(self, msg: Odometry):
        """
        Updates the robot's current pose from Odometry messages.
        Input:
            msg (nav_msgs.msg.Odometry): The incoming odometry data.
        Output:
            None
        Parameter description:
            self.current_pose: Stores the latest pose of the robot.
        """
        self.current_pose = msg.pose.pose

    def _task_mode_callback(self, msg):
        with self._lock:
            if self.task_mode != msg.data:
                self.task_mode = msg.data

                # if self.task_mode == "instruction_following":
                #     self.lookahead_dist = 0.5

                self.logger.loginfo(f"Path task mode CHANGED to {msg.data}. Path state has been reset.")


    # Visualize
    def publish_waypoint_markers(self, target_path, current_index, ld_idex):

        current_pose = target_path[current_index].pose.position
        next_pose = target_path[ld_idex].pose.position

        markers = []

        # Current waypoint (SPHERE)
        current_marker = Marker()
        current_marker.header.frame_id = "map"
        current_marker.header.stamp = rospy.Time.now()
        current_marker.ns = "waypoints"
        current_marker.id = 0
        current_marker.type = Marker.SPHERE
        current_marker.action = Marker.ADD
        current_marker.pose.position.x = current_pose.x
        current_marker.pose.position.y = current_pose.y
        current_marker.pose.position.z = 0.0
        current_marker.pose.orientation.w = 1.0
        current_marker.scale.x = 0.2
        current_marker.scale.y = 0.2
        current_marker.scale.z = 0.2
        current_marker.color.r = 0.0
        current_marker.color.g = 1.0
        current_marker.color.b = 0.0
        current_marker.color.a = 1.0
        markers.append(current_marker)

        # Current waypoint ID (text)
        current_text = Marker()
        current_text.header.frame_id = "map"
        current_text.header.stamp = rospy.Time.now()
        current_text.ns = "waypoints_text"
        current_text.id = 1
        current_text.type = Marker.TEXT_VIEW_FACING
        current_text.action = Marker.ADD
        current_text.pose.position.x = current_pose.x
        current_text.pose.position.y = current_pose.y
        current_text.pose.position.z += 0.3  # 구 위쪽에 띄우기
        current_text.pose.orientation.w = 1.0
        current_text.scale.z = 0.2  # 텍스트 크기
        current_text.color.r = 0.0
        current_text.color.g = 0.0
        current_text.color.b = 0.0
        current_text.color.a = 1.0
        current_text.text = str(current_index)
        markers.append(current_text)

        # Next waypoint (SPHERE)
        next_marker = Marker()
        next_marker.header.frame_id = "map"
        next_marker.header.stamp = rospy.Time.now()
        next_marker.ns = "waypoints"
        next_marker.id = 2
        next_marker.type = Marker.SPHERE
        next_marker.action = Marker.ADD
        next_marker.pose.position.x = next_pose.x
        next_marker.pose.position.y = next_pose.y
        next_marker.pose.position.z = 0.0
        next_marker.pose.orientation.w = 1.0
        next_marker.scale.x = 0.2
        next_marker.scale.y = 0.2
        next_marker.scale.z = 0.2
        next_marker.color.r = 1.0
        next_marker.color.g = 0.0
        next_marker.color.b = 0.0
        next_marker.color.a = 1.0
        markers.append(next_marker)

        # Next waypoint ID (text)
        next_text = Marker()
        next_text.header.frame_id = "map"
        next_text.header.stamp = rospy.Time.now()
        next_text.ns = "waypoints_text"
        next_text.id = 3
        next_text.type = Marker.TEXT_VIEW_FACING
        next_text.action = Marker.ADD
        next_text.pose.position.x = next_pose.x
        next_text.pose.position.y = next_pose.y
        next_text.pose.position.z += 0.3
        next_text.pose.orientation.w = 1.0
        next_text.scale.z = 0.2
        next_text.color.r = 0.0
        next_text.color.g = 0.0
        next_text.color.b = 0.0
        next_text.color.a = 1.0
        next_text.text = str(ld_idex)
        markers.append(next_text)

        # Publish each marker
        for m in markers:
            self.marker_pub.publish(m)

    def calculate_curvature(self, p1, p2, p3):
        x1, y1 = p1.pose.position.x, p1.pose.position.y
        x2, y2 = p2.pose.position.x, p2.pose.position.y
        x3, y3 = p3.pose.position.x, p3.pose.position.y

        numerator = 2 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # 분모 계산: (세 변의 길이의 곱)
        d12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        d23 = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        d31 = math.sqrt((x3 - x1) ** 2 + (y1 - y3) ** 2)

        denominator = d12 * d23 * d31

        # 분모가 0에 가까우면 (세 점이 거의 일직선) 곡률은 0
        if denominator < 1e-9:
            return 0.0

        return numerator / denominator

    def run(self):
        """
        The main control loop for path following.
        Input:
            None
        Output:
            None
        Parameter description:
            None
        """
        while not rospy.is_shutdown():
            with self._lock:
                if self.path_received and self.target_path and self.current_pose:
                    if self.current_waypoint_index >= len(self.target_path.poses):
                        # self.logger.loginfo("Path following complete!")
                
                        # TODO: End signal
                        continue

                    if self.is_stuck:
                        min_idx = self.current_waypoint_index
                        min_dist = float('inf')
                        idx = 0
                        for target_p in self.target_path.poses[self.current_waypoint_index:]:
                            dist = math.hypot(target_p.pose.position.x - self.current_pose.position.x, target_p.pose.position.y - self.current_pose.position.y)
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = idx + self.current_waypoint_index
                            if idx >30:
                                break
                            idx += 1

                        self.current_waypoint_index = min_idx

                        self.ld_idx = float(self.current_waypoint_index) + random.randint(0, 30)

                        if self.ld_idx >= len(self.target_path.poses):
                            self.ld_idx = float(len(self.target_path.poses) - 1)
                        t_p = self.target_path.poses[int(self.ld_idx)]
                        tx, ty = t_p.pose.position.x, t_p.pose.position.y

                        self.logger.loginfo(f"self.current_waypoint_index = {self.current_waypoint_index}")
                        self.logger.loginfo(f"self.ld_idx = {self.ld_idx}")
                    
                        waypoint_2d = Pose2D()
                    
                        waypoint_2d.x = tx
                        waypoint_2d.y = ty
                    
                        self.waypoint_pub.publish(waypoint_2d)

                        self.publish_waypoint_markers(self.target_path.poses, self.current_waypoint_index, int(self.ld_idx))

                        if self.prev_waypoint_index != self.current_waypoint_index:
                            self.stuck = 0
                            self.is_stuck = False

                        continue

                    if self.prev_ld_idx > self.prev_waypoint_index + 40:
                        self.current_waypoint_index = int(self.ld_idx) - 10

                    min_idx = self.current_waypoint_index
                    min_dist = float('inf')
                    idx = 0
                    for target_p in self.target_path.poses[self.current_waypoint_index:]:
                        dist = math.hypot(target_p.pose.position.x - self.current_pose.position.x, target_p.pose.position.y - self.current_pose.position.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = idx + self.current_waypoint_index
                        if idx > 30:
                            break
                        idx += 1

                    self.current_waypoint_index = min_idx

                    target_pose_stamped = self.target_path.poses[-1]

                    robot_pos = self.current_pose.position
                    waypoint_pos = target_pose_stamped.pose.position
                    
                    dist_to_goal = math.sqrt(
                        (robot_pos.x - waypoint_pos.x)**2 +
                        (robot_pos.y - waypoint_pos.y)**2
                    )
                                        
                    if dist_to_goal <= (self.arrival_threshold-0.2) and self.ld_idx >= (len(self.target_path.poses) - 5):
                        self.stuck = 0
                        self.is_stuck = False
                        self.logger.loginfo(f"Arrived at path end; dist_to_goal={dist_to_goal:.3f} m")
                        continue
                    else:
                        if self.prev_waypoint_index == self.current_waypoint_index:
                            self.stuck +=1
                            if self.stuck > 50:
                                self.is_stuck = True
                                self.logger.loginfo(f"Robot stuck !!!!!!!!!!")
                                
                        else:
                            self.stuck = 0
                            self.is_stuck = False

                        

                    target_idx = self.current_waypoint_index + 15
                    if target_idx >= len(self.target_path.poses):
                        target_idx = len(self.target_path.poses) - 1
                    middle_idx = (self.current_waypoint_index  + target_idx) // 2

                    current_ld = self.lookahead_dist
                    if self.current_waypoint_index  != middle_idx and self.current_waypoint_index != target_idx:
                        curvature = self.calculate_curvature(self.target_path.poses[self.current_waypoint_index], self.target_path.poses[middle_idx], self.target_path.poses[target_idx])
                        factor = min(curvature, 1.0)
                        current_ld = self.lookahead_dist - (self.lookahead_dist - self.min_ld) * factor


                    self.ld_idx = float(self.current_waypoint_index) + 5
                    max_dist = 0
                    while self.ld_idx < len(self.target_path.poses):
                        t_p = self.target_path.poses[int(self.ld_idx)]
                        px, py = t_p.pose.position.x, t_p.pose.position.y
                        dist = math.hypot(px - self.current_pose.position.x, py - self.current_pose.position.y)
                        if dist < current_ld:
                            if dist > self.min_ld and dist < max_dist:
                                break
                            self.ld_idx += 1
                            max_dist = dist
                        else:
                            if self.ld_idx < self.prev_ld_idx:
                                self.ld_idx = self.prev_ld_idx
                            break
                        

                    if self.prev_waypoint_index == self.current_waypoint_index and int(self.prev_ld_idx) == int(self.ld_idx):
                        self.ld_idx += 0.2

                    if self.ld_idx >= len(self.target_path.poses):
                        self.ld_idx = float(len(self.target_path.poses) - 1)
                    t_p = self.target_path.poses[int(self.ld_idx)]
                    tx, ty = t_p.pose.position.x, t_p.pose.position.y
                
                    # target_pose_stamped = self.target_path.poses[self.current_waypoint_index]
                
                    waypoint_2d = Pose2D()
                
                    waypoint_2d.x = tx
                    waypoint_2d.y = ty
                
                    self.waypoint_pub.publish(waypoint_2d)
                    

                    self.prev_waypoint_index = self.current_waypoint_index
                    self.prev_ld_idx = self.ld_idx

                    # self.logger.loginfo("Path follower is working")
                
                    self.publish_waypoint_markers(self.target_path.poses, self.current_waypoint_index, int(self.ld_idx))

                    self.delay = 0

                else:
                    if not self.path_received and self.target_path and self.current_pose:
                        self.delay +=1
                        self.logger.loginfo(f"Path delay: {self.delay}")
                        if self.delay > 15:
                            if self.path_follower_mode == 1:
                                self.target_path = self.target_path1
                                self.path_received = True
                                self.delay = 0
                            elif self.path_follower_mode == 2:
                                self.target_path = self.target_path2
                                self.path_received = True
                                self.delay = 0
                            elif self.path_follower_mode == 3:
                                self.target_path = self.target_path3
                                self.path_received = True
                                self.delay = 0
                            else:
                                self.target_path = None
                                self.path_received = True
                                self.delay = 0

                    if not self.path_received:
                        self.logger.loginfo("Path follower not working: self.path_received is False")
                    elif not self.target_path:
                        self.logger.loginfo("Path follower not working: self.target_path is None or empty")
                    elif not self.current_pose:
                        self.logger.loginfo("Path follower not working: self.current_pose is None")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', anonymous=True)
        follower = PathFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
    
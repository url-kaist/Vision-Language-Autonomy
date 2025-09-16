#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external")
sys.path.append("/ws/external/ai_module/src/")
import math
import rospy
from geometry_msgs.msg import Pose2D, PoseStamped ,PoseArray,Point
from nav_msgs.msg import Odometry, Path          
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Bool, String
from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
from exploration.srv import AstarPath, AstarPathRequest #
from std_msgs.msg import Int16
from ai_module.src.utils.logger import Logger
from std_srvs.srv import Trigger, TriggerResponse


class WaypointSelector:
    def __init__(self) -> None:
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='WaypointSelector', log_path="/ws/external/log/exploration/waypoint_selector.log")

        self.is_real_world = rospy.get_param('~real_world', False)
        if self.is_real_world:
            self.logger.loginfo("Hello Real World!!")
            self.frame_id = "world"
        else:
            self.frame_id = "map"

        self.visited_R      = rospy.get_param("~visited_radius", 0.5)
        self.closed_R       = rospy.get_param("~closed_radius",  1.0)
        self.repick_T       = rospy.get_param("~repick_period",  1.0)
        self.stall_T        = rospy.get_param("~stall_time",     5.0)
        self.move_eps       = rospy.get_param("~move_epsilon",   0.05)
        self.no_frontier_timeout = rospy.get_param("~no_frontier_timeout", 5.0)  # 10 seconds default
        

        self.logger.loginfo("Waiting for A* path service...")
        rospy.wait_for_service('/astar_path') 
        self.astar_path_service = rospy.ServiceProxy('/astar_path', AstarPath)
        self.logger.loginfo("A* path service found.")
        
        self.robot_x = self.robot_y = 0.0
        self.last_pose_x = self.last_pose_y = 0.0
        self.last_move_t = rospy.Time.now()
        self.pose_ok     = False

        self.frontiers  = {}            
        self.closed_xy  = []            
        self.last_goal  = None          

        self.stop_flag  = False       
        self.exploration_strategy = "geometric_frontier"
        
        # No frontier fallback tracking
        self.no_frontier_start_time = None  # When we first detected no frontiers
        self.has_switched_to_coverage = False  # Prevent multiple switches
        self.desired_frontier_strategy = None  # Strategy to return to when frontiers are found
        
        # Coverage restart tracking to prevent infinite restart loops
        self.last_coverage_restart_time = None
        self.coverage_restart_cooldown = 5.0  # 5 seconds minimum between restarts
        self.coverage_start_time = None  # Track when coverage path started
        
        # Backup retry mechanism for failed restarts
        self.coverage_retry_timer = rospy.Timer(rospy.Duration(10.0), self._coverage_retry_check)

        self.logger.loginfo(f"No frontier fallback timeout: {self.no_frontier_timeout}s")
        self.logger.loginfo("Waiting for Path Follower services...")
        try:
            rospy.wait_for_service('/path_follower/active_signal', timeout=5.0)
            rospy.wait_for_service('/path_follower/status', timeout=5.0)
            self.activate_follower = rospy.ServiceProxy('/path_follower/active_signal', Trigger)
            self.get_follower_status = rospy.ServiceProxy('/path_follower/status', Trigger)
            self.mode_pub = rospy.Publisher('/path_follower_mode', Int16, queue_size=1, latch=True)
            self.logger.loginfo("Connected to Path Follower services.")
            self.path_follower_available = True
        except (rospy.ServiceException, rospy.ROSException, rospy.ROSInterruptException) as e:
            self.logger.logwarn(f"Path Follower services not available: {e}")
            self.path_follower_available = False
        self.is_following_path = False
        rospy.Subscriber("/frontiers_data", PoseArray, self._frontiers_callback, queue_size=10)
        rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=20)
        rospy.Subscriber("/exploration_strategy", String, self._strategy_cmd_callback, queue_size=1)
        rospy.Subscriber("/success", Bool, self._stop_callback, queue_size=1)
        rospy.Subscriber("/clip/ready",Bool,self._task_callback,queue_size=1)
        rospy.Subscriber("/question_type", String, self._question_type_callback, queue_size=1)

        self.goal_pub = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=1)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1, latch=True)
        self.selected_frontier_pub = rospy.Publisher("/selected_frontier_markers", MarkerArray, queue_size=1)
        self.task_ready = False
        self.question_type = None

        self.goal_pub = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=1)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.repick_T), self._on_timer)
        
        # Strategy publisher for triggering coverage path regeneration
        self.strategy_pub = rospy.Publisher("/exploration_strategy", String, queue_size=1, latch=False)

        self.instruction_following_pub = rospy.Publisher("/instruction_following_exp_status", String, queue_size=1, latch=False)
        
        # Initialize coverage planning if it's the default strategy
        if self.exploration_strategy == "coverage_planning":
            self.logger.loginfo("Default strategy is coverage_planning - initializing coverage planning")
            rospy.Timer(rospy.Duration(1.0), self._initialize_default_coverage_planning, oneshot=True)
        
        self.logger.loginfo("Object navigation node ready")

    def _question_type_callback(self, msg: String) -> None:
        if msg.data:
            if self.question_type is None and msg.data == "instruction_following":
                self.exploration_strategy = "vg_first"
            self.question_type = msg.data
            print(f"Question type: {self.question_type}")
        else:
            self.question_type = None
            print(f"Question type not received")

    def _task_callback(self, msg: Bool) -> None:
        if msg.data:
            self.task_ready = True
            print(f" Task Ready : {self.task_ready}")
        else:
            print(f"TASK IS NOT READY") 

    def _stop_callback(self, msg: Bool) -> None:
        if msg.data and not self.stop_flag:
            self.logger.loginfo("Object navigation target found → stop publishing goals")
            self.stop_flag = True
            self.last_goal = None           
            self.goal_pub.publish(Pose2D(x=self.robot_x, y=self.robot_y, theta=0.0))
        elif (not msg.data) and self.stop_flag:
            self.logger.loginfo("Object navigation target flag reset → resume exploration")
            self.stop_flag = False
            self.last_goal = None           

    def _odom_callback(self, msg: Odometry) -> None:
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        if self.pose_ok and self._dist(self.robot_x, self.robot_y,
                                       self.last_pose_x, self.last_pose_y) > self.move_eps:
            self.last_move_t = rospy.Time.now()
            self.last_pose_x, self.last_pose_y = self.robot_x, self.robot_y
        self.pose_ok = True

        if self.last_goal and self._dist(self.robot_x, self.robot_y, *self.last_goal) < self.visited_R:
            # Special handling for instruction_following with semantic_frontier
            if self.question_type == "instruction_following" and self.exploration_strategy in ["geometric_frontier", "geometric_atsp", "semantic_frontier", "semantic_atsp"]:
                self.logger.loginfo("Frontier reached during instruction_following with frontier")
                self.logger.loginfo("Stopping exploration and signaling frontier arrival")
                
                # Stop exploration
                self.stop_flag = True
                self.closed_xy.append(self.last_goal)
                self.last_goal = None
                
                # Send frontier_arrived signal
                self.instruction_following_pub.publish("frontier_arrived")
                self.logger.loginfo("Published 'frontier_arrived' to /instruction_following_exp_status")
                
                # Change exploration strategy to vg_first
                # self.strategy_pub.publish(String("vg_first"))
                self.logger.loginfo("Changed exploration strategy to 'vg_first'")
                
                return 

            elif self.question_type != "instruction_following" and self.exploration_strategy in ["geometric_frontier", "geometric_atsp", "semantic_frontier", "semantic_atsp"]:
                self.logger.loginfo("Frontier reached during instruction_following with frontier")
                
                # Stop exploration
                self.closed_xy.append(self.last_goal)
                self.last_goal = None

                self.instruction_following_pub.publish("frontier_arrived")

                return

            else:
                # Normal behavior for other cases
                self.closed_xy.append(self.last_goal)
                self.last_goal = None


    def _strategy_cmd_callback(self, msg: String) -> None:
        new_strategy = msg.data
        if new_strategy in ["semantic_frontier", "geometric_frontier", 
                            "semantic_atsp", "geometric_atsp"]:
            
            # Check if we're currently in auto-fallback coverage mode
            if self.exploration_strategy == "coverage_planning" and self.has_switched_to_coverage:
                # If no frontiers available, defer the strategy change
                if not self.frontiers:
                    self.desired_frontier_strategy = new_strategy
                    self.mode_pub.publish(Int16(2))

                    rospy.sleep(1.0)
                    self.logger.loginfo(f"No frontiers available - deferring switch to {new_strategy} until frontiers are found")
                    self.logger.loginfo("Continuing with coverage_planning for now")
                    return
                else:
                    # Frontiers are available, switch immediately
                    self.logger.loginfo(f"Frontiers available - switching from coverage_planning to {new_strategy}")
                    self._switch_from_coverage_to_frontier(new_strategy)
                    return
            
            # Normal strategy change (not in auto-fallback mode)
            if self.exploration_strategy != new_strategy:
                if self.exploration_strategy == "stop":
                    self.stop_flag = False
                    self._ensure_pf_services(timeout_total=0.5)
                    self.is_following_path = False

                # If switching FROM coverage_planning, immediately stop coverage and reset mode
                if self.exploration_strategy == "coverage_planning":
                    self.logger.loginfo("Switching from coverage_planning to frontier strategy - stopping coverage")
                    self.is_following_path = False
                    # Reset to mode 1 immediately for frontier exploration
                    self.mode_pub.publish(Int16(1))

                    rospy.sleep(1.0)

                self.exploration_strategy = new_strategy
                self.logger.loginfo(f"Exploration strategy changed to: {self.exploration_strategy}")
                self.mode_pub.publish(Int16(1))
                self.last_goal = None
                
                # Reset no frontier fallback tracking when strategy changes
                self.no_frontier_start_time = None
                self.has_switched_to_coverage = False
                self.desired_frontier_strategy = None

        elif new_strategy in ['vg_first', 'vg_first_inference']:

            self.instruction_following_pub.publish("vg_handling")
            self.logger.loginfo("Published 'vg_handling' to /instruction_following_exp_status")

            self.exploration_strategy = "stop"
            self.logger.loginfo(f"Received {new_strategy} command → stop exploration")
            self.stop_flag = True
            self.logger.loginfo("Setting Path Follower mode to 3.")
            self.mode_pub.publish(Int16(3))

            rospy.sleep(1.0)
            
            # Reset smart recovery tracking
            self.desired_frontier_strategy = None

        elif new_strategy == "coverage_planning":
            if self.exploration_strategy != new_strategy:
                if self.exploration_strategy == "stop":
                    self.stop_flag = False
                    self._ensure_pf_services(timeout_total=0.5)
                    self.is_following_path = False

                self.exploration_strategy = "coverage_planning"
                self.logger.loginfo(f"Exploration strategy changed to: {self.exploration_strategy}")
                self.last_goal = None
                
                # Reset no frontier fallback tracking when switching to coverage planning
                self.no_frontier_start_time = None
                self.has_switched_to_coverage = False
                self.desired_frontier_strategy = None
                
                # For coverage planning, we want to stop frontier exploration but allow coverage path following
                self.stop_flag = False  # Allow timer to run for path follower status monitoring
                self._start_coverage_planning()
        else:
            self.logger.logwarn(f"Received unknown exploration strategy command: {new_strategy}. Keeping current strategy: {self.exploration_strategy}")

    def _frontiers_callback(self, msg: PoseArray) -> None:
        """
        PoseArray 메시지로부터 프론티어 데이터를 파싱하여 self.frontiers를 업데이트합니다.
        pose.position.x, y 에 위치가, pose.position.z 에 점수가 담겨있습니다.
        """
        frontiers_data = {}
        for i, pose in enumerate(msg.poses):
            x = pose.position.x
            y = pose.position.y
            score = pose.position.z  # z값에서 점수를 추출
            
            # self.frontiers 딕셔너리를 채웁니다. id는 메시지 순서대로 부여합니다.
            frontiers_data[i] = (x, y, score)
        
        self.frontiers = frontiers_data

    def _publish_selected_frontier_marker(self, best_frontier):
        """선택된 프론티어를 RViz에 눈에 띄게 표시합니다 (화살표 + 하이라이트)."""
        if best_frontier is None:
            return
        marker_array = MarkerArray()
        x, y = best_frontier

        # --- 1. 기존 마커 삭제 (잔상 방지) ---
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.selected_frontier_pub.publish(marker_array)
        
        # 새로운 마커들을 위한 MarkerArray 재생성
        marker_array = MarkerArray()

        # --- 2. 바닥 하이라이트용 원판(Cylinder) 마커 ---
        highlight_marker = Marker()
        highlight_marker.header.frame_id = self.frame_id
        highlight_marker.header.stamp = rospy.Time.now()
        highlight_marker.ns = "selected_frontier_highlight"
        highlight_marker.id = 0
        highlight_marker.type = Marker.CYLINDER
        highlight_marker.action = Marker.ADD

        highlight_marker.pose.position.x = x
        highlight_marker.pose.position.y = y
        highlight_marker.pose.position.z = 0.01  # 바닥 바로 위에 위치
        highlight_marker.pose.orientation.w = 1.0

        highlight_marker.scale.x = 1.0  # 지름 1.0m
        highlight_marker.scale.y = 1.0
        highlight_marker.scale.z = 0.02 # 매우 얇은 두께

        highlight_marker.color.r = 0.0
        highlight_marker.color.g = 1.0
        highlight_marker.color.b = 1.0  # 밝은 청록색 (Cyan)
        highlight_marker.color.a = 0.4  # 반투명

        marker_array.markers.append(highlight_marker)

        # --- 3. 위에서 아래를 가리키는 화살표(Arrow) 마커 ---
        arrow_marker = Marker()
        arrow_marker.header.frame_id = self.frame_id
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "selected_frontier_arrow"
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # 화살표의 시작점과 끝점 설정
        start_point = Point(x=x, y=y, z=1.5)  # 공중 1.5m 높이에서 시작
        end_point = Point(x=x, y=y, z=0.1)    # 지면 0.1m 높이를 가리킴
        arrow_marker.points.append(start_point)
        arrow_marker.points.append(end_point)

        arrow_marker.scale.x = 0.15  # 화살표 몸통 두께
        arrow_marker.scale.y = 0.3   # 화살촉 크기
        arrow_marker.scale.z = 0.0   # (사용 안 함)

        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 1.0  # 밝은 청록색 (Cyan)
        arrow_marker.color.a = 1.0  # 불투명

        marker_array.markers.append(arrow_marker)
        
        # 최종적으로 마커 배열 발행
        self.selected_frontier_pub.publish(marker_array)

    def _on_timer(self, _evt) -> None:

        if not self.task_ready:
            return

        if self.stop_flag:
            return

        if not self.path_follower_available:
            self.logger.loginfo("Path Follower not connected. Attempting to reconnect...")
            # 짧은 타임아웃(1초)으로 재연결을 시도하여 루프가 오래 멈추는 것을 방지
            if not self._ensure_pf_services(timeout_total=1.0):
                self.logger.logwarn("Reconnection failed. Will try again on the next cycle.")
                return # 재연결에 실패하면 이번 사이클은 건너뜁니다.
            self.logger.loginfo("Successfully reconnected to Path Follower.")
        
        if self.is_following_path:
            # Check for smart recovery even while following coverage path
            if (self.exploration_strategy == "coverage_planning" and 
                self.has_switched_to_coverage and 
                self.desired_frontier_strategy and 
                self.frontiers):
                self.logger.loginfo(f"Frontiers found during coverage! Interrupting coverage to switch to {self.desired_frontier_strategy}")
                
                # Stop current path following
                self.is_following_path = False
                try:
                    self.activate_follower()  # Deactivate path follower
                except:
                    pass
                
                # Switch to frontier strategy
                self._switch_from_coverage_to_frontier(self.desired_frontier_strategy)
                return

            # if not self.path_follower_available:
            #     self.logger.logwarn("Path Follower is not available, cannot check status. Resetting state.")
            #     self.is_following_path = False
            #     return

            try:
                status_response = self.get_follower_status()
                current_status = status_response.message
                self.logger.loginfo(f"Path Follower is running. Current status: {current_status}")

                if current_status == "completed":
                    self.logger.loginfo("Path following is complete! Ready to select a new frontier.")
                    self.is_following_path = False
                    
                    # PathFollower 비활성화
                    self.activate_follower() 
                    
                    # Check if we were doing coverage planning
                    if self.exploration_strategy == "coverage_planning":
                        self.logger.loginfo("=== COVERAGE COMPLETION DEBUG ===")
                        self.logger.loginfo(f"Strategy: {self.exploration_strategy}")
                        self.logger.loginfo(f"Coverage start time: {self.coverage_start_time}")
                        
                        # Check if we've been following the path for a meaningful amount of time
                        current_time = rospy.Time.now()
                        
                        # Check cooldown first (more important than duration)
                        if self.last_coverage_restart_time:
                            time_since_last_restart = (current_time - self.last_coverage_restart_time).to_sec()
                            self.logger.loginfo(f"Time since last restart: {time_since_last_restart:.2f}s (cooldown: {self.coverage_restart_cooldown}s)")
                            if time_since_last_restart < self.coverage_restart_cooldown:
                                self.logger.logwarn("*** RESTART BLOCKED BY COOLDOWN - will retry after timer ***")
                                # Don't return here - let the timer retry later
                                return
                        else:
                            self.logger.loginfo("No previous restart time - first restart")
                        
                        # # Check path duration (but be more lenient)
                        # if self.coverage_start_time:
                        #     path_following_duration = (current_time - self.coverage_start_time).to_sec()
                        #     self.logger.loginfo(f"Coverage path was followed for {path_following_duration:.2f} seconds")
                            
                        #     if path_following_duration < 1.0:  # Very quick completion (< 1 second)
                        #         self.logger.logwarn(f"Coverage path completed very quickly ({path_following_duration:.2f}s)")
                        #         self.logger.logwarn("This suggests robot was already at endpoint - will restart anyway to generate new path")
                        #         # Continue to restart instead of blocking
                        #     elif path_following_duration < 3.0:  # Quick but reasonable
                        #         self.logger.loginfo(f"Coverage path completed quickly ({path_following_duration:.2f}s) - proceeding with restart")
                        # else:
                        #     self.logger.logwarn("Coverage start time is None - this is unexpected but proceeding with restart")
                        
                        self.logger.loginfo("Coverage planning completed! Restarting coverage from current position...")
                        self.logger.loginfo("*** CALLING _restart_coverage_planning() ***")
                        self._restart_coverage_planning()
                        return
                
                    # 현재 목표를 방문한 것으로 처리 (for frontier strategies only)
                    if self.last_goal:
                        self.closed_xy.append(self.last_goal)
                        self.last_goal = None
                
                return 
                
            except rospy.ServiceException as e:
                self.logger.logerr(f"Failed to get Path Follower status: {e}. Resetting state.")
                self.is_following_path = False
                return

        # Check for no frontier fallback condition
        if not self.frontiers and self.exploration_strategy in ["geometric_frontier", "geometric_atsp", "semantic_frontier", "semantic_atsp"] and self.question_type != "instruction_following":
            self.logger.loginfo(f"=== NO FRONTIER FALLBACK CHECK ===")
            self.logger.loginfo(f"Frontiers: {len(self.frontiers)}, Strategy: {self.exploration_strategy}")
            self.instruction_following_pub.publish("no_frontier")
            self._check_no_frontier_fallback()
            return

        elif not self.frontiers and self.exploration_strategy in ["geometric_frontier", "geometric_atsp", "semantic_frontier", "semantic_atsp"] and self.question_type == "instruction_following":
            self.logger.loginfo(f"=== Send no frontier message to manager ===")
            self.instruction_following_pub.publish("no_frontier")
            return

        else:
            # Reset no frontier tracking if we have frontiers again
            if self.frontiers:
                if self.no_frontier_start_time is not None:
                    self.logger.loginfo(f"Frontiers found again ({len(self.frontiers)}) - resetting fallback tracking")
                self.no_frontier_start_time = None
                self.has_switched_to_coverage = False
        
        if not (self.pose_ok and self.frontiers):
            return
        
        self.logger.log(f"Exploration strategy: {self.exploration_strategy}, Frontiers: {len(self.frontiers)}, Closed: {len(self.closed_xy)}")
        
        if self.last_goal and (rospy.Time.now() - self.last_move_t).to_sec() >= self.stall_T:
            self.logger.logwarn(f"Object navigation stall {self.stall_T:.1f}s → close frontier")
            self.closed_xy.append(self.last_goal)
            self.last_goal = None

        if self.last_goal:
            return

        best, best_score = None, -1e9

        if self.exploration_strategy == "semantic_frontier":
            best, best_score = self._select_frontier_value_based()
        elif self.exploration_strategy == "geometric_frontier":
            best, best_score = self._select_frontier_distance_based()
        elif self.exploration_strategy == "geometric_atsp":
            best , best_score = self._solve_atsp_distance_based()
        elif self.exploration_strategy == "semantic_atsp":
            best , best_score = self._solve_atsp_value_based()
        elif self.exploration_strategy == "coverage_planning":
            # Coverage planning is handled separately, no frontier selection needed
            self.logger.loginfo("Coverage planning active - no frontier selection needed")
            return
        else:  
            print("WTF?")

        if best is None:
            self.logger.logwarn("No valid goal found in this cycle.")
            self._publish_selected_frontier_marker(None) if hasattr(self, '_publish_selected_frontier_marker') else None

            # Fallback to coverage planning when no valid frontier goal is found
            if self.exploration_strategy in ["semantic_frontier", "geometric_frontier", "semantic_atsp", "geometric_atsp"] and self.question_type != "instruction_following":
                self.logger.logwarn("No valid frontier goal found - switching to coverage planning as fallback")
                
                # Remember the current strategy for potential recovery
                original_strategy = self.exploration_strategy
                
                # Switch strategy internally
                self.exploration_strategy = "coverage_planning"
                self.last_goal = None
                self.has_switched_to_coverage = True
                
                # Set the desired strategy for smart recovery (only if none is already set)
                if self.desired_frontier_strategy is None:
                    self.desired_frontier_strategy = original_strategy
                    self.logger.loginfo(f"Will auto-switch back to {original_strategy} when frontiers are found")
                
                # Reset no frontier tracking
                self.no_frontier_start_time = None
                self.no_frontier_start_position = None
                
                # Trigger coverage path generation for fallback
                self.logger.loginfo("Triggering coverage path generation for no-goal fallback")
                coverage_regen_pub = rospy.Publisher("/coverage_path_regenerate", String, queue_size=1, latch=False)
                rospy.sleep(0.1)  # Brief pause to ensure publisher is ready
                
                current_time = rospy.Time.now()
                regen_msg = f"no_goal_fallback_{current_time.to_sec()}"
                coverage_regen_pub.publish(String(regen_msg))
                self.logger.loginfo(f"Sent coverage regeneration request for no-goal fallback: {regen_msg}")
                
                # Wait a moment for path generation to complete
                rospy.sleep(1.0)

                self.instruction_following_pub.publish("no_frontier")
                
                # Start coverage planning
                self._start_coverage_planning()
                return  # Important: return here to avoid duplicate execution

            return
        self._publish_selected_frontier_marker(best)

        self.logger.loginfo("Final goal selected. Requesting final path...")
        final_path, path_len = self._get_astar_path(self.robot_x, self.robot_y, best[0], best[1])

        if path_len != float('inf') and final_path.poses:
            self.logger.loginfo("Valid path found. Commanding Path Follower.")
            original_path_list = self.path_to_list(final_path)
            smoothed_path_list = self.smooth_path_moving_average(original_path_list, window_size=15)
            smoothed_path_msg = self.list_to_path(smoothed_path_list, frame_id=final_path.header.frame_id)

            if not self.path_follower_available:
                self.logger.logerr("Cannot start path following, service is not available.")
                return

            try:
                self.logger.loginfo("Activating Path Follower first...")
                
                while True: # turn on the path follower
                    response = self.activate_follower()
                    if response.success:
                        self.logger.loginfo(f"Path follower is running.")
                        break
                    
                if not response.success: # 응답이 True여야 성공
                    self.logger.logerr("Failed to activate Path Follower!")
                    return

                # 2. 경로 추종 모드 설정 (latch=True)
                self.logger.loginfo("Setting Path Follower mode to 1.")
                self.mode_pub.publish(Int16(1))
                rospy.sleep(1.0)  # 모드 설정이 반영될 시간을 잠시 대기

                # 3. 경로 발행 (latch=True)
                self.logger.loginfo("Publishing path to /planned_path.")
                self.path_pub.publish(smoothed_path_msg)

                # 4. 상태 변경 및 로그 기록
                self.is_following_path = True
                self.last_goal = best
                self.logger.loginfo(f"New goal ({best[0]:.2f}, {best[1]:.2f}) sent to Path Follower.")
                self.instruction_following_pub.publish("frontier_moving")

            except rospy.ServiceException as e:
                self.logger.logerr(f"Failed to activate Path Follower service: {e}")

        else:
            self.logger.logwarn("Could not retrieve a valid final path to the selected goal.")
  
    def _ensure_pf_services(self, timeout_total=3.0) -> bool:
        """PF 서비스가 사용 직전 항상 살아있도록 보장한다.
        - 서버가 늦게 떠도 주기적으로 복구
        - 서버 재시작/네임스페이스 변동 후 프록시 재생성
        """
        deadline = rospy.Time.now() + rospy.Duration(timeout_total)
        names = ['/path_follower/status']
        ok = False
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            try:
                for n in names:
                    rospy.wait_for_service(n, timeout=0.5)
                ok = True
                break
            except rospy.ROSException:
                pass
        self.path_follower_available = ok
        if not ok:
            return False

        # 서버가 살아있어도 소켓이 죽었을 수 있으니 프록시를 재생성해 연결을 새로 연다.
        try:
            self.activate_follower.close()  # rospy.ServiceProxy는 close() 있어도 없어도 무방
        except Exception:
            pass
        try:
            self.get_follower_status.close()
        except Exception:
            pass

        self.activate_follower    = rospy.ServiceProxy('/path_follower/active_signal', Trigger)
        self.get_follower_status  = rospy.ServiceProxy('/path_follower/status', Trigger)
        return True
      

    def _get_astar_path(self, start_x: float, start_y: float, end_x: float, end_y: float) -> (Path, float):
        """
        C++ A* 서비스 노드에 경로와 길이를 요청합니다.
        """
        try:
            req = AstarPathRequest()
            req.start_x = start_x
            req.start_y = start_y
            req.end_x = end_x
            req.end_y = end_y
            resp = self.astar_path_service(req)

            if resp.path_length < 0:
                self.logger.logwarn(f"A* service could not find a path from ({start_x:.2f}, {start_y:.2f}) to ({end_x:.2f}, {end_y:.2f}).")
                return Path(), float('inf')
            
  
            resp.path.header.stamp = rospy.Time.now()
            resp.path.header.frame_id = self.frame_id

            return resp.path, resp.path_length
        
        except rospy.ServiceException as e:
            self.logger.logerr(f"A* service call failed: {e}")
            return Path(), float('inf') 
        
    def _select_frontier_value_based(self):
        best, best_score = None, -1e9
        for x, y, v in self.frontiers.values():
            if any(self._dist(x, y, cx, cy) < self.closed_R for cx, cy in self.closed_xy):
                continue
           
            score = v
            if score > best_score:
                best, best_score = (x, y), score
        return best, best_score

    def _select_frontier_distance_based(self):
        best, best_score = None, 1e9 
        for x, y, v in self.frontiers.values():
            if any(self._dist(x, y, cx, cy) < self.closed_R for cx, cy in self.closed_xy):
                continue
            _, dist_to_robot = self._get_astar_path(self.robot_x, self.robot_y, x, y)
            score = dist_to_robot 
            if score < best_score: 
                best, best_score = (x, y), score
        return best, -best_score 
    
    def _solve_atsp_value_based(self):
        MAX_ATSP_FRONTIERS = 10 
        candidate_frontiers = []
        for fid, (x, y, v) in self.frontiers.items():
            if not any(self._dist(x, y, cx, cy) < self.closed_R for cx, cy in self.closed_xy):
                candidate_frontiers.append(((x, y), v))
        
        if not candidate_frontiers:
            return None, -1e9
        
        candidate_frontiers.sort(key=lambda item: item[1], reverse=True)
        candidate_frontiers = candidate_frontiers[:MAX_ATSP_FRONTIERS]

        if not candidate_frontiers:
            return None, -1e9

        nodes = [(self.robot_x, self.robot_y)] + [f[0] for f in candidate_frontiers]
        values = [0] + [f[1] for f in candidate_frontiers] 
        num_nodes = len(nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        value_weight = 5.0 
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                _, dist_actual = self._get_astar_path(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                if dist_actual == float('inf'):
                    cost = 1e9 
                else:
                    cost = dist_actual - (value_weight * values[j])
                distance_matrix[i, j] = max(0, cost) 

        permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
        
        if len(permutation) > 1:
            next_frontier_index = permutation[1]
            best = nodes[next_frontier_index]
            best_score = values[next_frontier_index] 
            return best, best_score
        else:
            return None, -1e9

    def _solve_atsp_distance_based(self):
        MAX_ATSP_FRONTIERS = 10 
        candidate_frontiers = []
        for fid, (x, y, v) in self.frontiers.items():
            if not any(self._dist(x, y, cx, cy) < self.closed_R for cx, cy in self.closed_xy):
                candidate_frontiers.append((x, y))
        
        if not candidate_frontiers:
            return None, -1e9
            
        candidate_frontiers.sort(key=lambda item: self._dist(self.robot_x, self.robot_y, item[0], item[1]))
        candidate_frontiers = candidate_frontiers[:MAX_ATSP_FRONTIERS]

        if not candidate_frontiers:
            return None, -1e9

        nodes = [(self.robot_x, self.robot_y)] + candidate_frontiers
        num_nodes = len(nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                _, dist_actual = self._get_astar_path(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                distance_matrix[i, j] = dist_actual if dist_actual != float('inf') else 1e9
        
        permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
        
        if len(permutation) > 1:
            next_frontier_index = permutation[1]
            best = nodes[next_frontier_index]
            _, best_score_dist = self._get_astar_path(self.robot_x, self.robot_y, best[0], best[1])
            return best, -best_score_dist 
        else:
            return None, -1e9

    def _dist(self,x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)
    
    def path_to_list(self,path_msg):
        return [[p.pose.position.x, p.pose.position.y] for p in path_msg.poses]

    def list_to_path(self,path_list, frame_id="map"):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id
        for x, y in path_list:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        return path_msg

    def smooth_path_moving_average(self,path, window_size=20):
        if window_size < 3:
            return path
            
        path_np = np.array(path)
        smoothed_path = path_np.copy()
        w = window_size // 2
        
        for i in range(w, len(path_np) - w):
            smoothed_path[i] = np.mean(path_np[i-w:i+w+1], axis=0)
            
        return smoothed_path.tolist()

    def _start_coverage_planning(self):
        """
        Start coverage path planning by setting the path follower to mode 2
        and triggering the coverage planner if needed.
        """
        try:
            if not self.path_follower_available:
                self.logger.logerr("Cannot start coverage planning, Path Follower service is not available.")
                return

            self.logger.loginfo("Activating Path Follower for coverage planning...")
            
            # Activate path follower
            while True:
                response = self.activate_follower()
                if response.success:
                    self.logger.loginfo(f"Path follower is running.")
                    break
            
            if not response.success:
                self.logger.logerr("Failed to activate Path Follower for coverage planning!")
                return

            # CRITICAL FIX: Directly trigger coverage path generation for fallback
            self.logger.loginfo("Triggering coverage path generation directly for fallback")
            # Use the dedicated regeneration topic to trigger path calculation
            coverage_regen_pub = rospy.Publisher("/coverage_path_regenerate", String, queue_size=1, latch=False)
            rospy.sleep(0.1)  # Brief pause to ensure publisher is ready

            # Set path follower to mode 2 for coverage planning
            self.logger.loginfo("Setting Path Follower mode to 2 for coverage planning.")
            self.mode_pub.publish(Int16(2))
            
            # Add a small delay to ensure mode is set
            rospy.sleep(1.0)
            
            current_time = rospy.Time.now()
            regen_msg = f"coverage_path_generate_{current_time.to_sec()}"
            coverage_regen_pub.publish(String(regen_msg))
            self.logger.loginfo(f"Sent coverage generation request for coverage planning: {regen_msg}")
            
            # Wait a moment for path generation to complete
            rospy.sleep(1.0)
            
            self.is_following_path = True
            self.coverage_start_time = rospy.Time.now()  # Track when coverage started
            self.logger.loginfo("Coverage planning initiated successfully.")

            # self.instruction_following_pub.publish(String("coverage_planning"))
            
        except rospy.ServiceException as e:
            self.logger.logerr(f"Failed to start coverage planning: {e}")

    def _restart_coverage_planning(self):
        """
        Restart coverage planning by triggering a new path generation from current robot position.
        This is used when coverage path is completed and we want to repeat it.
        """
        self.logger.loginfo("=== RESTART COVERAGE PLANNING CALLED ===")
        current_time = rospy.Time.now()
        
        if self.last_coverage_restart_time and (current_time - self.last_coverage_restart_time).to_sec() < self.coverage_restart_cooldown:
            self.logger.logwarn(f"Skipping coverage planning restart due to cooldown ({self.coverage_restart_cooldown}s).")
            return

        try:
            # Clean approach: Use a dedicated topic to signal coverage path regeneration
            # This avoids conflicts with the main exploration_strategy topic
            self.logger.loginfo("Requesting coverage path regeneration from current robot position...")

            print("restarting coverage planning")
            
            # # Create a dedicated publisher for coverage path regeneration requests
            # coverage_regen_pub = rospy.Publisher("/coverage_path_regenerate", String, queue_size=1, latch=False)
            # rospy.sleep(3.0)  # Brief pause to ensure publisher is ready
            
            # # Send regeneration request with current robot position timestamp
            # regen_msg = f"regenerate_from_position_{current_time.to_sec()}"
            # coverage_regen_pub.publish(String(regen_msg))
            # self.logger.loginfo(f"Sent coverage regeneration request: {regen_msg}")
            
            # Wait a moment for path generation to complete
            rospy.sleep(1.0)
            
            # Start the path following
            self.logger.loginfo("Starting coverage planning after regeneration...")
            self._start_coverage_planning()
            self.last_coverage_restart_time = current_time
            
            self.logger.loginfo("Coverage planning restart completed - new path should be generated from current position")
            
        except Exception as e:
            self.logger.logerr(f"Failed to restart coverage planning: {e}")
            # Fallback: just restart without forcing regeneration
            self.logger.logwarn("Using fallback restart method...")
            self._start_coverage_planning()

    def _switch_from_coverage_to_frontier(self, new_strategy):
        """
        Helper method to cleanly switch from coverage_planning to a frontier-based strategy
        """
        self.logger.loginfo("Switching from coverage_planning to frontier strategy - stopping coverage")
        self.is_following_path = False
        # Reset to mode 1 immediately for frontier exploration
        self.mode_pub.publish(Int16(1))

        rospy.sleep(1.0)

        self.exploration_strategy = new_strategy
        self.logger.loginfo(f"Exploration strategy changed to: {self.exploration_strategy}")
        self.last_goal = None
        
        # Reset smart recovery tracking
        self.no_frontier_start_time = None
        self.has_switched_to_coverage = False
        self.desired_frontier_strategy = None

    def _check_no_frontier_fallback(self):
        """
        Check if we should fallback to coverage planning when no frontiers are available
        and the robot hasn't moved for the specified timeout period.
        """
        current_time = rospy.Time.now()
        
        # Initialize tracking time if this is the first time we have no frontiers
        if self.no_frontier_start_time is None:
            self.no_frontier_start_time = current_time
            self.logger.loginfo("No frontiers detected - starting fallback timer")
            return
            
        # Check if we've already switched to coverage planning
        if self.has_switched_to_coverage:
            # If we're already in coverage mode, check if we should allow re-triggering
            # This can happen if the coverage path completes but we're still in no-frontier situation
            # if self.is_following_path:
            self.logger.loginfo("Already in coverage planning and following path - no action needed")
            return
            # else:
            #     # Remember the current strategy for potential recovery
            #     original_strategy = self.exploration_strategy
                
            #     # Switch strategy internally (don't publish to /exploration_strategy - that's for manager)
            #     self.exploration_strategy = "coverage_planning"
            #     self.last_goal = None
            #     self.has_switched_to_coverage = True
                
            #     # Reset the starting position for future no-frontier episodes
            #     self.no_frontier_start_position = None
                
            #     # Set the desired strategy for smart recovery (only if none is already set)
            #     if self.desired_frontier_strategy is None:
            #         self.desired_frontier_strategy = original_strategy
            #         self.logger.loginfo(f"Will auto-switch back to {original_strategy} when frontiers are found")
                
            #     # CRITICAL FIX: Directly trigger coverage path generation for fallback
            #     self.logger.loginfo("Triggering coverage path generation directly for fallback")
            #     # Use the dedicated regeneration topic to trigger path calculation
            #     # coverage_regen_pub = rospy.Publisher("/coverage_path_regenerate", String, queue_size=1, latch=False)
            #     # rospy.sleep(0.1)  # Brief pause to ensure publisher is ready
                
            #     # current_time = rospy.Time.now()
            #     # regen_msg = f"fallback_regenerate_{current_time.to_sec()}"
            #     # coverage_regen_pub.publish(String(regen_msg))
            #     # self.logger.loginfo(f"Sent coverage regeneration request for fallback: {regen_msg}")
                
            #     # Wait a moment for path generation to complete
            #     rospy.sleep(1.0)
                
            #     # Start coverage planning (set mode 2 and activate path follower)
            #     self._start_coverage_planning()
            
        # Calculate how long we've been without frontiers
        no_frontier_duration = (current_time - self.no_frontier_start_time).to_sec()

        # Check if robot hasn't moved significantly using distance threshold
        current_position = np.array([self.robot_x, self.robot_y])
        
        # Calculate distance from the position when we first detected no frontiers
        if hasattr(self, 'no_frontier_start_position') and self.no_frontier_start_position is not None:
            distance_moved = np.linalg.norm(current_position - self.no_frontier_start_position)
        else:
            # First time - record starting position
            self.no_frontier_start_position = current_position
            distance_moved = 0.0
        
        # Define meaningful movement threshold (e.g., 1.0 meter)
        meaningful_movement_threshold = 3.0  # meters
        robot_stuck_in_area = distance_moved < meaningful_movement_threshold
        
        self.logger.loginfo(f"No frontiers for {no_frontier_duration:.1f}s, moved {distance_moved:.2f}m from start position")
        self.logger.loginfo(f"Timeout threshold: {self.no_frontier_timeout}s")
        self.logger.loginfo(f"Movement threshold: {meaningful_movement_threshold}m")
        self.logger.loginfo(f"Duration check: {no_frontier_duration >= self.no_frontier_timeout}")
        self.logger.loginfo(f"Stuck in area check: {robot_stuck_in_area}")
        
        # Switch to coverage planning if both conditions are met
        if no_frontier_duration >= self.no_frontier_timeout and robot_stuck_in_area:
            self.logger.logwarn(f"No frontiers found for {no_frontier_duration:.1f}s and robot only moved {distance_moved:.2f}m")
            self.logger.logwarn("*** AUTOMATICALLY SWITCHING TO COVERAGE_PLANNING ***")
            
            # Remember the current strategy for potential recovery
            original_strategy = self.exploration_strategy
            
            # Switch strategy internally (don't publish to /exploration_strategy - that's for manager)
            self.exploration_strategy = "coverage_planning"
            self.last_goal = None
            self.has_switched_to_coverage = True
            
            # Reset the starting position for future no-frontier episodes
            self.no_frontier_start_position = None
            
            # Set the desired strategy for smart recovery (only if none is already set)
            if self.desired_frontier_strategy is None:
                self.desired_frontier_strategy = original_strategy
                self.logger.loginfo(f"Will auto-switch back to {original_strategy} when frontiers are found")
            
            # CRITICAL FIX: Directly trigger coverage path generation for fallback
            self.logger.loginfo("Triggering coverage path generation directly for fallback")
            # Use the dedicated regeneration topic to trigger path calculation
            # coverage_regen_pub = rospy.Publisher("/coverage_path_regenerate", String, queue_size=1, latch=False)
            # rospy.sleep(0.1)  # Brief pause to ensure publisher is ready
            
            # current_time = rospy.Time.now()
            # regen_msg = f"fallback_regenerate_{current_time.to_sec()}"
            # coverage_regen_pub.publish(String(regen_msg))
            # self.logger.loginfo(f"Sent coverage regeneration request for fallback: {regen_msg}")
            
            # Wait a moment for path generation to complete
            rospy.sleep(1.0)
            
            # Start coverage planning (set mode 2 and activate path follower)
            self._start_coverage_planning()
        else:
            self.logger.loginfo(f"Conditions not met yet - continuing to wait")
            self.logger.loginfo(f"Need: duration>={self.no_frontier_timeout} AND movement<{meaningful_movement_threshold}m")
            self.logger.loginfo(f"Have: duration={no_frontier_duration:.1f}, movement={distance_moved:.2f}m")

            

    def _initialize_default_coverage_planning(self, event):
        """
        Initialize coverage planning when it's set as the default strategy.
        This is called with a delay to ensure all services are ready.
        """
        if self.exploration_strategy == "coverage_planning":
            self.logger.loginfo("Initializing default coverage planning strategy")
            self._start_coverage_planning()

    def _coverage_retry_check(self, event):
        """
        Periodically checks if coverage planning needs to be restarted due to a failure.
        This is a backup mechanism to ensure coverage planning doesn't get stuck.
        """
        if (self.exploration_strategy == "coverage_planning" and 
            not self.is_following_path and 
            not self.stop_flag):
            
            current_time = rospy.Time.now()
            
            # Check if we haven't restarted in a while and should retry
            if self.last_coverage_restart_time:
                time_since_restart = (current_time - self.last_coverage_restart_time).to_sec()
                if time_since_restart > self.coverage_restart_cooldown * 2:  # 10 seconds
                    self.logger.logwarn("Coverage planning seems stuck - no path following for >10s. Attempting backup restart...")
                    self._restart_coverage_planning()
            elif self.coverage_start_time:
                # If we have a start time but no restart time, check if we've been idle too long
                time_since_start = (current_time - self.coverage_start_time).to_sec()
                if time_since_start > 30.0:  # 30 seconds idle
                    self.logger.logwarn("Coverage planning idle for >30s with no restarts. Attempting backup restart...")
                    self._restart_coverage_planning()

if __name__ == "__main__":
    rospy.init_node("frontier_waypoint_selector")
    WaypointSelector()
    rospy.spin()
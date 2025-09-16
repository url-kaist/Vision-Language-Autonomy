#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import ColorRGBA, String, Float64
from std_msgs.msg import Header
from scipy.spatial import cKDTree
import math
import os
import yaml
from datetime import datetime
from typing import Tuple, List

class ExplorationCoverageAnalyzer:
    def __init__(self):
        rospy.init_node('exploration_coverage_analyzer', anonymous=True)
        
        # Parameters
        self.terrain_threshold = rospy.get_param("~terrain_threshold", 0.1)
        self.terrain_topic = rospy.get_param("~terrain_topic", "/traversable_area")
        self.map_topic = rospy.get_param("~map_topic", "/mapUGV")
        self.update_rate = rospy.get_param("~update_rate", 1.0)  # Hz
        self.coverage_threshold = rospy.get_param("~coverage_threshold", 85.0)  # Coverage percentage threshold
        self.log_to_terminal = rospy.get_param("~log_to_terminal", False)  # Whether to log to terminal
        self.map_name = rospy.get_param("~map_name", "unity")  # Map/environment name
        
        # Data storage
        self.terrain_xyz = None
        self.terrain_kd = None
        self.occupancy_grid = None
        self.grid_info = None
        self.fixed_grid_info = None  # Fixed grid info for consistent calculations
        
        # Exploration tracking
        self.exploration_start_time = None
        self.last_robot_pose = None
        self.robot_trajectory_distance = 0.0
        self.coverage_threshold_reached = False
        self.logging_stopped = False  # Flag to stop logging after threshold reached
        
        # Publishers
        self.coverage_marker_pub = rospy.Publisher("/exploration_coverage_markers", MarkerArray, queue_size=1)
        self.explored_traversable_pub = rospy.Publisher("/explored_traversable_area", PointCloud2, queue_size=1)
        self.unexplored_traversable_pub = rospy.Publisher("/unexplored_traversable_area", PointCloud2, queue_size=1)
        
        # Subscribers
        self.robot_pose_sub = rospy.Subscriber("/state_estimation_at_scan", Odometry, self._robot_pose_callback, queue_size=1)
        rospy.Subscriber(self.terrain_topic, PointCloud2, self._terrain_callback, queue_size=1)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_callback, queue_size=1)
        
        # Setup logging
        self._setup_logging()
        
        # Timer for periodic analysis
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self._analysis_timer)
        
        rospy.loginfo("Exploration Coverage Analyzer initialized")
        rospy.loginfo(f"Map/Environment: {self.map_name}")
        rospy.loginfo(f"Coverage threshold set to: {self.coverage_threshold}%")
        if self.log_to_terminal:
            rospy.loginfo("Terminal logging: ENABLED (verbose)")
        else:
            rospy.loginfo("Terminal logging: DISABLED (file-only + milestones)")
            rospy.loginfo(f"Coverage data will be logged to: {self.log_file_path}")
    
    def _setup_logging(self):
        """Setup file logging for coverage data"""
        # Create logs directory if it doesn't exist
        log_dir = "/ws/external/exploration_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"exploration_coverage_{timestamp}.yaml")
        
        # Initialize log file with header
        log_header = {
            "exploration_session": {
                "start_time": datetime.now().isoformat(),
                "map_name": self.map_name,
                "coverage_threshold": self.coverage_threshold,
                "update_rate_hz": self.update_rate,
                "data_format": "timestamp, elapsed_time_seconds, coverage_percentage, trajectory_distance_meters, threshold_reached"
            }
        }
        
        with open(self.log_file_path, 'w') as f:
            yaml.dump(log_header, f, default_flow_style=False)
        
        rospy.loginfo(f"Coverage logging started: {self.log_file_path}")
    
    def _robot_pose_callback(self, msg: Odometry):
        """Callback for robot pose to track trajectory distance"""
        current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        if self.last_robot_pose is not None:
            # Calculate distance from last pose to current pose
            dx = current_pose[0] - self.last_robot_pose[0]
            dy = current_pose[1] - self.last_robot_pose[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.robot_trajectory_distance += distance
        
        self.last_robot_pose = current_pose
        
        # Set exploration start time on first pose received
        if self.exploration_start_time is None:
            self.exploration_start_time = rospy.Time.now()
            rospy.loginfo("Exploration started - beginning coverage tracking")
    
    def _terrain_callback(self, msg: PointCloud2):
        """Callback for traversable area point cloud"""
        try:
            pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)), dtype=np.float32)
            if pts.size > 0:
                self.terrain_xyz = pts
                self.terrain_kd = cKDTree(pts[:, :2])
                rospy.loginfo(f"Received terrain data: {pts.shape[0]} points")
        except Exception as e:
            rospy.logwarn(f"Error processing terrain data: {e}")
    
    def _map_callback(self, msg: OccupancyGrid):
        """Callback for occupancy grid map"""
        self.occupancy_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.grid_info = msg.info
        # rospy.loginfo(f"Received occupancy grid: {msg.info.width}x{msg.info.height}")
        
        # Set fixed grid info on first map update (for consistent calculations)
        if self.fixed_grid_info is None:
            self.fixed_grid_info = msg.info
            rospy.loginfo(f"Fixed grid info set: origin=({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f}), resolution={msg.info.resolution}")
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        if self.grid_info is None:
            return None, None
        
        ox = self.grid_info.origin.position.x
        oy = self.grid_info.origin.position.y
        res = self.grid_info.resolution
        
        gi = int((y - oy) / res)
        gj = int((x - ox) / res)
        
        return gi, gj
    
    def _grid_to_world(self, gi: int, gj: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        if self.grid_info is None:
            return None, None
        
        ox = self.grid_info.origin.position.x
        oy = self.grid_info.origin.position.y
        res = self.grid_info.resolution
        
        x = ox + (gj + 0.5) * res
        y = oy + (gi + 0.5) * res
        
        return x, y
    

    
    def _count_total_traversable_cells(self):
        """Count total traversable cells in the entire traversable area"""
        # Simply return the total number of traversable points
        return len(self.terrain_xyz) if self.terrain_xyz is not None else 0
    
    def _is_on_terrain(self, x: float, y: float, z: float = 0.0) -> bool:
        """Check if a point is on traversable terrain"""
        if self.terrain_xyz is None or self.terrain_xyz.size < 10:
            return True
        
        if self.terrain_kd is not None:
            dist_xy, idx = self.terrain_kd.query([x, y], k=1)
            if not np.isfinite(dist_xy) or idx >= self.terrain_xyz.shape[0]:
                return False
            terrain_z = float(self.terrain_xyz[int(idx), 2])
            dz = abs(terrain_z - z)
            dist_3d = math.sqrt(dist_xy**2 + dz**2)
            return dist_3d <= self.terrain_threshold
        
        # Fallback: check all terrain points
        for xyz in self.terrain_xyz:
            dx = x - xyz[0]
            dy = y - xyz[1]
            dz = z - xyz[2]
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            if dist_3d <= self.terrain_threshold:
                return True
        return False
    
    def analyze_coverage(self) -> Tuple[float, int, int, List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Analyze exploration coverage"""
        if self.occupancy_grid is None or self.terrain_xyz is None:
            return 0.0, 0, 0, [], []
        
        h, w = self.occupancy_grid.shape
        observed_traversable_cells = 0
        explored_points = []
        unexplored_points = []
        
        # For each traversable point, check if it's within the observed grid and has been observed
        for xyz in self.terrain_xyz:
            x, y, z = xyz[0], xyz[1], xyz[2]
            
            # Convert to grid coordinates
            gi, gj = self._world_to_grid(x, y)
            
            # Check if this traversable point is within the observed grid
            if gi is not None and gj is not None and 0 <= gi < h and 0 <= gj < w:
                # Check if this cell has been observed
                if self.occupancy_grid[gi, gj] != -1:  # Observed (free or occupied)
                    observed_traversable_cells += 1
                    explored_points.append((x, y, z))
                else:  # Unknown space
                    unexplored_points.append((x, y, z))
            else:
                # Point is outside observed grid - count as unexplored
                unexplored_points.append((x, y, z))
        
        # Get total traversable cells in the entire traversable area
        total_traversable_cells = len(self.terrain_xyz)  # All traversable points
        
        # Detailed debug logging
        # rospy.loginfo(f"=== DETAILED COVERAGE CALCULATION ===")
        # rospy.loginfo(f"Grid shape: {h}x{w}")
        # rospy.loginfo(f"Observed traversable cells: {observed_traversable_cells}")
        # rospy.loginfo(f"Total traversable cells: {total_traversable_cells}")
        # rospy.loginfo(f"Explored points: {len(explored_points)}")
        # rospy.loginfo(f"Unexplored points: {len(unexplored_points)}")
        
        # if self.fixed_grid_info:
        #     rospy.loginfo(f"Fixed grid origin: ({self.fixed_grid_info.origin.position.x:.2f}, {self.fixed_grid_info.origin.position.y:.2f})")
        #     rospy.loginfo(f"Fixed grid resolution: {self.fixed_grid_info.resolution}")
        
        # if self.grid_info:
        #     rospy.loginfo(f"Current grid origin: ({self.grid_info.origin.position.x:.2f}, {self.grid_info.origin.position.y:.2f})")
        #     rospy.loginfo(f"Current grid resolution: {self.grid_info.resolution}")
        
        # Calculate coverage percentage: how much of traversable area has been observed
        coverage_percentage = (observed_traversable_cells / total_traversable_cells * 100) if total_traversable_cells > 0 else 0.0
        rospy.loginfo(f"Coverage percentage: {coverage_percentage:.2f}%")
        rospy.loginfo(f"=====================================")
        
        return coverage_percentage, observed_traversable_cells, total_traversable_cells, explored_points, unexplored_points
    
    def _publish_coverage_markers(self, coverage_percentage: float, observed_cells: int, total_cells: int):
        """Publish coverage visualization markers"""
        ma = MarkerArray()
        
        # Create text marker showing coverage percentage
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "coverage_text"
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = Point(0, 0, 2.0)  # Display above origin
        text_marker.pose.orientation = Quaternion(0, 0, 0, 1)
        text_marker.scale.z = 0.5
        
        # Change color based on threshold status
        if coverage_percentage >= self.coverage_threshold:
            text_marker.color = ColorRGBA(0, 1, 0, 1)  # Green when threshold reached
        else:
            text_marker.color = ColorRGBA(1, 1, 0, 1)  # Yellow when below threshold
        
        # Add threshold status to text
        threshold_status = "THRESHOLD REACHED" if coverage_percentage >= self.coverage_threshold else f"Target: {self.coverage_threshold}%"
        text_marker.text = f"Coverage: {coverage_percentage:.1f}% ({observed_cells}/{total_cells})\n{threshold_status}"
        text_marker.lifetime = rospy.Duration(0)
        ma.markers.append(text_marker)
        
        # Add trajectory distance marker
        if self.exploration_start_time is not None:
            elapsed_time = (rospy.Time.now() - self.exploration_start_time).to_sec()
            distance_marker = Marker()
            distance_marker.header.frame_id = "map"
            distance_marker.header.stamp = rospy.Time.now()
            distance_marker.ns = "coverage_stats"
            distance_marker.id = 1
            distance_marker.type = Marker.TEXT_VIEW_FACING
            distance_marker.action = Marker.ADD
            distance_marker.pose.position = Point(0, 0, 1.5)  # Below coverage text
            distance_marker.pose.orientation = Quaternion(0, 0, 0, 1)
            distance_marker.scale.z = 0.3
            distance_marker.color = ColorRGBA(0.5, 0.5, 1, 1)  # Blue
            distance_marker.text = f"Time: {elapsed_time:.1f}s | Distance: {self.robot_trajectory_distance:.1f}m"
            distance_marker.lifetime = rospy.Duration(0)
            ma.markers.append(distance_marker)
        
        self.coverage_marker_pub.publish(ma)
    
    def _publish_explored_traversable_area(self, explored_points: List[Tuple[float, float, float]]):
        """Publish explored traversable area as PointCloud2"""
        if not explored_points:
            return
        
        # Create PointCloud2 message
        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()
        
        # Convert points to numpy array
        points_array = np.array(explored_points, dtype=np.float32)
        
        # Create PointCloud2 message
        cloud_msg = pc2.create_cloud_xyz32(header, points_array)
        self.explored_traversable_pub.publish(cloud_msg)
        
        # rospy.loginfo(f"Published explored traversable area: {len(explored_points)} points")
    
    def _publish_unexplored_traversable_area(self, unexplored_points: List[Tuple[float, float, float]]):
        """Publish unexplored traversable area as PointCloud2"""
        if not unexplored_points:
            return
        
        # Create PointCloud2 message
        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()
        
        # Convert points to numpy array
        points_array = np.array(unexplored_points, dtype=np.float32)
        
        # Create PointCloud2 message
        cloud_msg = pc2.create_cloud_xyz32(header, points_array)
        self.unexplored_traversable_pub.publish(cloud_msg)
        
        # rospy.loginfo(f"Published unexplored traversable area: {len(unexplored_points)} points")
    
    def _analysis_timer(self, event):
        """Periodic analysis timer callback"""
        coverage_percentage, observed_cells, total_cells, explored_points, unexplored_points = self.analyze_coverage()
        
        if total_cells > 0:
            # Calculate elapsed time and log data
            elapsed_time = 0.0
            if self.exploration_start_time is not None:
                elapsed_time = (rospy.Time.now() - self.exploration_start_time).to_sec()
            
            # Check if coverage threshold is reached
            threshold_reached = coverage_percentage >= self.coverage_threshold
            
            # Log to file
            self._log_coverage_data(elapsed_time, coverage_percentage, threshold_reached)
            
            # Log coverage information to terminal
            if self.log_to_terminal:
                rospy.loginfo(f"=== EXPLORATION COVERAGE ANALYSIS ===")
                rospy.loginfo(f"Elapsed time: {elapsed_time:.1f}s")
                rospy.loginfo(f"Coverage: {coverage_percentage:.2f}%")
                rospy.loginfo(f"Trajectory distance: {self.robot_trajectory_distance:.2f}m")
                rospy.loginfo(f"Observed traversable cells: {observed_cells}")
                rospy.loginfo(f"Total traversable cells: {total_cells}")
                rospy.loginfo(f"Explored areas: {len(explored_points)} points")
                rospy.loginfo(f"Unexplored areas: {len(unexplored_points)} points")
                rospy.loginfo(f"=====================================")
            
            # Check and log threshold status
            if threshold_reached and not self.coverage_threshold_reached:
                self.coverage_threshold_reached = True
                self.logging_stopped = True  # Stop logging after threshold reached
                rospy.logwarn(f"COVERAGE THRESHOLD REACHED: {coverage_percentage:.2f}% >= {self.coverage_threshold}%")
                rospy.logwarn("Exploration target coverage achieved!")
                rospy.loginfo("File logging stopped - coverage target reached!")
                
                # Log final entry with completion status
                self._log_final_entry(elapsed_time, coverage_percentage)
                
            elif threshold_reached:
                if self.log_to_terminal:
                    rospy.loginfo(f"Coverage threshold maintained: {coverage_percentage:.2f}% >= {self.coverage_threshold}%")
            
            # Always log important events regardless of log_to_terminal setting
            if not self.log_to_terminal:
                # Only log important milestones
                if coverage_percentage >= 25 and not hasattr(self, '_milestone_25'):
                    self._milestone_25 = True
                    rospy.loginfo(f"Coverage milestone: 25% reached ({coverage_percentage:.1f}%)")
                elif coverage_percentage >= 50 and not hasattr(self, '_milestone_50'):
                    self._milestone_50 = True
                    rospy.loginfo(f"Coverage milestone: 50% reached ({coverage_percentage:.1f}%)")
                elif coverage_percentage >= 75 and not hasattr(self, '_milestone_75'):
                    self._milestone_75 = True
                    rospy.loginfo(f"Coverage milestone: 75% reached ({coverage_percentage:.1f}%)")
            
            # Publish visualization
            self._publish_coverage_markers(coverage_percentage, observed_cells, total_cells)
            self._publish_explored_traversable_area(explored_points)
            self._publish_unexplored_traversable_area(unexplored_points)
    
    def _log_coverage_data(self, elapsed_time: float, coverage_percentage: float, threshold_reached: bool):
        """Log coverage data to YAML file"""
        # Stop logging if threshold reached and logging is disabled
        if self.logging_stopped:
            return
            
        try:
            # Read existing log file
            with open(self.log_file_path, 'r') as f:
                log_data = yaml.safe_load(f)
            
            # Initialize data_entries if it doesn't exist
            if 'data_entries' not in log_data['exploration_session']:
                log_data['exploration_session']['data_entries'] = []
            
            # Add new data entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "coverage_percentage": round(coverage_percentage, 2),
                "trajectory_distance_meters": round(self.robot_trajectory_distance, 2),
                "threshold_reached": threshold_reached
            }
            
            log_data['exploration_session']['data_entries'].append(entry)
            
            # Write updated log file
            with open(self.log_file_path, 'w') as f:
                yaml.dump(log_data, f, default_flow_style=False)
                
        except Exception as e:
            rospy.logerr(f"Error logging coverage data: {e}")
    
    def _log_final_entry(self, elapsed_time: float, coverage_percentage: float):
        """Log final entry when coverage threshold is reached"""
        try:
            # Read existing log file
            with open(self.log_file_path, 'r') as f:
                log_data = yaml.safe_load(f)
            
            # Add final completion entry
            final_entry = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "coverage_percentage": round(coverage_percentage, 2),
                "trajectory_distance_meters": round(self.robot_trajectory_distance, 2),
                "threshold_reached": True,
                "status": "EXPLORATION_COMPLETED"
            }
            
            # Initialize data_entries if it doesn't exist
            if 'data_entries' not in log_data['exploration_session']:
                log_data['exploration_session']['data_entries'] = []
            
            log_data['exploration_session']['data_entries'].append(final_entry)
            
            # Add completion summary to the header
            log_data['exploration_session']['completion_summary'] = {
                "completion_time": datetime.now().isoformat(),
                "final_coverage_percentage": round(coverage_percentage, 2),
                "total_exploration_time_seconds": round(elapsed_time, 2),
                "total_trajectory_distance_meters": round(self.robot_trajectory_distance, 2),
                "status": "SUCCESS"
            }
            
            # Write updated log file
            with open(self.log_file_path, 'w') as f:
                yaml.dump(log_data, f, default_flow_style=False)
                
            rospy.loginfo(f"Final completion entry logged to: {self.log_file_path}")
                
        except Exception as e:
            rospy.logerr(f"Error logging final entry: {e}")

def main():
    try:
        analyzer = ExplorationCoverageAnalyzer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main() 
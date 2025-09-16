#!/usr/bin/env python3
import rospy
import json
import yaml
import os
import open3d as o3d
import numpy as np
import inspect
import sys
sys.path.append('/ws/external/')
from tf.transformations import euler_from_quaternion
from shapely.geometry import Polygon
import similaritymeasures
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from std_msgs.msg import String
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid

from ai_module.src.visual_grounding.scripts.structures.occupancy_grid import CustomOccupancyGrid
from datetime import datetime

import sys
sys.path.append('/ws/external/ai_module/src/benchmark/scripts')
sys.path.append('/ws/external/')

from ai_module.src.utils.logger import Logger

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class BenchmarkRunner:
    def __init__(self, cfg):
        ### Ros
        rospy.init_node('benchmark_runner_node', anonymous=True)
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='Benchmark', log_path=cfg['log_path'])
        self.rate = rospy.Rate(1)  # 1 Hz

        # launch에서 넘긴 param 읽기
        scene = rospy.get_param("~scene", "home_building_2")
        question_type = rospy.get_param("~question_type", "object_reference")
        question_id = rospy.get_param("~question_id", 0)

        ### Default
        self.cfg = cfg
        self.scene = scene
        self.question_type = question_type
        self.question_id = int(question_id)
        results_dir = cfg['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        self.results_path = os.path.join(results_dir, f"{question_type}-{scene}-{question_id}.npy")
        self.results = {}
        
        # Publisher
        self.question_pub = rospy.Publisher('/challenge_question', String, queue_size=1)
        
        if self.question_type == 'object_reference':
            self.pred_box_pub = rospy.Publisher("/predicted_box", Marker, queue_size=1) # debug 
            self.gt_box_pub = rospy.Publisher("/ground_truth_box", Marker, queue_size=1) # debug
        elif self.question_type == 'instruction_following':
            self.gt_trajectory_pub = rospy.Publisher("/ground_truth_trajectory", PointCloud2, queue_size=1)
            self.pred_trajectory_pub = rospy.Publisher("/predicted_trajectory", PointCloud2, queue_size=1)

        # Subscriber
        if self.question_type == 'numerical':
            rospy.Subscriber('/answer', Int32, self._numerical_callback)
        elif self.question_type == 'object_reference':
            rospy.Subscriber('/answer', Marker, self._marker_callback)
        elif self.question_type == 'instruction_following':
            self.odom_queue = []
            self.last_callback_time = rospy.Time.now()
            self.occupancy_grid = None  # Initialize occupancy grid
            rospy.Subscriber('/state_estimation', Odometry, self._odom_callback, queue_size=10)
            self.occupancy_grid_sub = rospy.Subscriber("/mapUGV", OccupancyGrid, self._occupancy_grid_callback, queue_size=1)

        self.latest_numerical_response_time = None
        self.latest_selected_marker_time = None
        self.latest_odom_time = None
        self.start_time = rospy.Time.now()
        self.logger.loginfo(f"Benchmark runner is running!")

        ### Question and Answer
        self._get_question_answer() # get question and gt_answer, publish question to topic

        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

    def timer_callback(self, event):
        if self.question_type == 'object_reference':
            gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_yaw = self.gt_answer
            box_gt = self._get_3d_box_corners(gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_yaw)
            self.publish_box(box_gt, self.gt_box_pub, color=(0, 1, 0), ns="gt_box", marker_id=1)
        elif self.question_type == 'numerical':
            self.logger.loginfo(f"GT Answer: {self.gt_answer}")
        elif self.question_type == 'instruction_following':
            # Publish GT trajectory as point cloud
            self._publish_trajectory_pointcloud(self.gt_answer, self.gt_trajectory_pub, color=(0, 1, 0), z_offset=-2.0)
            self.logger.loginfo(f"GT Trajectory: {len(self.gt_answer)} points")
        else:
            raise NotImplementedError

    def _get_question_answer(self):
        # Load question and answer file
        question_dir = '/ws/external/questions'
        qa_path = os.path.join(question_dir, 'questions_answers.json')
        
        try:
            with open(qa_path, 'r') as file:
                qa_data = json.load(file)
        except FileNotFoundError:
            self.logger.logerr(f"Question and answer file {qa_path} not found!")
            return None
        
        scene_qa = [qa for qa in qa_data if qa['scene'] == self.scene][0]
        
        question = scene_qa['questions'][self.question_type][self.question_id]
        gt_answer = scene_qa['answers'][self.question_type][self.question_id]
        
        # Post-process the answer
        if self.question_type == 'object_reference':
            object_bbox = [
                float(gt_answer[1]), float(gt_answer[2]), float(gt_answer[3]), float(gt_answer[4]), float(gt_answer[5]), float(gt_answer[6]), float(gt_answer[7])
            ] # x, y, z, l, w, h, yaw
            gt_answer = object_bbox
        elif self.question_type == 'instruction_following':
            ply_path = os.path.join(question_dir, self.scene, gt_answer) + '.ply'
            if not os.path.exists(ply_path):
                self.logger.logerr(f"{RED}PLY file {ply_path} not found.{RESET}")
                return None
            
            # Load the trajectory from the PLY file
            pcd = o3d.io.read_point_cloud(ply_path)
            trajectory_data = o3d.geometry.PointCloud(pcd).points
            gt_answer = [list(point) for point in trajectory_data]  
            
        # Set the question and gt_answer
        self.question = question
        self.gt_answer = gt_answer
        
        # Log the question
        self.logger.loginfo(f"Scene: {self.scene}, Question ID: {self.question_id}, Question: {self.question}")

        # Save question and answer to results
        self.results['scene'] = self.scene
        self.results['question_type'] = self.question_type
        self.results['question_id'] = self.question_id
        self.results['question'] = self.question
        self.results['gt_answer'] = self.gt_answer
        
        # Publish the question to the topic
        timeout = rospy.Time.now() + rospy.Duration(20.0) # 20 seconds timeout
        while self.question_pub.get_num_connections() == 0:
            if rospy.Time.now() > timeout:
                self.logger.logwarn("No subscriber for /challenge_question topic. Proceeding anyway.")
                break
            rospy.sleep(0.1)

        self.question_pub.publish(String(data=self.question))

    ### Callbacks for responses
    def _numerical_callback(self, msg):
        self.logger.loginfo(f"Received numerical response: {msg.data}")
        
        system_answer = msg.data
        
        # Save the response to results
        self.results['system_answer'] = system_answer
        self.results['score'] = 1 if system_answer == int(self.gt_answer) else 0
        self.results['response_time'] = (rospy.Time.now() - self.start_time).to_sec()
        
        # Log the result
        if system_answer == int(self.gt_answer):
            self.logger.loginfo(f"Numerical response is correct! Expected: {self.gt_answer}, Received: {system_answer}, Score: {self.results['score']}/1")
        else:
            self.logger.logerr(f"Numerical response is incorrect! Expected: {self.gt_answer}, Received: {system_answer}, Score: {self.results['score']}/1")
        self.logger.loginfo(f"Response time: {self.results['response_time']} seconds")
        
        if self.results['response_time'] > 10 * 60:  # 10 minutes
            self.logger.logwarn(f"Response time exceeded 10 minutes!")
            
        self.save_results()
        self.logger.loginfo(f"Benchmark completed the evaluation")
        rospy.signal_shutdown("Evaluation complete")
              
    def _marker_callback(self, msg):
        self.logger.loginfo(f"Received selected object marker")
        
        # Extract the position, orientation, and scale from the marker message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        orientation = msg.pose.orientation
        l = msg.scale.x
        w = msg.scale.y
        h = msg.scale.z
        yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
        
        # Cal overlapping with ground truth
        gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_yaw = self.gt_answer
        
        self.logger.log(f"Predicted box: {x}, {y}, {z}, {l}, {w}, {h}, {yaw}")
        self.logger.log(f"Ground truth box: {gt_x}, {gt_y}, {gt_z}, {gt_l}, {gt_w}, {gt_h}, {gt_yaw}")
        
        box_pred = self._get_3d_box_corners(x, y, z, l, w, h, yaw)
        box_gt = self._get_3d_box_corners(gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_yaw)
        
        self.logger.log(f"Predicted box corners: {box_pred}")
        self.logger.log(f"Ground truth box corners: {box_gt}")

        iou_3d = self._compute_3d_iou(box_pred, box_gt)
        
        self.logger.log(f"3D IoU: {iou_3d:.4f}")
        
        # Cal score based on IoU 
        low_thres = self.cfg['object_reference']['iou_thresholds']['low']
        high_thres = self.cfg['object_reference']['iou_thresholds']['high']
        
        if iou_3d < low_thres:
            score = 0.0
        elif iou_3d > high_thres:
            score = 2.0
        else:
            score = 2.0 * (iou_3d - low_thres) / (high_thres - low_thres)
        
        # Save the response to results
        self.results['system_answer'] = [x, y, z, l, w, h, yaw]
        self.results['score'] = score
        self.results['response_time'] = (rospy.Time.now() - self.start_time).to_sec()
        
        # Log the response
        if iou_3d < low_thres:
            self.logger.logerr(f"Selected object does not match the ground truth! Selected object: {msg.ns}, IoU: {iou_3d:.4f}, Score: {score:.2f}/2.0")
        elif iou_3d > high_thres:
            self.logger.loginfo(f"Selected object matches the ground truth! Selected object: {msg.ns}, IoU: {iou_3d:.4f}, Score: {score:.2f}/2.0")
        else:
            self.logger.logwarn(f"Selected object partially matches the ground truth! Selected object: {msg.ns}, IoU: {iou_3d:.4f}, Score: {score:.2f}/2.0")
        self.logger.loginfo(f"Response time: {self.results['response_time']} seconds")
        
        if self.results['response_time'] > 10 * 60:  # 10 minutes
            self.logger.logwarn(f"Response time exceeded 10 minutes!")
            
        # Debug
        self.publish_box(box_pred, self.pred_box_pub, color=(1, 0, 0), ns="pred_box", marker_id=0)
        self.publish_box(box_gt, self.gt_box_pub, color=(0, 1, 0), ns="gt_box", marker_id=1)
            
        self.save_results()
        self.logger.loginfo(f"Benchmark completed the evaluation")
        rospy.signal_shutdown("Evaluation complete")
         
    def _odom_callback(self, msg):
        current_time = rospy.Time.now()
        if (current_time - self.last_callback_time).to_sec() <  1 / self.cfg['instruction_following']['record_frequency']:  # Avoid processing too frequently # 5 Hz
            return
        self.last_callback_time = current_time

        stamp = msg.header.stamp.to_nsec()
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        linear_velocity = msg.twist.twist.linear
        odom_data = {
            "position": np.array([position.x, position.y, position.z]),
            "orientation": np.array(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ),
            "linear_velocity": np.array(
                [linear_velocity.x, linear_velocity.y, linear_velocity.z]
            ),
        }
        
        self.odom_queue.append((stamp, odom_data))
        
        # Visualize the trajectory in RViz
        traj = [odom[1]["position"].tolist() for odom in self.odom_queue]
        self._publish_trajectory_pointcloud(traj, self.pred_trajectory_pub, color=(1, 0, 0), z_offset=-1.0)
        
        # Check if the robot has moved significantly
        if not hasattr(self, "last_movement_time"):
            self.last_movement_time = stamp
            self.last_movement_idx = len(self.odom_queue) - 1
            self.last_position = position
            return
       
        current_pos = np.array([position.x, position.y, position.z])
        last_pos = np.array([self.last_position.x, self.last_position.y, self.last_position.z])
        
        dist = np.linalg.norm(current_pos - last_pos)

        movement_threshold = self.cfg['instruction_following']['movement_threshold']
        movement_timeout = self.cfg['instruction_following']['movement_timeout']
        if dist > movement_threshold:
            # If the robot has moved more than the threshold, reset the last movement time
            self.last_position = position
            self.last_movement_idx = len(self.odom_queue) - 1
            self.last_movement_time = stamp
            return
        
        # If the robot has not moved significantly, check if it has been stationary for too long
        if stamp - self.last_movement_time >= movement_timeout * 1e9:  # Convert seconds to nanoseconds
            self.logger.logwarn(f"Robot has not moved > {movement_threshold} m for {movement_timeout} seconds. Start scoring.")
            self.odom = self.odom_queue[:self.last_movement_idx + 1]  # Remove stationary data
            
            # Calculate the trajectory score
            traj, score = self._score_trajectory()
            self.odom_queue.clear()

            self.last_movement_time = stamp
            self.last_position = position
            
            # Save the trajectory score
            self.results['score'] = score
            self.results['system_answer'] = traj
            self.results['response_time'] = (rospy.Time.now() - self.start_time).to_sec()
            
            # Log the trajectory score
            # self.write_log(f"Trajectory score: {score:.2f}/6.0", color=GREEN)
            self.logger.loginfo(f"Trajectory score (Frechet Distance): {score:.2f}/6.0")
            self.logger.loginfo(f"Response time: {self.results['response_time']} seconds")
            if self.results['response_time'] > 10 * 60:  # 10 minutes
                self.logger.logwarn(f"Response time exceeded 10 minutes!")
                
            # Visualize the trajectory in RViz
            self._publish_trajectory_pointcloud(traj, self.pred_trajectory_pub, color=(1, 0, 0), z_offset=-1.0)
            self._publish_trajectory_pointcloud(self.gt_answer, self.gt_trajectory_pub, color=(0, 1, 0), z_offset=-2.0)
            
            # Save the trajectory as image
            self._save_trajectory_image(traj, self.gt_answer, score)
            
            self.save_results()
            self.logger.loginfo(f"Benchmark completed the evaluation")
            rospy.signal_shutdown("Evaluation complete")
        else:            
            if not hasattr(self, "last_stationary_log_time"):
                self.last_stationary_log_time = stamp
            
            if stamp - self.last_stationary_log_time >= 5 * 1e9:  # 5 seconds
                self.logger.loginfo(f"Robot is stationary, waiting for movement... ({(stamp - self.last_movement_time) / 1e9:.2f} seconds)")
                self.last_stationary_log_time = stamp

    def _occupancy_grid_callback(self, msg):
        self.occupancy_grid = CustomOccupancyGrid(msg)


    ### Utility
    def _get_3d_box_corners(self, x, y, z, l, w, h, yaw):
        """Returns the 8 corner points of a 3D bounding box given its center (x, y, z), size (l, w, h), and yaw rotation."""
        dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
        corners = np.array([
            [-dx, -dy, -dz], [-dx,  dy, -dz], [ dx,  dy, -dz], [ dx, -dy, -dz],
            [-dx, -dy,  dz], [-dx,  dy,  dz], [ dx,  dy,  dz], [ dx, -dy,  dz],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # Z-axis rotation matrix
        rotated = corners @ rot.T
        return rotated + np.array([x, y, z])
        
    def _sort_polygon_ccw(self, points):
        """ Sorts a set of points in counter-clockwise order around their centroid."""
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]
        
    def _compute_3d_iou(self, corners1, corners2):
        """
        Computes the 3D IoU between two bounding boxes, each defined by 8 corner points.
        The intersection over union is calculated by combining the 2D IoU in the XY plane (using shapely)
        with the height overlap along the Z axis.
        """
        # Define the polygons in the XY plane
        xy_bottom1 = self._sort_polygon_ccw(corners1[:4, :2]) # bottom face of the box
        xy_bottom2 = self._sort_polygon_ccw(corners2[:4, :2])
        poly1 = Polygon(xy_bottom1)  
        poly2 = Polygon(xy_bottom2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        inter_area = poly1.intersection(poly2).area
        
        self.logger.log(f"Intersection area: {inter_area:.4f}, Poly1 area: {poly1.area:.4f}, Poly2 area: {poly2.area:.4f}")
        if inter_area == 0.0:
            return 0.0

        # Calculate the height overlap
        z1_min, z1_max = np.min(corners1[:, 2]), np.max(corners1[:, 2])
        z2_min, z2_max = np.min(corners2[:, 2]), np.max(corners2[:, 2])
        inter_h = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
        
        self.logger.log(f"Height overlap: {inter_h:.4f}, Z1: [{z1_min:.4f}, {z1_max:.4f}], Z2: [{z2_min:.4f}, {z2_max:.4f}]")

        inter_vol = inter_area * inter_h
        vol1 = poly1.area * (z1_max - z1_min)
        vol2 = poly2.area * (z2_max - z2_min)
        union_vol = vol1 + vol2 - inter_vol
        
        self.logger.log(f"Volume 1: {vol1:.4f}, Volume 2: {vol2:.4f}, Union volume: {union_vol:.4f}")

        return inter_vol / union_vol if union_vol > 0 else 0.0

    def _score_trajectory(self):
        # Get the trajectory data from the odometry queue
        actual_traj = [odom[1]["position"].tolist() for odom in self.odom_queue]
        if len(actual_traj) < 2:
            self.logger.logerr("Not enough trajectory data to compute score.")
            return actual_traj, 0.0
        
        # Calculate the trajectory score
        actual_xy = [[float(p[0]), float(p[1])] for p in actual_traj]
        gt_xy = [[float(p[0]), float(p[1])] for p in self.gt_answer]

        actual_xy = np.array(actual_xy)
        gt_xy = np.array(gt_xy)
        distance = similaritymeasures.frechet_dist(actual_xy, gt_xy)
        
        gt_total_travel_distance = self._compute_travel_distance(gt_xy)
        normalized_distance = distance / gt_total_travel_distance

        max_score = 6.0
        score = max_score * np.exp(-normalized_distance)
        score = round(min(score, max_score), 2)

        return actual_traj, score
    
    def _compute_travel_distance(self, traj):
        traj = np.array(traj)
        return sum(np.linalg.norm(traj[i] - traj[i-1]) for i in range(1, len(traj)))


    def publish_box(self, corners, publisher, frame_id="map", color=(0, 1, 0), ns="box", marker_id=0):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02 

        # Color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        # Create the 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ]

        for i1, i2 in edges:
            p1 = Point(x=corners[i1][0], y=corners[i1][1], z=corners[i1][2])
            p2 = Point(x=corners[i2][0], y=corners[i2][1], z=corners[i2][2])
            marker.points.append(p1)
            marker.points.append(p2)

        publisher.publish(marker)

    def _publish_trajectory_pointcloud(self, trajectory, publisher, color=(0, 1, 0), z_offset=-2.0):
        """Publish trajectory as colored point cloud for RViz visualization"""
        if not trajectory or len(trajectory) == 0:
            return
            
        # Convert trajectory to numpy array
        points = np.array(trajectory)
        
        # Apply z offset to separate trajectories visually
        points[:, 2] += z_offset
        
        # Create colored point cloud
        colors = np.tile(color, (len(points), 1))  # Repeat color for all points
        
        # Create point cloud message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        
        # Create fields for xyz and rgb
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        
        # Pack RGB into uint32 (ensure values are in 0-255 range)
        rgb_values = np.clip(colors * 255, 0, 255).astype(np.uint8)
        rgb_uint32 = (
            (rgb_values[:, 2].astype(np.uint32) << 16) |
            (rgb_values[:, 1].astype(np.uint32) << 8) |
            (rgb_values[:, 0].astype(np.uint32))
        )
        
        # Create structured array with proper data types
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32), 
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        
        cloud_data['x'] = points[:, 0].astype(np.float32)
        cloud_data['y'] = points[:, 1].astype(np.float32)
        cloud_data['z'] = points[:, 2].astype(np.float32)
        cloud_data['rgb'] = rgb_uint32.astype(np.uint32)
        
        # Create point cloud message
        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        publisher.publish(cloud_msg)
    
    def _save_trajectory_image(self, pred_trajectory, gt_trajectory, score):
        """Save trajectory visualization as image with GT and predicted trajectories"""
        if not pred_trajectory or not gt_trajectory:
            return
            
        # Always save simple trajectory version
        self._save_simple_trajectory_image(pred_trajectory, gt_trajectory, score)
        
        # Also save occupancy grid version if available
        if hasattr(self, 'occupancy_grid') and self.occupancy_grid is not None:
            self._save_occupancy_grid_trajectory_image(pred_trajectory, gt_trajectory, score)
        else:
            self.logger.logwarn("Occupancy grid not available, only simple trajectory image saved")
    
    def _save_occupancy_grid_trajectory_image(self, pred_trajectory, gt_trajectory, score):
        """Save trajectory visualization on occupancy grid map"""
        # Convert trajectories to numpy arrays
        pred_points = np.array(pred_trajectory)
        gt_points = np.array(gt_trajectory)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Display occupancy grid as background
        grid_data = self.occupancy_grid.grid_data
        origin_x = self.occupancy_grid.info.origin.position.x
        origin_y = self.occupancy_grid.info.origin.position.y
        resolution = self.occupancy_grid.info.resolution
        
        # Create extent for imshow (left, right, bottom, top)
        extent = [
            origin_x, 
            origin_x + grid_data.shape[1] * resolution,
            origin_y, 
            origin_y + grid_data.shape[0] * resolution
        ]
        
        # Display occupancy grid (0=free, 100=occupied, -1=unknown)
        # Convert to RGB for better visualization
        grid_rgb = np.zeros((grid_data.shape[0], grid_data.shape[1], 3), dtype=np.uint8)
        grid_rgb[grid_data == 0] = [255, 255, 255]  # Free space - white
        grid_rgb[grid_data == 100] = [0, 0, 0]      # Occupied - black
        grid_rgb[grid_data == -1] = [128, 128, 128] # Unknown - gray
        
        ax.imshow(grid_rgb, extent=extent, origin='lower', alpha=0.7)
        
        # Plot trajectories on top of occupancy grid
        ax.plot(gt_points[:, 0], gt_points[:, 1], 'g-', linewidth=3, 
                label=f'Ground Truth Trajectory ({len(gt_points)} points)', alpha=0.9)
        ax.plot(pred_points[:, 0], pred_points[:, 1], 'r-', linewidth=2, 
                label=f'Predicted Trajectory ({len(pred_points)} points)', alpha=0.9)
        
        # Mark start and end points
        ax.plot(gt_points[0, 0], gt_points[0, 1], 'go', markersize=10, label='GT Start')
        ax.plot(gt_points[-1, 0], gt_points[-1, 1], 'gs', markersize=10, label='GT End')
        ax.plot(pred_points[0, 0], pred_points[0, 1], 'ro', markersize=8, label='Pred Start')
        ax.plot(pred_points[-1, 0], pred_points[-1, 1], 'rs', markersize=8, label='Pred End')
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Trajectory Comparison on Occupancy Grid - Score: {score:.2f}/6.0\n'
                    f'Scene: {self.scene}, Question ID: {self.question_id}', fontsize=14)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper right')
        
        # Add score text box
        score_text = f'Frechet Distance Score: {score:.2f}/6.0'
        ax.text(0.02, 0.98, score_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save image with occupancy grid suffix
        image_dir = os.path.join(self.cfg['results_dir'], 'trajectory_images')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{self.question_type}-{self.scene}-{self.question_id}_occupancy_grid.png")
        
        plt.tight_layout()
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.loginfo(f"Trajectory image with occupancy grid saved to {image_path}")
    
    def _save_simple_trajectory_image(self, pred_trajectory, gt_trajectory, score):
        """Fallback method for simple trajectory visualization without occupancy grid"""
        # Convert trajectories to numpy arrays
        pred_points = np.array(pred_trajectory)
        gt_points = np.array(gt_trajectory)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot trajectories
        ax.plot(gt_points[:, 0], gt_points[:, 1], 'g-', linewidth=3, 
                label=f'Ground Truth Trajectory ({len(gt_points)} points)', alpha=0.8)
        ax.plot(pred_points[:, 0], pred_points[:, 1], 'r-', linewidth=2, 
                label=f'Predicted Trajectory ({len(pred_points)} points)', alpha=0.8)
        
        # Mark start and end points
        ax.plot(gt_points[0, 0], gt_points[0, 1], 'go', markersize=10, label='GT Start')
        ax.plot(gt_points[-1, 0], gt_points[-1, 1], 'gs', markersize=10, label='GT End')
        ax.plot(pred_points[0, 0], pred_points[0, 1], 'ro', markersize=8, label='Pred Start')
        ax.plot(pred_points[-1, 0], pred_points[-1, 1], 'rs', markersize=8, label='Pred End')
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Trajectory Comparison - Score: {score:.2f}/6.0\n'
                    f'Scene: {self.scene}, Question ID: {self.question_id}', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axis('equal')
        
        # Add score text box
        score_text = f'Frechet Distance Score: {score:.2f}/6.0'
        ax.text(0.02, 0.98, score_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save image
        image_dir = os.path.join(self.cfg['results_dir'], 'trajectory_images')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{self.question_type}-{self.scene}-{self.question_id}.png")
        
        plt.tight_layout()
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.loginfo(f"Trajectory image saved to {image_path}")

    def save_results(self):
        # results: scene, question_type, question_id, question, gt_answer, system_answer, score, response_time
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)        
        np.save(self.results_path, self.results)
        self.logger.loginfo(f"Results saved to {self.results_path}")


if __name__ == '__main__':
    config_path = rospy.get_param('~config', '/ws/external/ai_module/src/benchmark/config/benchmark.yaml')
    try:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
    except FileNotFoundError:
        rospy.logerr(f"{RED}Configuration file {config_path} not found.{RESET}")
        exit(1)
    
    node = BenchmarkRunner(cfg)
    rospy.spin()


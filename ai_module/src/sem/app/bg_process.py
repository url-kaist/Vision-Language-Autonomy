import numpy as np
from collections import defaultdict
from utils import voxel_downsample
import time
from sklearn.linear_model import RANSACRegressor

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import rospy
from collections import deque
from scipy import ndimage
import cv2
from datetime import datetime
import os

class BGProcessor:
    def __init__(self, voxel_size=0.1, grid_size=0.1, z_threshold=0.1, map_width=200, map_height=200, map_origin=(-12.5, -5.0)):
        self.voxel_size = voxel_size
        self.grid_size = grid_size

        self.floor_cloud = np.empty((0, 3), dtype=np.float32)
        self.wall_cloud = np.empty((0, 3), dtype=np.float32)

        self.wall_grids = set()
        self.wall_grid_dict = defaultdict(set)

        self.detected_planes = []

        self.occupancy_grid_size = 0.1
        self.z_threshold = z_threshold
        self.map_width = map_width
        self.map_height = map_height
        self.map_origin = map_origin  # (x, y)

        self.occupancy_grid_pub = rospy.Publisher("/bg_occupancy_map", OccupancyGrid, queue_size=1)
        self.occupancy_data = [0] * (self.map_width * self.map_height)
        self.occupancy_grid_msg = self.init_occupancy_grid()

        self.grid_pub = rospy.Publisher("/wall_grid", OccupancyGrid, queue_size=1)
        self.grid_data = [0] * (self.map_width * self.map_height)
        self.grid_msg = self.init_occupancy_grid()

    def init_occupancy_grid(self):
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = self.occupancy_grid_size
        grid.info.width = self.map_width
        grid.info.height = self.map_height

        origin = Pose()
        origin.position.x = self.map_origin[0]
        origin.position.y = self.map_origin[1]
        origin.position.z = 0.0
        grid.info.origin = origin

        return grid

    def update_occupancy_grid(self, cloud: np.ndarray):
        origin_x, origin_y = self.map_origin

        # occupied_grids = []
        for pt in cloud:
            if pt[2] < self.z_threshold or pt[2] > 1.5:
                continue
            gx = int((pt[0] - origin_x) / self.occupancy_grid_size)
            gy = int((pt[1] - origin_y) / self.occupancy_grid_size)
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                idx = gy * self.map_width + gx
                if self.occupancy_data[idx] != 100:
                    self.occupancy_data[idx] = 100
                    # occupied_grids.append((gx, gy))

        map_array = np.array(self.occupancy_data).reshape(self.map_height, self.map_width)
        filled_map = map_array.copy()
        filled_map[0, :] = 0
        filled_map[-1, :] = 0
        filled_map[:, 0] = 0
        filled_map[:, -1] = 0

        filled_map = ndimage.binary_fill_holes(map_array == 100).astype(int)

        labeled_array, num_features = ndimage.label(filled_map)
        for region_num in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_num)
            if region_size > 10000: #assume object smaller than 10mx10m
                filled_map[labeled_array == region_num] = 0

        filled_map[filled_map > 0] = 100
        self.occupancy_data = filled_map.flatten()

        self.occupancy_grid_msg.header.stamp = rospy.Time.now()
        self.occupancy_grid_msg.data = self.occupancy_data
        self.occupancy_grid_pub.publish(self.occupancy_grid_msg)
        self.save_occupancy_grid_image(self.occupancy_data, self.map_width, self.map_height)


    def save_occupancy_grid_image(self, map_data: list, width: int, height: int, save_dir="/ws/external/occupancy_map"):
        os.makedirs(save_dir, exist_ok=True)
        grid_array = np.array(map_data, dtype=np.uint8).reshape((height, width))

        image = np.full_like(grid_array, 255, dtype=np.uint8)
        # image[grid_array == 0] = 255

        image[grid_array == 100] = 0
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"occmap_{now}.png")
        # cv2.imwrite(filename, image)


    def accumulate_floor(self, cloud: np.ndarray):
        """Floor accumulation logic"""
        mask = cloud[:, 2] < 0.01  # z 값이 낮은 것들을 바닥으로 처리
        floor_points = cloud[mask]
        if floor_points.shape[0] > 0:
            self.floor_cloud = np.vstack([self.floor_cloud, floor_points])
            self.floor_cloud = voxel_downsample(
                self.floor_cloud, voxel_size=self.voxel_size
            )

    def accumulate_wall(self, cloud: np.ndarray, min_points=10, min_z_range=1.7):
        """Wall accumulation logic"""
        # self.wall_cloud = voxel_downsample(self.wall_cloud, voxel_size=0.05)
        gx = np.floor((cloud[:, 0] - self.map_origin[0]) / self.grid_size).astype(np.int32)
        gy = np.floor((cloud[:, 1] - self.map_origin[1]) / self.grid_size).astype(np.int32)
        keys = list(zip(gx, gy))

        grid_dict = defaultdict(list)
        for idx, key in enumerate(keys):
            grid_dict[key].append(idx)

        new_wall_points = []

        for key, idxs in grid_dict.items():
            if key in self.wall_grids:
                continue
            if len(idxs) < min_points:
                continue

            pts = cloud[idxs]
            z_vals = pts[:, 2]
            z_range = np.max(z_vals) - np.min(z_vals)

            if z_range >= min_z_range:
                new_wall_points.append(pts)
                self.wall_grids.add(key)

        if new_wall_points:
            new_wall_points = np.vstack(new_wall_points)
            self.wall_cloud = np.vstack([self.wall_cloud, new_wall_points])
            self.wall_cloud = voxel_downsample(
                self.wall_cloud, voxel_size=self.voxel_size
            )

        return new_wall_points

    def publish_wall_grid(self):
        # origin_x, origin_y = self.map_origin

        for key in self.wall_grids:
            gx, gy = key
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                idx = gy * self.map_width + gx
                if self.grid_data[idx] != 100:
                    self.grid_data[idx] = 100

        # self.grid_msg.info.origin.position.x = self.map_origin[0]
        # self.grid_msg.info.origin.position.y = self.map_origin[1]
        # self.grid_msg.info.resolution = self.grid_size

        self.grid_msg.header.stamp = rospy.Time.now()
        self.grid_msg.data = self.grid_data
        self.grid_pub.publish(self.grid_msg)

        # grid_img = Image.fromarray()

    def detect_planes(self, cloud: np.ndarray, distance_threshold=0.05):
        ransac = RANSACRegressor()
        x_y_points = cloud[:, :2]  # x, y 좌표만 고려하여 평면 추출
        ransac.fit(x_y_points, cloud[:, 2])  # z값을 예측

        # 평면을 추출하여 법선 벡터와 함께 저장
        inlier_mask = ransac.inlier_mask_
        inlier_points = cloud[inlier_mask]
        plane_normal = ransac.estimator_.coef_

        # 평면을 추출하여 벽 점들을 구간별로 나누어 저장
        if self.detected_planes:
            for existing_plane in self.detected_planes:
                # 기존 평면의 법선 벡터와의 차이를 계산
                angle_diff = np.arccos(
                    np.clip(np.dot(existing_plane["normal"], plane_normal), -1.0, 1.0)
                )
                if angle_diff < np.pi / 18:  # 10도 이하 차이는 같은 평면으로 간주
                    existing_plane["points"].append(inlier_points)  # 기존 평면에 추가
                    break
            else:
                # 새로운 평면 추가
                self.detected_planes.append(
                    {"normal": plane_normal, "points": [inlier_points]}
                )
        else:
            # 첫 번째 평면 추가
            self.detected_planes.append(
                {"normal": plane_normal, "points": [inlier_points]}
            )

    def process_bg_cloud(self, cloud: np.ndarray):
        """Process incoming point cloud data"""
        t3 = time.time()
        cloud = voxel_downsample(cloud, voxel_size=self.voxel_size)
        t0 = time.time()
        self.accumulate_floor(cloud)
        t1 = time.time()
        new_wall_points = self.accumulate_wall(cloud)
        t2 = time.time()
        self.update_occupancy_grid(cloud)
        # self.detect_planes(new_wall_points)
        t3 = time.time()
        self.publish_wall_grid()
        # print(f"downsample time: {t0 - t3:.4f}s")
        # print(f"Floor processing time: {t1 - t0:.4f}s")
        # print(f"Wall processing time: {t2 - t1:.4f}s")
        # print(f"update_occupancy_grid time: {t3 - t2:.4f}s")
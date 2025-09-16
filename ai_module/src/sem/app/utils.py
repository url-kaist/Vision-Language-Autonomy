import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import open3d as o3d
import std_msgs.msg
import math
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree


def load_depth_intrinsics(H, W):
    hfov = 90 * np.pi / 180
    vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
    fx = W / (2.0 * np.tan(hfov / 2.0))
    fy = H / (2.0 * np.tan(vfov / 2.0))
    cx = W / 2
    cy = H / 2
    depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return depth_camera_matrix

def compute_iou_3d(min1, max1, min2, max2):
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.clip(inter_max - inter_min, a_min=0, a_max=None)
    inter_vol = np.prod(inter_size)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol

def is_included(min1, max1, min2, max2):
    return (np.all(min1 >= min2) and np.all(max1 <= max2)) or (
        np.all(min2 >= min1) and np.all(max2 <= max1)
    )

def compute_overlap_ratio(min1, max1, min2, max2):
    if np.any(max1-min1 < 0.05): 
        max1 += 0.05
    if np.any(max2-min2 < 0.05): 
        max2 += 0.05
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.clip(inter_max - inter_min, a_min=0, a_max=None)
    inter_vol = np.prod(inter_size)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    if vol1 <= 0 or vol2 <= 0:
        return 0.0
    ratio = max(inter_vol / vol1, inter_vol / vol2)
    return ratio

def check_close_points(cloud1, cloud2, distance_threshold=0.2, required_close_points=3):
    tree = KDTree(cloud1)
    close_point_count = 0

    for point in cloud2:
        indices = tree.query_ball_point(point, r=distance_threshold)
        if len(indices) > 0:
            close_point_count += 1
        if close_point_count > required_close_points:
            return True
        
    return False

def convert_pointcloud2_to_xyz(cloud_msg):
    points = list(
        pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    )
    return np.array(points).reshape(-1, 3)
import struct

# JW

def get_rgb_from_pointcloud(cloud_msg):
    # Check if the 'rgb' field exists in the point cloud
    has_rgb = any(f.name == 'rgb' for f in cloud_msg.fields)

    if not has_rgb:
        # Handle the case where there is no 'rgb' field
        return None

    points_with_rgb = pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)

    xyz = []
    bgr = []
    for p in points_with_rgb:
        xyz.append(p[:3])
        rgb_float_bytes = struct.pack('f', p[3])
        rgb_int = struct.unpack('I', rgb_float_bytes)[0]

        r = (rgb_int >> 16) & 0x0000ff
        g = (rgb_int >> 8) & 0x0000ff
        b = (rgb_int) & 0x0000ff
        bgr.append([b, g, r])
    return np.array(xyz).reshape(-1, 3), np.array(bgr)

def transform_to_body(cloud, R_w2b, t_w2b):
    return cloud @ R_w2b.T + t_w2b


def voxel_downsample(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    return downsampled_points


def cluster_dbscan(cloud, eps=0.2, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cloud)
    unique, counts = np.unique(
        clustering.labels_[clustering.labels_ != -1], return_counts=True
    )
    if len(counts) == 0:
        return np.array([])

    best_label = unique[np.argmax(counts)]
    return cloud[clustering.labels_ == best_label]

def find_overlapping_points_dbscan(pc1, pc2, eps=0.2, min_samples=10):
    """
    DBSCAN을 사용하여 두 점군(point cloud) 배열의 겹치는 부분을 찾습니다.

    Args:
        pc1 (np.array): 첫 번째 점군 배열 (N x 3).
        pc2 (np.array): 두 번째 점군 배열 (M x 3).
        eps (float): DBSCAN에서 이웃을 정의하는 최대 거리.
        min_samples (int): 클러스터로 간주되기 위한 최소 샘플 수.

    Returns:
        bool: 겹치는 부분이 있으면 True, 없으면 False.
    """
    combined_pc = np.concatenate((pc1, pc2), axis=0)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(combined_pc)
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    print(f"len(unique_labels) {len(unique_labels)}")
    if len(unique_labels) > 1:
        return False
    # for label in unique_labels:
    #     cluster_indices = np.where(labels == label)[0]
        
    #     pc1_points_in_cluster = np.sum(cluster_indices < len(pc1))
    #     pc2_points_in_cluster = np.sum(cluster_indices >= len(pc1))
        
    #     if pc1_points_in_cluster > 0 and pc2_points_in_cluster > 0:
    #         # print(f"겹치는 부분 발견! 클러스터 레이블: {label}")
    #         return True
            
    return True


def get_distance_between_poses(pose1, pose2):
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    return np.linalg.norm(pos1 - pos2)


def get_angle_between_poses(pose1, pose2):
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    relative_rot = rot1.inv() * rot2
    angle_rad = relative_rot.magnitude()  # 회전 각도 (radian)
    return np.degrees(angle_rad)

def fast_voxel_downsample(points, voxel_size=0.05):
    if len(points) == 0:
        return points

    coords = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(coords, axis=0, return_index=True)
    return points[indices]

import cv2
import matplotlib.pyplot as plt
import os

def distance_transform(occupancy_map, reselotion, tmp_path, visualize=False):
    """
        Perform distance transform on the occupancy map to find the distance of each cell to the nearest occupied cell.
        :param occupancy_map: 2D numpy array representing the occupancy map.
        :param reselotion: The resolution of each cell in the grid map in meters.
        :param path: The path to save the distance transform image.
        :return: The distance transform of the occupancy map.
    """

    # print("occupancy_map shape: ", occupancy_map.shape)
    # for i in range(occupancy_map.shape[0]):
        # print(occupancy_map[i, :])
    # occupancy_map = np.where(occupancy_map < 128, 1, 0).astype(np.uint8)
    bw = occupancy_map.copy()
    full_map = occupancy_map.copy()

    # invert the image
    # bw = np.where(occupancy_map > 0, 1, 0).astype(np.uint8)
    # bw = cv2.bitwise_not(bw)

    # Perform the distance transform algorithm
    bw = np.uint8(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # print("range of dist: ", np.min(dist), np.max(dist))
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    if visualize:
        plt.figure()
        plt.imshow(dist, cmap="jet", origin="lower")
        plt.savefig(os.path.join(tmp_path, "dist.png"))

    dist = np.uint8(dist)
    # apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(dist, (11, 1), 10)
    if visualize:
        plt.figure()
        plt.imshow(blur, cmap="jet", origin="lower")
        plt.savefig(os.path.join(tmp_path, "dist_blur.png"))
    _, dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if visualize:
        plt.figure()
        plt.imshow(dist, cmap="jet", origin="lower")
        plt.savefig(os.path.join(tmp_path, "dist_thresh.png"))

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype("uint8")
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("number of seeds, aka rooms: ", len(contours))

    # print the area of each seed
    # for i in range(len(contours)):
        # print("area of seed {}: ".format(i), cv2.contourArea(contours[i]))

    # remove small seed contours
    min_area_m = 0.5
    min_area = (min_area_m / reselotion) ** 2
    # print("min_area: ", min_area)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    # print("number of contours after remove small seeds: ", len(contours))

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    circle_radius = 1  # in pixels
    cv2.circle(markers, (3, 3), circle_radius, len(contours) + 1, -1)

    # Perform the watershed algorithm
    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map, markers)

    if visualize:
        plt.figure()
        plt.imshow(markers, cmap="jet", origin="lower")
        plt.savefig(os.path.join(tmp_path, "markers.png"))

    # find the vertices of each room
    room_vertices = []
    for i in range(len(contours)):
        room_vertices.append(np.where(markers == i + 1))
    room_vertices = np.array(room_vertices, dtype=object).squeeze()
    print("room_vertices shape: ", room_vertices.shape)

    return room_vertices


import math

def get_bounding_box_corners(x, y, length, width, orientation):
    # orientation 기준: length 방향이 x축 기준 회전 각 (라디안)
    # length 방향 축벡터
    dx = math.cos(orientation)
    dy = math.sin(orientation)

    # width 방향 축벡터 (수직)
    wx = -dy
    wy = dx

    # 각 변의 절반 크기
    hl = length / 2
    hw = width / 2

    # 4개 꼭짓점 좌표 계산 (시계 방향 or 반시계 방향)
    # 좌상(x1,y1), 우하(x2,y2) 기준 bounding box로 쓸 수 있도록 min/max 좌표도 함께 구함
    corners = []
    corners.append((x - hl*dx - hw*wx, y - hl*dy - hw*wy))  # 왼쪽 아래
    corners.append((x - hl*dx + hw*wx, y - hl*dy + hw*wy))  # 오른쪽 아래
    corners.append((x + hl*dx + hw*wx, y + hl*dy + hw*wy))  # 오른쪽 위
    corners.append((x + hl*dx - hw*wx, y + hl*dy - hw*wy))  # 왼쪽 위

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]

    def edge_length(e):
        (ax, ay), (bx, by) = e
        return math.hypot(bx - ax, by - ay)

    lengths = [edge_length(e) for e in edges]

    idx_short = lengths.index(min(lengths))
    idx_opposite = (idx_short + 2) % 4

    short_edge = edges[idx_short]
    opposite_edge = edges[idx_opposite]

    (ax_s, ay_s), (bx_s, by_s) = short_edge
    cx_short = (ax_s + bx_s) / 2
    cy_short = (ay_s + by_s) / 2

    (ax_o, ay_o), (bx_o, by_o) = opposite_edge
    cx_opposite = (ax_o + bx_o) / 2
    cy_opposite = (ay_o + by_o) / 2

    center_line = [
        (cx_short, cy_short),
        (cx_opposite, cy_opposite)
    ]


    return (x1, y1), (x2, y2), corners, center_line



def generate_random_position(center, instance_id, class_label, radius=0.5):
    """
    Generate a random position within a given radius of the center.
    :param center: List of [x, y, z] coordinates representing the center.
    :param radius: The radius within which the random position is generated.
    :return: A list of [x, y, z] representing the random position.
    """
    unique_seed = hash((instance_id, class_label))  # Combine instance_id and class_label for a unique hash
    random.seed(unique_seed)

    x_offset = random.uniform(-radius, radius)
    y_offset = random.uniform(-radius, radius)

    # Ensure that the generated point is within the radius (simple Euclidean check)
    while x_offset**2 + y_offset**2 > radius**2:
        x_offset = random.uniform(-radius, radius)
        y_offset = random.uniform(-radius, radius)

    random_position = [center[0] + x_offset, center[1] + y_offset, center[2]]
    return random_position

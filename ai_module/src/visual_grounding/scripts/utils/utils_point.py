import struct
import numpy as np

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


def filter_close_points(points, min_dist=0.5):
    """
    points: (N, 2) or (N, 3) ndarray
    min_dist: minimum distance between two points
    """
    filtered = [points[0]]
    for p in points[1:]:
        dists = np.linalg.norm(np.array(filtered) - p, axis=1)
        if np.all(dists >= min_dist):
            filtered.append(p)
    return np.array(filtered)


def sample_points_from_polygon_exterior(polygon, step_size=0.5):
    boundary = polygon.exterior
    total_length = boundary.length
    num_samples = int(total_length // step_size) + 1

    sampled_points = []
    for i in range(num_samples):
        distance = i * step_size
        point = boundary.interpolate(distance)
        sampled_points.append([point.x, point.y, 0.0])

    return np.array(sampled_points)


def find_closest_point(query_points, key_points, extra_margin=0.0):
    """
    query_points: (N, 3) np.array
    key_points: (M, 3) np.array
    """
    N, Ndim = query_points.shape
    M, Mdim = key_points.shape
    if Ndim > Mdim:
        key_points = np.hstack([key_points, np.zeros((M, 1))])
    elif Mdim > Ndim:
        query_points = np.hstack([query_points, np.zeros((N, 1))])

    diff = key_points[None, :, :] - query_points[:, None, :]  # (N, M, 2)
    dists = np.linalg.norm(diff, axis=2)  # (N, M)

    closest_idxs = np.argmin(dists, axis=1)  # (N,)

    closest_points = key_points[closest_idxs]  # (N, 2)
    if extra_margin > 0.0:
        vectors = closest_points - query_points
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        offsets = vectors / norms * extra_margin  # (N, 2)
        closest_points = closest_points + offsets

    return closest_points


def pointcloud2_to_xy_array(cloud_msg):
    points = []
    for pt in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y")):
        points.append([pt[0], pt[1]])
    has_traversable_area = False if len(points) == 0 else True
    return np.array(points).reshape(-1, 2), has_traversable_area


def make_msg_from_points(points, frame_id="map", color=None):
    """
    Create a sensor_msgs/PointCloud2 message from a numpy array.

    Args:
        points (np.ndarray): shape (N, 2) or (N, 3)
        frame_id (str): target coordinate frame

    Returns:
        PointCloud2 message
    """
    assert points.ndim == 2 and points.shape[1] in [2, 3], "Points must be Nx2 or Nx3 array"

    # If only x, y provided, pad z = 0
    if points.shape[1] == 2:
        z = np.zeros((points.shape[0], 1))
        points = np.hstack([points, z])  # (N, 3)

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    if color:
        fields.append(PointField('rgb', 12, PointField.FLOAT32, 1))
        r, g, b = color
        rgb_float = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        point_data = [tuple(p) + (rgb_float,) for p in points]
    else:
        point_data = [tuple(p) for p in points]
    cloud_msg = pc2.create_cloud(header, fields, point_data)
    return cloud_msg


def make_marker_array_from_points(points, ns="", color=(1.0, 0.0, 0.0), frame_id="map"):
    marker_array = MarkerArray()
    for id, target_point in enumerate(points):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3] if len(color) > 3 else 1.0  # 불투명
        marker.lifetime = rospy.Duration(0)  # 0이면 계속 표시
        marker_array.markers.append(marker)
    return marker_array

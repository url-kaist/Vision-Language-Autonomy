import numpy as np
try:
    import rospy
except:
    import sys
    sys.path.append("/ws/external/ai_module/src/utils/debug")
    import ai_module.src.utils.debug
    import rospy
    use_ros = False
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray


def min_distance(point, points, ord=2, return_index=False):
    """
    point: (x, y) 형태의 iterable
    points: (N, 2) 형태의 iterable 또는 np.ndarray
    ord: 거리 노름 (1, 2, np.inf 등). 기본 L2(유클리드)
    return_index: True면 (min_dist, idx) 반환
    """
    p = np.asarray(point, dtype=float).reshape(1, 2)
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"points는 (N, 2) 형태여야 합니다: points={points}; P={P}")
    if P.shape[0] == 0:
        raise ValueError("points가 비어 있습니다.")

    diffs = P - p  # (N, 2)
    dists = np.linalg.norm(diffs, ord=ord, axis=1)  # (N,)
    idx = int(np.argmin(dists))
    min_dist = float(dists[idx])
    return (min_dist, idx) if return_index else min_dist


def is_equal(a, b, tol=1e-6):
    try:
        if type(a) != type(b):
            return False

        if isinstance(a, np.ndarray):
            if a.shape != b.shape:
                return False
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                return np.allclose(a, b, atol=tol, rtol=0.0, equal_nan=True)
            return np.allclose(a, b, atol=tol)

        if isinstance(a, dict):
            if a.keys() != b.keys():
                return False
            return all(is_equal(a[k], b[k], tol) for k in a.keys())

        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(is_equal(x, y, tol) for x, y in zip(a, b))

        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return abs(a - b) <= tol
    except Exception as e:
        print(f"a: {a}")
        print(f"a: {b}")
        raise RuntimeError(f"Error occurs at is_equal: {e}")

    return a == b


def pointcloud2_to_xy_array(cloud_msg):
    points = []
    for pt in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y")):
        points.append([pt[0], pt[1]])
    has_traversable_area = False if len(points) == 0 else True
    return np.array(points).reshape(-1, 2), has_traversable_area


def find_closest_point(query_points, key_points, extra_margin=0.0):
    """
    query_points: (N, 3) np.array
    key_points: (M, 3) np.array
    """
    try:
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
    except Exception as e:
        raise RuntimeError(f"Error occurs at find_closest_point: {e}")

    return closest_points


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


def _point_to_segment_dist2(P, A, B):
    AB = B - A
    AB2 = np.sum(AB**2, axis=1)
    AB2 = np.where(AB2 == 0, 1e-12, AB2)
    AP = P[:, None, :] - A[None, :, :]
    t = np.sum(AP * AB[None, :, :], axis=2) / AB2[None, :]
    t = np.clip(t, 0.0, 1.0)
    C = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    diff = P[:, None, :] - C
    dist2 = np.sum(diff**2, axis=2)
    return dist2


def filter_waypoints_by_path(waypoints_xy, path_xy, radius):
    try:
        waypoints_xy = np.asarray(waypoints_xy, dtype=np.float64) # (N, 3)
        path_xy = np.asarray(path_xy, dtype=np.float64)
        r2 = float(radius)**2

        N = len(waypoints_xy)
        if N == 0:
            return np.array([], dtype=bool), waypoints_xy

        if waypoints_xy.shape[1] == 3:
            waypoints_xy = waypoints_xy[:, :2]

        M = len(path_xy)
        if M == 0:
            kept = np.ones(N, dtype=bool)
            return kept, waypoints_xy
        if M == 1:
            dist2 = np.sum((waypoints_xy - path_xy[0])**2, axis=1)
            kept = dist2 >= r2
            return kept, waypoints_xy[kept]

        A = path_xy[:-1]
        B = path_xy[1:]
        dist2_all = _point_to_segment_dist2(waypoints_xy, A, B)
        min_dist2 = np.min(dist2_all, axis=1)
        kept = min_dist2 >= r2
    except Exception as e:
        raise RuntimeError(f"Error occurs at filter_waypoints_by_path: {e}")
    return kept, waypoints_xy[kept]


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

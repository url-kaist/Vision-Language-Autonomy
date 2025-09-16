import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import splprep, splev



def smooth_path_gradient(path, alpha=0.5, beta=0.2, iterations=50):
    path_np = np.array(path)
    smoothed_path = path_np.copy()
    
    for _ in range(iterations):
        for i in range(1, len(path_np) - 1):
            data_term = alpha * (path_np[i] - smoothed_path[i])
            smooth_term = beta * (smoothed_path[i-1] + smoothed_path[i+1] - 2 * smoothed_path[i])
            smoothed_path[i] += data_term + smooth_term
            
    return smoothed_path.tolist()

def smooth_path_spline(path, num_points=100, smoothness=1.0, degree=3):
    if len(path) < degree + 1:
        return path

    path_np = np.array(path)
    x, y = path_np.T
    
    tck, u = splprep([x, y], s=smoothness, k=degree)
    
    u_new = np.linspace(u.min(), u.max(), num_points)
    
    x_new, y_new = splev(u_new, tck)
    
    smoothed_path = np.vstack((x_new, y_new)).T
    return smoothed_path.tolist()

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
    return kept, waypoints_xy[kept]

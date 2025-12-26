import numpy as np


def theta_from_agent_pose(orientation):
    qx, qy, qz, qw = orientation
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    theta = np.arctan2(siny_cosp, cosy_cosp)
    return theta

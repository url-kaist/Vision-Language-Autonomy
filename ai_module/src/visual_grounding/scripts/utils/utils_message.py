import sys
sys.path.append('/ws/external')
import numpy as np
from ai_module.src.utils.refine_bbox import point3d_to_xyzlwh
try:
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from tf.transformations import euler_from_quaternion
except:
    pass


def get_3d_box_corners(x, y, z, l, w, h, yaw):
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

def xyzlwh2marker(x, y, z, l, w, h, id=0, color=(0.0, 0.0, 1.0, 0.8), style='cube'):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "answer"
    marker.id = id
    marker.action = Marker.ADD

    if style == 'cube':
        marker.type = Marker.CUBE
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = l
        marker.scale.y = w
        marker.scale.z = h
    elif style == 'box':
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.02

        yaw = euler_from_quaternion([0.0, 0.0, 0.0, 1.0])[2]
        corners = get_3d_box_corners(x, y, z, l, w, h, yaw)

        # Create the 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
        ]
        for i1, i2 in edges:
            p1 = Point(x=corners[i1][0], y=corners[i1][1], z=corners[i1][2])
            p2 = Point(x=corners[i2][0], y=corners[i2][1], z=corners[i2][2])
            marker.points.append(p1)
            marker.points.append(p2)
    else:
        raise NotImplementedError

    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3]) if len(color) > 3 else 0.8

    return marker


def object_to_marker(object, *args, **kwargs):
    if isinstance(object, dict):
        assert all(key in object.keys() for key in ['center', 'min_bbox', 'max_bbox'])
        center = object['center']
        min_bbox = object['min_bbox']
        max_bbox = object['max_bbox']
    else:
        center = object.center
        min_bbox = object.min_bbox
        max_bbox = object.max_bbox

    scale = np.array(max_bbox) - np.array(min_bbox)
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    l, w, h = float(scale[0]), float(scale[1]), float(scale[2])

    return xyzlwh2marker(x, y, z, l, w, h, *args, **kwargs)


def point_3d_to_marker(point_3d, *args, **kwargs):
    x, y, z, l, w, h, yaw = point3d_to_xyzlwh(point_3d)
    return xyzlwh2marker(x, y, z, l, w, h, *args, **kwargs)

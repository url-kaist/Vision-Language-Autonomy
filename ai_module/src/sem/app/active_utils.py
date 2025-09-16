import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import cascaded_union, polygonize

try:
    import rospy
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
except:
    pass


def quaternion_to_yaw(q):
    x, y, z, w = q

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw


def get_fov_marker_from_pose(pose, fov_rad, max_depth, marker_id=0, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "fov"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    position = pose["position"]  # (x, y, z)
    if "rotation" in pose.keys():
        rotation = pose["rotation"]  # 3x3
    else:
        orientation = pose["orientation"]  # 3x3
        rotation = Rotation.from_quat(orientation).as_matrix()

    heading_vector = rotation @ np.array([1.0, 0.0, 0.0])  # forward direction
    yaw = np.arctan2(heading_vector[1], heading_vector[0])

    num_points = 30
    angles = np.linspace(-fov_rad / 2, fov_rad / 2, num_points)
    points = []

    points.append(Point(x=position[0], y=position[1], z=position[2]))

    for angle in angles:
        global_angle = yaw + angle
        # np.array([position.x, position.y, position.z]),
        x = position[0] + max_depth * np.cos(global_angle)
        y = position[1] + max_depth * np.sin(global_angle)
        z = position[2]
        points.append(Point(x=x, y=y, z=z))
    points.append(Point(x=position[0], y=position[1], z=position[2]))
    marker.points = points

    return marker


def create_fov_polygon(position, yaw, fov_rad, max_depth, num_points=30):
    points = [[position.x, position.y]] # start point: the position of robot
    angles = np.linspace(-fov_rad / 2, fov_rad / 2, num_points)

    for angle in angles:
        theta = yaw + angle
        x = position.x + max_depth * np.cos(theta)
        y = position.y + max_depth * np.sin(theta)
        points.append([x, y])
    return Polygon(points)


def make_ray_marker(robot_pos, yaw, fov, num_rays=60, max_depth=5.0, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "rays"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.01
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.7

    angles = np.linspace(-fov/2, fov/2, num_rays)
    for angle in angles:
        theta = yaw + angle
        x1, y1 = robot_pos.x, robot_pos.y
        x2 = x1 + max_depth * np.cos(theta)
        y2 = y1 + max_depth * np.sin(theta)
        marker.points.append(Point(x=x1, y=y1, z=0.0))
        marker.points.append(Point(x=x2, y=y2, z=0.0))

    return marker


def get_convex_hull_polygon(points_2d): # Loose
    if points_2d.shape[0] < 3:
        # rospy.logwarn(f"[ConvexHull] Not enough points to build hull: got {points_2d.shape[0]}")
        return None, None
    if np.linalg.matrix_rank(points_2d - points_2d[0]) < 2:
        return None, None
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]
    return Polygon(hull_pts), hull_pts


def get_tight_convex_hull_polygon(points_2d, alpha=1.0):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    Args:
        points (np.ndarray): np.array of shape (n,2) points.
        alpha (float): alpha value to control tightness.
    Returns:
        shapely.geometry.Polygon: the resulting polygon
    """
    if len(points_2d) < 4:
        # No need to compute; just use convex hull
        return MultiPoint(list(points_2d)).convex_hull

    tri = Delaunay(points_2d)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points_2d[ia], points_2d[ib], points_2d[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 1e-10)**0.5
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    edge_lines = [LineString([points_2d[i], points_2d[j]]) for i, j in edges]
    m = polygonize(edge_lines)

    poly = cascaded_union(list(m))
    return poly, np.array(poly.exterior.coords)


def split_hull_segments_by_fov(hull_pts, fov_poly):
    segments = []

    for i in range(len(hull_pts)):
        pt1 = hull_pts[i]
        pt2 = hull_pts[(i + 1) % len(hull_pts)]
        line = LineString([pt1, pt2])
        if fov_poly.intersects(line):
            segments.append(('in', [pt1, pt2]))
        else:
            segments.append(('out', [pt1, pt2]))

    return segments


def split_hull_segments_by_two_polygons(hull_pts, poly1, poly2):
    """
    두 개의 polygon을 기준으로 각각 intersection 검사
    Returns:
        dict with keys:
            'map_in_fov_in': segments intersecting both poly1 and poly2
            'map_in_fov_out': segments intersecting poly1 only
            'map_out': segments intersecting neither
    """
    map_in_fov_in = []
    map_in_fov_out = []
    map_out = []

    for i in range(len(hull_pts)):
        pt1 = hull_pts[i]
        pt2 = hull_pts[(i + 1) % len(hull_pts)]
        line = LineString([pt1, pt2])

        in_poly1 = poly1.intersects(line)
        in_poly2 = poly2.intersects(line)

        if in_poly1 and in_poly2:
            map_in_fov_in.append([pt1, pt2])
        elif in_poly1 and not in_poly2:
            map_in_fov_out.append([pt1, pt2])
        else:
            map_out.append([pt1, pt2])

    return {
        'map_in_fov_in': map_in_fov_in,
        'map_in_fov_out': map_in_fov_out,
        'map_out': map_out
    }


def make_line_marker(hull_pts, color, marker_id, ns="hull_segments", frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.scale.x = 0.07
    if len(color) == 3:
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
    elif len(color) == 4:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    else:
        raise ValueError

    try:
        marker.type = Marker.LINE_LIST
        for p1, p2 in hull_pts:
            marker.points.append(Point(x=p1[0], y=p1[1], z=0.0))
            marker.points.append(Point(x=p2[0], y=p2[1], z=0.0))
    except:
        marker.type = Marker.LINE_STRIP
        marker.points = ([Point(x=pt[0], y=pt[1], z=0.0) for pt in hull_pts] +
                         [Point(x=hull_pts[0][0], y=hull_pts[0][1], z=0.0)])

    return marker

#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
from tf.transformations import quaternion_from_euler

def make_arrow_marker(
    start_xyz,
    dxdy,
    frame_id="world",
    marker_id=0,
    color=(1.0, 0.0, 0.0, 1.0),
    shaft_diameter=0.05,
    head_diameter=0.1,
    head_length=0.2,
):
    """
    start_xyz : (x, y, z)
    dxdy      : (dx, dy)
    """

    x, y, z = start_xyz
    dx, dy = dxdy

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "arrow_dxdy"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Arrow는 points[0] -> points[1]
    p_start = Point(x=x, y=y, z=z)
    p_end = Point(x=x + dx, y=y + dy, z=z)

    marker.points.append(p_start)
    marker.points.append(p_end)

    # scale.x = shaft diameter
    # scale.y = head diameter
    # scale.z = head length
    marker.scale.x = shaft_diameter
    marker.scale.y = head_diameter
    marker.scale.z = head_length

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    marker.lifetime = rospy.Duration(0)  # 계속 유지

    return marker


def main():
    rospy.init_node("dxdy_arrow_visualizer")

    pub = rospy.Publisher(
        "/dxdy_arrow_marker",
        Marker,
        queue_size=1
    )

    rate = rospy.Rate(10)

    # 예시 값
    x1, y1 = -1.2000000476837158, 3.5
    x2, y2 = -1.1797705292701721, 3.9610007405281067
    dx, dy = x2 - x1, y2 - y1
    print(f"dx, dy: {dx}, {dy}") # 0.0202295184135437, 0.4610007405281067
    theta = math.atan2(dy, dx)
    print(f"theta: {theta}") # 1.526942712448124

    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, theta)

    while not rospy.is_shutdown():
        # marker = make_arrow_marker(
        #     start_xyz=(x1, y1, 0.0),
        #     dxdy=(dx, dy),
        #     frame_id="world",
        #     marker_id=0,
        #     color=(0.0, 1.0, 0.0, 1.0)  # green
        # )
        # pub.publish(marker)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "arrow_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # ===== pose =====
        marker.pose.position.x = x1
        marker.pose.position.y = y1
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        # ===== scale =====
        marker.scale.x = 0.1      # 화살표 길이
        marker.scale.y = 0.05        # shaft diameter
        marker.scale.z = 0.1         # head diameter

        # ===== color =====
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0)

        pub.publish(marker)
        rate.sleep()


if __name__ == "__main__":
    main()

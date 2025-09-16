#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external/")

import rospy, math
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from ai_module.src.utils.logger import Logger


class RangeFilterNode:
    def __init__(self):
        rospy.init_node('range_filter', anonymous=True)
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='RangeFilterNode', log_path="/ws/external/log/exploration/range_filter.log")

        self.min_range = rospy.get_param('~min_range', 0.01) 
        self.max_range = rospy.get_param('~max_range', 5)     
        self.eps       = rospy.get_param('~eps',       0.1)
        self.sub = rospy.Subscriber('/sensor_scan',  PointCloud2, self.callback, queue_size=1)
        self.pub = rospy.Publisher( '/sensor_scan', PointCloud2, queue_size=1)
        self.logger.loginfo("Range Filter Node Initialized")
        rospy.spin()

    def callback(self, msg: PointCloud2):
        header = msg.header
        filtered = []
        field_names = [f.name for f in msg.fields]
        use_intensity = 'intensity' in field_names

        if use_intensity:
            for x, y, z, intensity in pc2.read_points(
                    msg, field_names=('x','y','z','intensity'), skip_nans=True):
                r = math.hypot(x, y, z)
                if self.min_range < r < (self.max_range - self.eps) and intensity > 0:
                    filtered.append((x, y, z))
        else:
            for x, y, z in pc2.read_points(
                    msg, field_names=('x','y','z'), skip_nans=True):
                r = math.hypot(x, y, z)
                if self.min_range < r < (self.max_range - self.eps):
                    filtered.append((x, y, z))
        new_msg = pc2.create_cloud_xyz32(header, filtered)
        new_msg.is_dense = False
        self.pub.publish(new_msg)

if __name__ == '__main__':
    try:
        RangeFilterNode()
    except rospy.ROSInterruptException:
        pass

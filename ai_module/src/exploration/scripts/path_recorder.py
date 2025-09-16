import sys
sys.path.append("/ws/external")
import threading
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from std_srvs.srv import Trigger, TriggerResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from geometry_msgs.msg import PoseStamped

from ai_module.src.utils.logger import Logger


class PathRecorder:
    def __init__(self):
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='PathRecorder', log_path="/ws/external/log/exploration/path_recorder.log")
        self.logger.log(f"Path Recorder")
        self.is_real_world = rospy.get_param('~real_world', False)
        if self.is_real_world:
            self.logger.loginfo("Hello Real World!!")
            self.frame_id = "world"
        else:
            self.frame_id = "map"

        self.path_points = []  # [[x1, y1], [x2, y2], ...]
        self.dist_thresh = 0.05  # Record only when the minimum movement distance (m) is exceeded

        self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=1)
        self.running = False
        self.is_paused = False
        self.trigger_srv = rospy.Service("/path_recorder/trigger", Trigger, self._trigger_callback)
        self.pause_srv = rospy.Service("/path_recorder/pause", SetBool, self._pause_callback)
        self.path_pub = rospy.Publisher("/path_recorder/path", Path, queue_size=1)

    def _trigger_callback(self, req):
        self.running = not self.running
        self.logger.loginfo(f"State changed to {self.running}")
        if self.running:
            self.path_points = []
            self.logger.loginfo(f"Path points reset")
        return TriggerResponse(success=self.running, message="current state")

    def _pause_callback(self, req: SetBoolRequest) -> SetBoolResponse:
        if not self.running:
            return SetBoolResponse(success=False, message="Not recording. Start first.")
        if req.data:  # pause
            if self.is_paused:
                return SetBoolResponse(success=True, message="Already paused.")
            self.is_paused = True
            return SetBoolResponse(success=True, message="Paused.")
        else:  # resume
            if not self.is_paused:
                return SetBoolResponse(success=True, message="Already running")
            self.is_paused = False
            return SetBoolResponse(success=True, message="Resumed.")

    def _odom_callback(self, msg):
        if self.running and not self.is_paused:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

            if not self.path_points:
                self.path_points.append([x, y])
                return

            last_x, last_y = self.path_points[-1]
            dist = np.hypot(x - last_x, y - last_y)
            if dist >= self.dist_thresh:
                self.path_points.append([x, y])

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.frame_id

        for (x, y) in self.path_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)


if __name__ == "__main__":
    rospy.init_node("path_recorder")
    recorder = PathRecorder()
    rospy.spin()

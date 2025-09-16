#!/usr/bin/env python3
import concurrent.futures
import os
import time
import threading
import copy
import queue

from base_visual_grounder import BaseActiveVisualGrounder, Status
from ai_module.src.utils.logger import LoggerConfig, LOG_DIR
from ai_module.src.utils.utils import (pointcloud2_to_xy_array, filter_close_points, find_closest_point, \
                                       filter_waypoints_by_path, make_marker_array_from_points)
from ai_module.src.visual_grounding.scripts.structures.inference_result import InferenceResult

from ai_module.src.visual_grounding.scripts.utils.utils_message import object_to_marker

try:
    import rospy
    from std_msgs.msg import String, Int32
    from visualization_msgs.msg import Marker
    use_rospy = True
except:
    use_rospy = False


ANSWER_TYPE = {'find': Int32, 'count': Marker}


class ActiveGrounder(BaseActiveVisualGrounder):
    def __init__(self, logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if logger is None:
            logger = self.logger
        self._init_all(logger=logger, *args, **kwargs)


if __name__ == "__main__":
    rospy.init_node('visual_grounding')
    node_name = rospy.get_name()
    node_name = node_name.strip('/')
    logger_cfg = LoggerConfig(
        quiet=False, prefix=f"ActiveGrounder{node_name.split('_')[-1]}",
        log_path=os.path.join(LOG_DIR, f'{node_name}.log'),
        no_intro=False
    )
    ag = ActiveGrounder(node_name=node_name, logger_cfg=logger_cfg)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    main_thread = threading.Thread(target=ag.main_loop, daemon=True)
    main_thread.start()

    executor.submit(ag.inference_loop, 2.0)
    # executor.submit(ag.validation_loop, 2.0)
    executor.submit(ag.navigation_loop, 5.0)

    rospy.spin()
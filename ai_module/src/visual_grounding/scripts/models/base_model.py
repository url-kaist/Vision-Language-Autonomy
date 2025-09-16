import os
import sys
from typing import Optional
from ai_module.src.utils.logger import LoggerConfig, build_logger, LOG_DIR, StreamToLogger
try:
    import rospy
except:
    sys.path.append("/ws/external/ai_module/src/utils/debug")
    import ai_module.src.utils.debug
    import rospy


class BaseModel:
    def __init__(
            self,
            *,
            logger=None,
            logger_cfg: Optional[LoggerConfig]=None,
            display_name: str=None,
            **kwargs,
    ):
        self.logger = build_logger(logger=logger, logger_cfg=logger_cfg)
        if self.logger.__class__.__name__ != 'Logger':
            sys.stdout = StreamToLogger(self.logger, level="INFO")
            sys.stderr = StreamToLogger(self.logger, level="ERROR")
        self.logger.loginfo(f"Initialized {display_name or self.__class__.__name__}")
        self.debug = rospy.get_param('~debug', False)


if __name__ == "__main__":
    logger_cfg = LoggerConfig(
        quiet=False, prefix='Test for BaseModel',
        log_path=os.path.join(LOG_DIR, 'test.log'),
        no_intro=False
    )

    base_model = BaseModel(logger_cfg=logger_cfg)

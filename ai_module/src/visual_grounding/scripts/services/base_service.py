import os
import sys
sys.path.append('/ws/external')
from typing import Type

from ai_module.src.visual_grounding.scripts.models.base_model import BaseModel

try:
    import rospy
    from std_srvs.srv import Empty, EmptyResponse
    use_ros = True
except:
    use_ros = False


class BaseServiceClient(BaseModel):
    def __init__(
            self,
            service_name: str,
            service_type: Type,
            timeout: float = 5.0,
            **kwargs,
    ):
        super().__init__(display_name=f"{self.__class__.__name__}({service_name})", **kwargs)

        self.service_name = service_name
        self.service_type = service_name
        self.timeout = timeout

        self._proxy = rospy.ServiceProxy(service_name, service_type)
        self.use_ros = use_ros

    def __call__(self, *args, **kwargs):
        try:
            return self._proxy(*args, **kwargs)
        except rospy.ServiceException as e:
            self.logger.logerr(f"Service call to {self.service_name} failed: {e}")
            return None


if __name__ == "__main__":
    client = BaseServiceClient("/test", Empty)
    rospy.wait_for_service(client.service_name)
    response = client()
    print(response)

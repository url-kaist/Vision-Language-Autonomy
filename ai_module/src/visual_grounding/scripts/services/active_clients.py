import os
import sys
sys.path.append("/ws/external")
import time
import threading

from ai_module.src.utils.logger import LoggerConfig, LOG_DIR
from ai_module.src.sem.app.load_shm import load_graph_from_shared_memory, load_object_data_from_shared_memory
from ai_module.src.visual_grounding.scripts.services.base_service import BaseServiceClient
from ai_module.src.visual_grounding.scripts.models.base_model import BaseModel

try:
    import rospy
except:
    sys.path.append("/ws/external/ai_module/src/utils/debug")
    import ai_module.src.utils.debug
    import rospy
from std_srvs.srv import Empty, EmptyResponse, Trigger, TriggerResponse, SetBool, SetBoolResponse


class ActiveClients(BaseModel):
    name = "active_grounder"

    def __init__(self, use_ros=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data
        self.is_running = False
        self.is_paused = False

        # Clients
        # scene_graph_trigger = BaseServiceClient("/scene_graph/active_signal", Trigger, logger=self.logger)
        self.trigger_clients = {
            'path_recorder_trigger': BaseServiceClient("/path_recorder/trigger", Trigger, logger=self.logger),
            # 'scene_graph_trigger': scene_graph_trigger,
            # 'path_follower_trigger': BaseServiceClient("/path_follower/active_signal", Trigger, logger=self.logger),
        }
        self.pause_clients = {
            'path_recorder_pause': BaseServiceClient("/path_recorder/pause", SetBool, logger=self.logger),
            # 'path_follower_trigger': BaseServiceClient("/path_follower/active_signal", Trigger, logger=self.logger),
        }
        self.use_ros = use_ros

        self.clients_ready = {
            'path_recorder_trigger': False,
            # 'scene_graph_trigger': False,
            # 'path_follower_trigger': False,
        }

        for func_name, client in self.trigger_clients.items():
            self.logger.logrich(f"Wait: {client.service_name}", name=client.service_name)
            rospy.wait_for_service(client.service_name)
            self.logger.logrich(f"Ready: {client.service_name}", name=client.service_name)
            self.logger.stop(client.service_name)
        for func_name, client in self.pause_clients.items():
            self.logger.logrich(f"Wait: {client.service_name}", name=client.service_name)
            rospy.wait_for_service(client.service_name)
            self.logger.logrich(f"Ready: {client.service_name}", name=client.service_name)
            self.logger.stop(client.service_name)

    def start(self):
        for client_name, ready in self.clients_ready.items():
            if not ready:
                res = self.trigger_clients[client_name]()
                self.clients_ready[client_name] = res.success
        success = all(self.clients_ready.values())
        if success:
            self.is_running = True
        return success

    def pause(self):
        is_paused = True
        for client_name, client in self.pause_clients.items():
            success = client(True)
            if not success:
                is_paused = False
        self.is_paused = is_paused
        return is_paused

    def resume(self):
        is_paused = False
        for client_name, client in self.pause_clients.items():
            success = client(False)
            if not success:
                is_paused = True
        self.is_paused = is_paused
        return is_paused

    def end(self):
        for client_name, ready in self.clients_ready.items():
            if ready:
                res = self.trigger_clients[client_name]()
                self.clients_ready[client_name] = res.success
        success = not any(self.clients_ready.values())
        if success:
            self.is_running = False
        return success


if __name__ == "__main__":
    logger_cfg = LoggerConfig(
        quiet=False, prefix='Test for BaseModel',
        log_path=os.path.join(LOG_DIR, 'test.log'),
        no_intro=False
    )
    active_clients = ActiveClients(logger_cfg=logger_cfg, use_ros=False)

    DATA_DIR = "/ws/external/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)

    for dir in dir_sorted:
        active_clients.update_scene_graph(dir=dir)
        time.sleep(0.1)

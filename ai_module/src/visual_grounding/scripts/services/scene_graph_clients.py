import os
import sys
sys.path.append("/ws/external")
import time
import json
from datetime import datetime
from typing import Optional, Any

from ai_module.src.utils.logger import Logger, LOG_DIR
from ai_module.src.sem.app.load_shm import load_graph_from_shared_memory, load_object_data_from_shared_memory
from ai_module.src.visual_grounding.scripts.services.base_service import BaseServiceClient
from ai_module.src.visual_grounding.scripts.models.base_model import BaseModel
from ai_module.src.visual_grounding.scripts.structures.scene_graph import SceneGraph

try:
    import rospy
    from std_srvs.srv import Empty, EmptyResponse
    use_ros = True
except:
    Empty = None
    use_ros = False


class SceneGraphClients(BaseModel):
    name = "scene_graph"

    def __init__(self, use_ros=True, *args, **kwargs):
        super().__init__(**kwargs)

        # Data
        self.sg = None

        # Clients
        if use_ros:
            self.clients = {
                'save_data': BaseServiceClient("/save_data", Empty, logger=self.logger)
            }
        else:
            self.clients = {}
        self.use_ros = use_ros

        # get_scene_graph
        self.scene_graph_shm_name = "scene_graph_shm"
        self.scene_graph_obj_shm_name = "object_shm"
        self.offline_map_dir = "/ws/external/offline_map"

        if use_ros:
            for func_name, client in self.clients.items():
                self.logger.logrich(f"Wait: {client.service_name}", name=client.service_name)
                rospy.wait_for_service(client.service_name)
                self.logger.logrich(f"Ready: {client.service_name}", name=client.service_name)
                self.logger.stop(client.service_name)

    def start(self, candidate_names, reference_names, *args, **kwargs):
        self.sg = SceneGraph(
            candidate_names=candidate_names, reference_names=reference_names,
            *args, **kwargs
        )

    @staticmethod
    def get_latest_dirname(dir):
        def is_timestamp_format(name):
            try:
                datetime.strptime(name, "%Y%m%d_%H%M%S")
                return True
            except ValueError:
                return False

        timestamp_dirs = [
            d for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d)) and is_timestamp_format(d)
        ]
        latest_dirname = max(timestamp_dirs) if timestamp_dirs else None
        return os.path.join(dir, latest_dirname) if latest_dirname else None

    def _get_scene_graph(self):
        current_scene_graph, current_objects = None, None

        current_scene_graph_res = load_graph_from_shared_memory(self.scene_graph_shm_name)
        current_objects_res = load_object_data_from_shared_memory(self.scene_graph_obj_shm_name)

        # save_dir = self.get_latest_dirname(self.offline_map_dir)
        # if save_dir is not None:
        #     pcd_path = os.path.join(save_dir, 'hash_pointcloud.pcd')
        #     if not os.path.exists(pcd_path):
        #         self.logger.logwarn(f"{pcd_path} does not exist")
        #         return None, None
        #     pcd = o3d.io.read_point_cloud(pcd_path)
        #     if len(pcd.points) == 0:
        #         self.logger.logwarn(f"{pcd_path} is empty point cloud")
        #         return None, None
        #     # points = np.asarray(pcd.points)

        if current_scene_graph_res is not None:
            scene_graph, scene_graph_shm = current_scene_graph_res
            try:
                scene_graph_shm.close()
                current_scene_graph = scene_graph
            except Exception as e:
                self.logger.logwarn(f"Exception occurs when closing the shared memory of the scene graph: {e}")
            try:
                scene_graph_shm.unlink()
            except FileNotFoundError:
                # Shared memory already cleaned up or doesn't exist
                pass
            except Exception as e:
                self.logger.logwarn(f"Exception occurs when unlinking the shared memory of the scene graph: {e}")

        if current_objects_res is not None:
            objects, objects_shm = current_objects_res
            try:
                objects_shm.close()
                current_objects = objects
            except Exception as e:
                self.logger.logwarn(f"Exception occurs when closing the shared memory of the objects: {e}")
            try:
                objects_shm.unlink()
            except FileNotFoundError:
                # Shared memory already cleaned up or doesn't exist
                pass
            except Exception as e:
                self.logger.logwarn(f"Exception occurs when unlinking the shared memory of the objects: {e}")

        return current_scene_graph, current_objects

    def update_scene_graph(self, *args, **kwargs) -> Optional[Any]:
        if self.use_ros:
            # ROS version
            try:
                res = self.clients['save_data']()
           
                scene_graph, objects = self._get_scene_graph()
                if (scene_graph is None) or (objects is None):
                    self.logger.logwarn(f"graph or objects is None")
                    self.logger.logwarn(f"  > graph: {scene_graph}")
                    self.logger.logwarn(f"  > objects: {objects}")
                    return
                self.sg.update(scene_graph, objects)
                self.logger.logrich(f"Scene Graph: {self.sg}", name="scene_graph")
            except rospy.ServiceException as e:
                self.logger.logerr(f"Failed to get and update scene graph: {e}")
        else:
            dir = kwargs.get('dir')
            with open(os.path.join(dir, 'scene_graph.json'), 'r', encoding='utf-8') as f:
                scene_graph = json.load(f)
            with open(os.path.join(dir, 'objects.json'), 'r', encoding='utf-8') as f:
                objects = json.load(f)
            if dir:
                dir = os.path.join(dir.split("/offline_map")[0], "keyframes")
            self.sg.update(scene_graph, objects, dir=dir)
            self.logger.logrich(f"Scene Graph: {self.sg}", name="scene_graph")


if __name__ == "__main__":
    from ai_module.src.visual_grounding.scripts.debug.debug_utils import *
    
    logger = Logger(
        quiet=False, prefix='Test for BaseModel',
        log_path=os.path.join("/ws/external/test_data/log", 'test.log'),
        no_intro=False
    )
    
    occupancy_grid = load_occupancy_grid_data('/ws/external/test_data/occupancy_grid.json')
    
    scene_graph_clients = SceneGraphClients(logger=logger, use_ros=False)
    scene_graph_clients.start(candidate_names=['path', 'sofa', 'coffee table'], reference_names=[])

    DATA_DIR = "/ws/external/test_data/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)

    for dir in dir_sorted:
        scene_graph_clients.update_scene_graph(dir=dir)

        # keyframes = scene_graph_clients.sg.keyframes
        keyframes = scene_graph_clients.sg.history_keyframes
        
        for kf_id, kf in keyframes.items():
            style = {
                'image': {
                    'show': True,
                    'text': {
                        'color': 'green',
                        'size': 1,
                        'thickness': 2,
                        'distance_colors': {
                            'enabled': True
                        }
                    },
                    'object_annotation': {
                        'candidate': {
                            'show': True,
                            'color': 'orange'
                        },
                        'dtype': 'all'
                    }
                },
                'occupancy_grid': {
                    'show': True,
                    'grid': {
                        'occupied_color': 'black',
                        'free_color': 'white',
                        'unknown_color': 'gray',
                        'width': 100,
                        'height': 100
                    },
                    'current_point': {
                        'color': 'red',
                        'radius': 5,
                        'thickness': -1
                    },
                    'text': {
                        'color': 'green',
                        'size': 1,
                        'thickness': 2,
                        'distance_colors': {
                            'enabled': True
                        }
                    },
                    'object': {
                        'color': 'blue',
                        'thickness': 0,
                        'group_colors': {
                            'candidate': 'yellow',
                            'reference': 'orange'
                        }
                    }
                }
            }

            kf.get_movable_points(occupancy_grid, distances=[1.2, 2, 3.5], step_angle=40, style=style, node_name="test", suffix="_movable_points")

            styles = {
                # 'reference': {'show': True, 'color': 'red'},
                'candidate': {'show': True, 'color': 'green'},
            }
            kf.annotate(styles, node_name="test", suffix="_annotated_global")

        time.sleep(0.1)

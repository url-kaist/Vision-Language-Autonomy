#!/usr/bin/env python3
import concurrent.futures
import os
import sys
import time
import threading
import copy
import numpy as np
from collections import deque
import math
import cv2
from cv_bridge import CvBridge

from base_visual_grounder import BaseActiveVisualGrounder, Status
from ai_module.src.utils.logger import LoggerConfig, LOG_DIR
from ai_module.src.utils.utils import (make_marker_array_from_points, find_closest_point, filter_close_points, filter_waypoints_by_path)
from ai_module.src.visual_grounding.scripts.structures.inference_result import InferenceResult2, InferenceResult2PriorityQueue
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframes, CurrentKeyframe
from ai_module.src.visual_grounding.scripts.vlms.prompt import PROMPT
from ai_module.src.visual_grounding.scripts.vlms.system_instruction import SYSTEM_INSTRUCTION
from ai_module.src.visual_grounding.scripts.vlms.utils.helpers import parse_json

from ai_module.src.visual_grounding.scripts.vlms.query_manager import QueryManager, QueryStatus
from ai_module.src.visual_grounding.scripts.vlms.plugin_manager import PluginManager, InferencePlugin, ValidationPlugin, PathGenerationPlugin, PathEvaluationPlugin

from visualization_msgs.msg import Marker
from ai_module.src.visual_grounding.scripts.models.base_visual_grounder import ANSWER_TYPE
from ai_module.src.visual_grounding.scripts.structures.entity import Entities
from ai_module.src.visual_grounding.scripts.structures.aggregated_result import AggregatedResult
try:
    import rospy
    from std_msgs.msg import String
    from traversable_authority.srv import (BlockBBox, BlockSegment)
    from sensor_msgs.msg import Image, CompressedImage
    from std_srvs.srv import Trigger, TriggerResponse
    use_rospy = True
except:
    use_rospy = False

ANSWER_TOPIC_NAME = {'find': 'selected_object_marker', 'count': '/numerical_response'}


class VisualFollower(BaseActiveVisualGrounder):
    def __init__(self, logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if logger is None:
            logger = self.logger
        self._init_all(logger=logger, *args, **kwargs)
        
        self.current_image_dir = "/ws/external/keyframes"
        self.bridge = CvBridge()

        """ Thread Management """
        self.executor = None
        self.main_thread = None
        self.stop_event = threading.Event()
        
        """ Query Manager """
        self.query_manager = QueryManager(max_workers=3, config_file=os.path.join(os.path.dirname(__file__), 'vlms/query_config.json'))
        self.plugin_manager = PluginManager()
        self._register_default_plugins()
        self._register_plugins_to_query_manager()
        self.query_manager.start()
        
        """ Visualization """
        self.path_annotation_style = {
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
                    # 'object_annotation': {
                    #     'candidate': {
                    #         'show': True,
                    #         'color': 'orange',
                    #         'alpha': 1,
                    #         'draw_id': False
                    #     },
                    #     'etype': 'all'
                    # }
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
                    'path_history': {
                        'color': 'yellow',
                        'alpha': 0.5,
                        'alpha_decay': 0.02,
                    },
                    # 'object': {
                    #     'color': 'blue',
                    #     'thickness': -1,
                    #     'group_colors': {
                    #         'candidate': 'orange',
                    #         'reference': 'orange'
                    #     }
                    # }
                }
            }
    
    def _register_default_plugins(self):
        self.plugin_manager.register_plugin('inference_query', InferencePlugin())
        self.plugin_manager.register_plugin("validation_query", ValidationPlugin())
        self.plugin_manager.register_plugin('path_generation_query', PathGenerationPlugin())
        self.plugin_manager.register_plugin('path_evaluation_query', PathEvaluationPlugin())
    
    def _register_plugins_to_query_manager(self):
        self.logger.loginfo("Registering plugins to query manager...")
        for query_type, plugin in self.plugin_manager.plugins.items():
            self.logger.loginfo(f"Registering plugin: {query_type} -> {plugin.name}")
            plugin.set_visual_follower(self)
            self.query_manager.register_worker(query_type, plugin)
        self.logger.loginfo(f"Registered {len(self.plugin_manager.plugins)} plugins to query manager")
    
    def _init_vars(self, *args, **kwargs) -> None:
        super()._init_vars(*args, **kwargs)
        # Extract node number from node_name (e.g., "visual_grounding_0" -> "0")
        self.node_number = self.node_name.split('_')[-1] if '_' in self.node_name else "0"

    def _init_services(self, *args, **kwargs) -> None:
        super()._init_services(*args, **kwargs)
        self.trav_bbox_cli = rospy.ServiceProxy("/traversable/apply_bbox", BlockBBox)
        self.trav_segment_cli = rospy.ServiceProxy("/traversable/apply_segment", BlockSegment)
        # Remove the active signal service client - we'll use subscriber instead
        # self.srv_node_active_signal_client = rospy.ServiceProxy(self.node_name + "/active_signal", Trigger)

    def _init_publishers(self, *args, **kwargs):
        super()._init_publishers(*args, **kwargs)
        self.find_pub = rospy.Publisher('/selected_object_marker', Marker, queue_size=10)
        
        self.latest_exploration_strategy = None

    def _init_subscribers(self, *args, **kwargs):
        super()._init_subscribers(*args, **kwargs)
        
        if self.is_real_world:
            self.is_compressed_image = True
            self.rgb_sub = rospy.Subscriber("/camera/image", CompressedImage, self._rgb_callback, queue_size=1)
        else:
            self.is_compressed_image = False
            self.rgb_sub = rospy.Subscriber("/camera/image", Image, self._rgb_callback, queue_size=1)
        
    def _init_threads(self, *args, **kwargs) -> None:
        super()._init_threads(*args, **kwargs)
        
        self.inference_queue = deque(maxlen=1)
        self.inference_results = InferenceResult2PriorityQueue(10)
        self.excluded_objects = set() # Entities
        self.path_generation_queue = deque(maxlen=1)
        self.path_evaluation_queue = deque(maxlen=1)
        self.arrived_event = threading.Event()
        self.arrived_time = None
        self.semantic_frontier_pub_duration = 2
        self.inference_ready_event = threading.Event()
        self.inference_result_lock = threading.RLock()
        self.mission_completed_event = threading.Event()
        self.inference_processing_event = threading.Event()
        self.path_evaluation_processing_event = threading.Event()
        self.agent_pose_threshold = 0.5
        self.validation_count_threshold = 3
        
        
        self.path_generation_result_lock = threading.RLock()
        self.path_generation_ready_event = threading.Event()
        self.generated_path_points = None
        
        # Track previous candidate objects to detect changes
        self.previous_candidate_ids = set()
        self.previous_reference_ids = set()
        
        # Timing controls for periodic queue operations (2 seconds interval)
        self.path_generation_interval = 6.0  # seconds
        self.path_evaluation_interval = 10.0
        self.last_path_generation_time = 0.0
                        
        self.query_count_lock = threading.RLock()
        self.query_count = {
            'inference':{
                'request': 0,
                'response': 0
            },
            'validation':{
                'request': 0,
                'response': 0
            },
            'path_generation':{
                'request': 0,
                'response': 0
            },
            'path_evaluation':{
                'request': 0,
                'response': 0
            }
        }
                
    def _reset_vars(self):        
        # clear queries in query manager
        for query_id in list(self.query_manager.active_queries.keys()):
            self.query_manager.cancel_query(query_id)
        self.query_manager.clear_completed_responses()
        
        self.query_count = {
            'inference':{
                'request': 0,
                'response': 0
            },
            'validation':{
                'request': 0,
                'response': 0
            },
            'path_generation':{
                'request': 0,
                'response': 0
            },
            'path_evaluation':{
                'request': 0,
                'response': 0
            }
        }
        
        self._init_vars()

        self.latest_exploration_strategy = None

        # Main variables
        self.main_running = threading.Event()
        self.validation_running = threading.Event()
        self.answer = ""
        self.answer_result = None
        self.ready = False
        self.subtask = None
        
        # Navigation variables
        self.path_points = None
        self.navigation_running = threading.Event()
        self.active_clients.end()
        
        # Inference variables
        self.inference_results.clear()
        self.excluded_objects = set()
        
        # Clear all events to prevent race conditions with previous tasks
        self.inference_queue = deque(maxlen=10)
        self.agg_results = AggregatedResult(**self.aggregated_results_cfg)
        self.inference_ready_event.clear()
        self.inference_processing_event.clear()
        self.path_evaluation_processing_event.clear()
        
        self.path_generation_queue = deque(maxlen=1)
        self.path_evaluation_queue = deque(maxlen=1)
        self.arrived_event = threading.Event()
        self.arrived_time = None
        self.mission_completed_event = threading.Event()
        
        self.path_generation_ready_event = threading.Event()
        self.generated_path_points = None

        # Reset candidate tracking
        self.previous_candidate_ids = set()
        self.previous_reference_ids = set()
        
        # Reset timing controls
        self.last_path_generation_time = 0.0
        
        self.processing_start_time = 0.0
        
        self.default_inference_options['prompt']['action'] = 'inference_find'
        self.default_inference_options['image']['suffix'] = '_annotated_global_object'
        self.default_validate_options['prompt']['action'] = 'validate_find'
        self.default_validate_options['image']['suffix'] = '_annotated_inference'
        self.logger.loginfo(f"======================== Reset completed: {self.node_name}")
    
    def _reset_callback(self, req):
        # Stop all running threads first
        if hasattr(self, 'stop_threads'):
            self.logger.loginfo("================================================")
            self.logger.loginfo("<_reset_callback.1> Stopping all threads before reset...")
            self.stop_threads()
        
        # Reset variables
        self._reset_vars()
        
        # Restart threads
        if hasattr(self, 'start_threads'):
            self.logger.loginfo("<_reset_callback.2> Restarting all threads after reset...")
            self.start_threads()
        
        self.logger.logrich(f"Instruction: ", name='instruction')
        self.logger.logrich(f"Action: ", name='action')
        self.logger.logrich(f"Target Name: ", name='target_name')
        self.logger.logrich(f"Inference: ", name='inference')
        self.logger.log("<_reset_callback.3> Visual grounding node has been reset.")
        
        return TriggerResponse(success=True, message="Visual grounding node has been reset.")
    
    def _rgb_callback(self, msg):
        try:
            if self.is_compressed_image:
                np_arr = np.frombuffer(msg.data, np.uint8)
                rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
            # Save current image to specified path
            image_path = os.path.join(self.current_image_dir, "place_-1.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, rgb)
            
            # Check if agent_pose is available, if not use default pose
            if self.agent_pose is None:
               
                pose = None
            else:
                # Convert agent_pose dictionary to 4x4 transformation matrix
                position = self.agent_pose.get("position", np.array([0.0, 0.0, 0.0]))
                orientation = self.agent_pose.get("orientation", np.array([0.0, 0.0, 0.0, 1.0]))
                
                # Convert quaternion to rotation matrix using numpy
                rotation_matrix = self._quat_to_rotation_matrix(orientation)
                
                # Create 4x4 transformation matrix
                pose = np.eye(4)
                pose[:3, :3] = rotation_matrix
                pose[:3, 3] = position
                
            self.current_image = CurrentKeyframe(
                image_path=image_path,
                image=rgb,
                pose=pose,
                pose_dict=self.agent_pose
            )
        except Exception as e:
            self.logger.logerr(f"Error in _rgb_callback: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
    
    def pub_exploration_strategy(self, strategy):
        self.exploration_strategy_pub.publish(String(strategy))
        self.latest_exploration_strategy = strategy
        self.logger.loginfo(f"Published exploration strategy: {strategy}")
    
    def standby(self, **kwargs):
        try:
            self.logger.loginfo(f"<main.1> Starting scene graph clients...")
            self.scene_graph_clients.start(
                candidate_names=self.candidate_names,
                reference_names=self.reference_names,
                **kwargs
            )
            self.logger.loginfo(f"<main.1> Scene graph clients started.")
            
            action = self.action
            self.logger.loginfo(f"<main.1> Action: {action}")
            
            self.answer_pub = rospy.Publisher(ANSWER_TOPIC_NAME.get(action, '/answer'), ANSWER_TYPE.get(action, String), queue_size=1)
            
            """ LLM Client """
            etypes = self.etypes
            self.default_inference_options['prompt'].update({'action': 'inference_find', 'rtype': 'inference'})
            self.default_inference_options['image']['suffix'] = [f'_annotated_global_{etype}' for etype in etypes]
            self.default_validate_options['prompt'].update({'action': 'validate_find', 'rtype': 'validate'})
            self.default_validate_options['image']['suffix'] = [f'_annotated_inference_{etype}' for etype in etypes]

            self.default_path_generation_options = copy.deepcopy(self.default_options)
            self.default_path_generation_options['prompt']['action'] = 'path_generation'
            self.default_path_generation_options['image']['suffix'] = ['_movable_points_image', '_movable_points_occupancy_grid']
            
            self.default_path_evaluation_options = copy.deepcopy(self.default_options)
            self.default_path_evaluation_options['prompt']['action'] = 'path_evaluation'
            # self.default_path_evaluation_options['image']['suffix'] = ['_total_path_history', '']
            self.default_path_evaluation_options['image']['suffix'] = ['_total_path_history_image', '_total_path_history_occupancy_grid', '']
            
            if "path between" in self.target_name: # TODO: more general instruction
                self.logger.loginfo(f"inference_follow_between")
                self.default_inference_options['prompt']['action'] = 'inference_follow_between'
                self.default_validate_options['prompt']['action'] = 'validate_follow_between'
            elif "path near" in self.target_name:
                self.logger.loginfo(f"path near detected")
                original_target_name = self.subtask.entity.target_name
                self.subtask.entity.target_name = original_target_name.replace("the path near ", "").replace("path near ", "")
                self.default_inference_options['prompt']['target_name'] = self.target_name
                self.default_validate_options['prompt']['target_name'] = self.target_name

            self.ready = True
            self.logger.loginfo(f"<main.1> Standby completed.")
        except Exception as e:
            self.logger.logerr(f"<main.1> Error in standby: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")

    def inference_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            if self.status != Status.PROCESSING:
                self.logger.loginfo(f"<inference_loop.1> Status is not processing. Let's sleep..")
                rate.sleep()
                continue

            if self.inference_results.empty():
                self.logger.loginfo(f"<inference_loop.1> Inference results is empty.")
            else:
                self.logger.loginfo(f"Size of inference_results: {self.inference_results.size()}")
            try:
                if self.mission_completed_event.is_set():
                    # Answer the question
                    self.logger.loginfo(f"Mission completed. Answer the final result.")
                    self.inference_ready_event.clear()
                    with self.inference_result_lock:
                        self.answer_result = InferenceResult2()
                        self.inference_results.clear()
                    self.mission_completed_event.clear()
                    rate.sleep()
                    continue
            except Exception as e:
                self.logger.logerr(f"<inference_loop.1> Error: {e}")
                import traceback
                self.logger.logerr(f"Traceback: {traceback.format_exc()}")

            try:
                if self.inference_ready_event.is_set():
                    self.logger.loginfo(f"Inference ready event is set. Check inference result.")
                    if self.inference_results.empty():
                        self.logger.loginfo(f"Inference results is empty. Skip answer.")
                        self.inference_ready_event.clear()
                        rate.sleep()
                        continue
                    
                    with self.inference_result_lock:
                        inference_result = self.inference_results.peek()
                        inference_result = copy.deepcopy(inference_result)
                    self.logger.logrich(f"Inference: {str(inference_result)}", name='inference')
                    self.logger.loginfo(f"Inference result: {inference_result}")
                    
                    try:
                        if self.path_generation_ready_event.is_set():
                            self.path_generation_ready_event.clear()
                            with self.path_generation_result_lock:
                                self.generated_path_points = None
                            self.logger.loginfo(f"Path generation ready event cleared.")
                            
                        if inference_result.confidence < self.confidence_threshold[0]:
                            self.logger.loginfo(f"Confidence {inference_result.confidence} < {self.confidence_threshold[0]}. New inference is needed.")
                            self.path_points = []
                            self.excluded_objects.update(inference_result.objects)
                            with self.query_count_lock:
                                self.query_count['path_generation']['request'] = 0
                                self.query_count['path_generation']['response'] = 0
                            
                            with self.inference_result_lock:
                                self.inference_results.pop()
                                if self.inference_results.empty():
                                    self.inference_ready_event.clear()                        
                        else:
                            if self.subtask.action == "avoid":
                                if inference_result.has_candidate and inference_result.confidence >= self.confidence_threshold[1]:
                                    self.logger.loginfo(f"Blocking the traversable area around objects: {inference_result.objects}")
                                    
                                    with self._traversable_lock:
                                        traversable_points = self.traversable_points
                                        self.logger.loginfo(f"traversable_points type: {type(traversable_points)}, length: {len(traversable_points) if traversable_points is not None else 'None'}")
                                    
                                    try:
                                        self.block_objects(traversable_points, inference_result.keyframes, inference_result.objects)
                                        self.logger.loginfo("block_objects completed successfully")
                                    except Exception as e:
                                        self.logger.logerr(f"Error in block_objects: {e}")
                                        import traceback
                                        self.logger.logerr(f"Traceback: {traceback.format_exc()}")

                                    # Answer the question
                                    self.logger.loginfo(f"Answer the final result.")
                                    self.inference_ready_event.clear()
                                    with self.inference_result_lock:
                                        self.answer_result = inference_result
                                        self.inference_results.clear()

                            else:
                                if self.arrived_event.is_set():
                                    self.logger.loginfo(f"Arrived event is set. Check inference result.")
                                    self.logger.loginfo(f"Inference result: {inference_result}")
                                    if inference_result.has_candidate:
                                        if inference_result.confidence >= self.confidence_threshold[1]:
                                            self.logger.loginfo(f"Arrived event is set. Answer the final result.")

                                            # Answer the question
                                            self.logger.loginfo(f"Answer the final result.")
                                            self.inference_ready_event.clear()
                                            with self.inference_result_lock:
                                                self.answer_result = inference_result
                                                self.inference_results.clear()
                                        elif inference_result.validation_count >= self.validation_count_threshold:
                                            self.logger.loginfo(f"Validation count {inference_result.validation_count} >= {self.validation_count_threshold}. Remove inference result.")
                                            self.path_points = None
                                            self.arrived_event.clear()
                                            self.excluded_objects.update(inference_result.objects)
                                            with self.query_count_lock:
                                                self.query_count['path_generation']['request'] = 0
                                                self.query_count['path_generation']['response'] = 0
                                            
                                            with self.inference_result_lock:
                                                self.inference_results.pop()
                                                if self.inference_results.empty():
                                                    self.inference_ready_event.clear()   
                                    else:
                                        self.logger.loginfo(f"Arrived event is set. No candidate found. Skip answer.")
                                        self.excluded_objects.update(inference_result.objects)
                            
                                        with self.inference_result_lock:
                                            self.inference_results.pop()
                                            if self.inference_results.empty():
                                                self.inference_ready_event.clear()   
                                                
                                        with self.query_count_lock:
                                            self.query_count['path_generation']['request'] = 0
                                            self.query_count['path_generation']['response'] = 0
                                    # self.arrived_event.clear()
                                else:
                                    self.logger.loginfo(f"Arrived event is not set. Skip answer.")
                                    if len(self.inference_queue) > 0:
                                        keyframes = self.inference_queue.popleft()
                                        self.inference(keyframes)
                    except Exception as e:
                        self.logger.logerror(f"Error in inference loop: {e}")
                    rate.sleep()
                    continue       
                else:
                    self.logger.logrich(f"Inference: None", name='inference')
                    self.logger.loginfo(f"Inference: None")
                    
                    if self.arrived_event.is_set():
                        self.logger.loginfo(f"Arrived event is set. Add path evaluation queue.")
                        self.add_path_evaluation_queue()
                        
                        self.arrived_event.clear()
                        self.path_generation_ready_event.clear()
                        with self.path_generation_result_lock:
                            self.generated_path_points = None
                        self.logger.loginfo(f"Path generation ready event cleared.")
                        
                        with self.query_count_lock:
                            self.query_count['path_generation']['request'] = 0
                            self.query_count['path_generation']['response'] = 0
            except Exception as e:
                self.logger.logerr(f"<inference_loop.2> Error: {e}")
                import traceback
                self.logger.logerr(f"Traceback: {traceback.format_exc()}")

            try:
                # Query to VLMS
                if len(self.inference_queue) > 0:
                    self.logger.loginfo(f"Inference queue: {len(self.inference_queue)}")
                    keyframes = self.inference_queue.popleft()
                    self.inference(keyframes)
                else:
                    if len(self.path_generation_queue) > 0:
                        self.logger.loginfo(f"Path generation queue: {len(self.path_generation_queue)}")
                        keyframes, agent_pose = self.path_generation_queue.popleft()
                        self.generate_path(keyframes, agent_pose)
                if len(self.path_evaluation_queue) > 0:
                    self.logger.loginfo(f"Path evaluation queue: {len(self.path_evaluation_queue)}")
                    keyframes = self.path_evaluation_queue.popleft()
                    self.evaluate_path(keyframes)
            except Exception as e:
                self.logger.logerr(f"<inference_loop.3> Error: {e}")
                import traceback
                self.logger.logerr(f"Traceback: {traceback.format_exc()}")
                
            rate.sleep() 

    def process(self, **kwargs):
        self.logger.loginfo(f"<process.1> Updating resource...")
        try:
            self.update_resource(**kwargs)
            self.logger.loginfo(f"<process.1> Resource updated successfully.")
        except Exception as e:
            self.logger.logerr(f"<process.1> Error updating resource: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
            return

        try:
            with self.sg_lock:
                if self.sg is None:
                    self.logger.logwarn("<process.1> Scene graph is None")
                    return
                candidate_entities = self.sg.get_candidate_entities(self.etypes[0])
                reference_entities = self.sg.get_reference_entities(self.etypes[0])
                
                if candidate_entities is None or reference_entities is None:
                    self.logger.logwarn("<process.1> Candidate or reference entities is None")
                    candidate_entities = Entities()
                    reference_entities = Entities()
                else:
                    candidate_ids = candidate_entities.ids
                    reference_ids = reference_entities.ids
            self.logger.loginfo(f"<process.1> Candidate objects: {candidate_ids}, Reference objects: {reference_ids}")
                
            self.logger.logrich(f"Candidate objects: {candidate_ids}", name="candidate_objects")
            
            # Check if candidate objects have changed
            current_candidate_ids = set(candidate_ids)
            current_reference_ids = set(reference_ids)
            candidate_changed = current_candidate_ids != self.previous_candidate_ids
            reference_changed = current_reference_ids != self.previous_reference_ids
        except Exception as e:
            self.logger.logerr(f"<process.1> Error: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
        
        with self.query_count_lock:
            self.logger.loginfo(f"<process.1> Query count: {self.query_count}")
        
        try:
            # Object-based inference
            should_run_inference = (
                not self.inference_ready_event.is_set() and 
                (candidate_changed or reference_changed)
            )
                    
            inference_signal = False
            if should_run_inference and not self.inference_processing_event.is_set():
                with self.query_count_lock:
                    self.query_count['inference']['request'] += 1
                
                query_count = self.query_count
                
                self.logger.loginfo(f"<process.2> [INFERENCE] Running inference - candidate_changed: {candidate_changed}, reference_changed: {reference_changed}, "
                                f"(candidate) previous: {self.previous_candidate_ids}, current: {current_candidate_ids}, "
                                f"(reference) previous: {self.previous_reference_ids}, current: {current_reference_ids}")
                
                keyframes = self.select_keyframes(entity_type=self.etypes[0])

                if len(keyframes) > 0:
                    self.logger.logrich(
                        f"<process.2> Selected keyframe: {keyframes} (#inference_queue={len(self.inference_queue)})",
                        name="selected_keyframe"
                    )
                    self.inference_queue.append(keyframes)
                    inference_signal = True
                    
                    self.inference_processing_event.set()
                    self.pub_exploration_strategy('vg_first_inference')
                else:
                    self.logger.loginfo(f"<process.2> No keyframes selected for inference")
            else:
                # self.logger.loginfo(f"[INFERENCE] No inference needed - candidate_changed: {candidate_changed}")
                if len(current_candidate_ids) == 0 and len(current_reference_ids) == 0:
                    self.logger.loginfo(f"<process.2> No candidate or reference objects. Skip inference.")
                    with self.query_count_lock:
                        self.query_count['inference']['request'] += 1
                        self.query_count['inference']['response'] += 1
            
            # Update previous candidate IDs for next comparison
            self.previous_candidate_ids = current_candidate_ids.copy()
            self.previous_reference_ids = current_reference_ids.copy()
        except Exception as e:
            self.logger.logerr(f"<process.2> Error: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")


        try:
            with self.query_count_lock:
                query_count = self.query_count

            if not inference_signal:
                self.logger.loginfo(f"<process.3> Query count['path_generation']: {query_count['path_generation']}")
                if query_count['path_generation']['request'] == 0:
                    if self.subtask.action != "avoid":
                        # Current-image based inference
                        # with self.sg_lock:
                        #     # Get the history keyframes
                        #     history_keyframes = self.sg.history_keyframes
                        # current_keyframe_id = min(history_keyframes.keys()) if history_keyframes else None
                        # current_keyframe = copy.deepcopy(history_keyframes.get_single(current_keyframe_id) if current_keyframe_id is not None else None)

                        with self.sg_lock:
                            keyframes = self.sg.keyframes
                            
                        if len(keyframes) > 0:
                            current_keyframe_id = max(keyframes.keys())
                            current_keyframe = keyframes.get_single(current_keyframe_id)
                        else:
                            current_keyframe = None
                        
                        path_xy = copy.deepcopy(self.path_xy)
                        if not inference_signal and self.subtask.action != "avoid":
                            if path_xy.shape[0] > 30:
                                short_path_xy = copy.deepcopy(path_xy)[-30:, :]
                            else:
                                short_path_xy = path_xy
                            if current_keyframe is not None:
                                self.logger.loginfo(f"<process.3> Getting movable points for keyframe {current_keyframe_id}")
                                self.logger.loginfo(f"<process.3> occupancy_grid: {len(self.occupancy_grid.data) if self.occupancy_grid is not None else 'None'}")
                                current_keyframe.get_movable_points(self.occupancy_grid,
                                                                        style=self.path_annotation_style, node_name=self.node_name,
                                                                        suffix="_movable_points", path_history=short_path_xy)
                                if current_keyframe.movable_points is not None and len(current_keyframe.movable_points) > 0:
                                    # self.logger.loginfo(f"Keyframe {current_keyframe_id} movable points: {len(current_keyframe.movable_points)}")
                        
                                    # Check if enough time has passed since last path generation
                                    # current_time = time.time()
                                    # if current_time - self.last_path_generation_time >= self.path_generation_interval:
                                    self.path_generation_queue.append([Keyframes({current_keyframe_id: current_keyframe}), self.agent_pose])
                                    # self.last_path_generation_time = current_time
                                    self.logger.loginfo(f"<process.3> Added to path_generation_queue (periodic: {self.path_generation_interval}s)")
                                    query_count['path_generation']['request'] += 1
                                    with self.query_count_lock:
                                        self.query_count = query_count
                                    # else:
                                    #     # self.logger.loginfo(f"Skipping path_generation_queue (last: {current_time - self.last_path_generation_time:.1f}s ago)")
                                    #     pass
                                else:
                                    self.logger.loginfo(f"<process.3> No movable points available.")
                                    with self.query_count_lock:
                                        self.query_count['path_generation']['request'] += 1
                                        self.query_count['path_generation']['response'] += 1
        except Exception as e:
            self.logger.logerr(f"<process.3> Error: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
  
    def timer_callback(self, event, **kwargs):
        self.logger.logrich(f"Status: {self.status}", name='status')
        self.logger.logrich(f"Answer: {self.answer}", name='answer')
        self.logger.loginfo(f"Status: {self.status}")
        self.logger.loginfo(f"Answer: {self.answer}")
        
        try:
            if self.status == Status.STANDBY:
                self.logger.loginfo(f"subtask: {self.subtask}")
                self.standby()
        except Exception as e:
            self.logger.logerr(f"<main.1> Error in standby: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")

        try:
            if self.status == Status.PROCESSING:
                if self.processing_start_time == 0.0:
                    self.processing_start_time = time.time()
                self.logger.logrich(f"Status: {self.status}  / #inference_queue={len(self.inference_queue)}", name='status')
                self.logger.loginfo(f"<main.2> Status: {self.status}  / #inference_queue={len(self.inference_queue)}")
                # Remove the service call since we're using subscriber now
                # self._get_node_active_signal()
                if self.node_active_signal:
                    self.logger.loginfo(f"<main.2> Node active signal is set. Process...")
                    self.process(**kwargs)
                else:
                    self.logger.loginfo(f"<main.2> Node active signal is not set. Skip process.")
        except Exception as e:
            self.logger.logerr(f"<main.2> Error in process: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")

        try:
            if self.status == Status.COMPLETED:
                self.logger.loginfo(f"<main.3> Status is completed. Answer the final result.")
                self.answer_the_question(self.answer_result)
        except Exception as e:
            self.logger.logerr(f"<main.3> Error in answer_the_question: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
        
    def inference(self, keyframes):
        if len(keyframes) == 0:
            self.logger.logwarn("No keyframes provided for inference")
            return

        data = {
            'keyframes': keyframes,
            'options': self.default_inference_options,
            'target_name': self.target_name,
            'candidate_object_ids': self.sg.get_related_entities(self.etypes[0]).ids,
        }

        try:
            def handle_response(response):
                try:
                    self.inference_processing_event.clear()
                    self.logger.loginfo(f"Inference response: {response.query_id} (processing time: {response.processing_time:.2f}s, total time: {response.total_time:.2f}s)")
                    if self.status != Status.PROCESSING:
                        self.logger.loginfo(f"Ignoring inference response - status changed to {self.status}")
                        return
                    
                    if response.status != QueryStatus.COMPLETED:
                        self.logger.logerr(f"Inference failed: {response.error}")
                        return
                    
                    if response.result is not None and response.result['target_name'] != self.target_name:
                        self.logger.logwarn(f"Inference target name mismatch: {response.result['target_name']} != {self.target_name}")
                        return                
                    
                    with self.query_count_lock:
                        self.query_count['inference']['response'] += 1
                        self.logger.loginfo(f"Inference query count: {self.query_count['inference']['response']}")
                    
                    if response.result['inference_result'] is None:
                        self.logger.logwarn(f"Inference result verification failed")
                        return
                    
                    with self.inference_result_lock:
                        inference_result = response.result['inference_result']
                        if self.inference_results.empty():
                            self.inference_results.push(inference_result)
                            self.logger.loginfo(f"New inference result added: {inference_result}")
                        else:
                            prev_inference_result = self.inference_results.pop()
                            if prev_inference_result.objects != inference_result.objects:
                                self.inference_results.push(inference_result)
                                self.logger.loginfo(f"New inference result added: {inference_result}")
                            else:
                                prev_inference_result.update_confidence(1)
                                self.inference_results.push(prev_inference_result)
                                self.logger.loginfo(f"Inference result updated: {prev_inference_result}")
                        self.inference_ready_event.set()
                        self.logger.loginfo(f"Inference ready event set.")
                except Exception as e:
                    self.logger.logerr(f"Error in handle_response (inference): {e}")
                        
            query_id = self.query_manager.submit_query(
                query_type='inference_query',
                data=data,
                callback=handle_response
            )
            self.logger.loginfo(f"Inference query submitted with ID: {query_id}")
        except Exception as e:
            self.logger.logerr(f"Failed to submit inference query: {e}")
            return
 
    def add_path_evaluation_queue(self):
        # draw the total path history on the occupancy grid
        # with self.sg_lock:
        #     # Get the history keyframes
        #     history_keyframes = self.sg.history_keyframes
        # current_keyframe_id = min(history_keyframes.keys()) if history_keyframes else None
        # current_keyframe = history_keyframes.get_single(current_keyframe_id) if current_keyframe_id is not None else None
        
        # with self.sg_lock:
        #     current_keyframe_id = max(self.sg.keyframes.keys())
        #     current_keyframe = self.sg.keyframes.get_single(current_keyframe_id)
        
        try:
            current_keyframe = self.current_image
            if current_keyframe is None:
                self.logger.logwarn("No current image provided for path evaluation")
                return
            
            with self.sg_lock:
                sg = self.sg
            
            num_latest_keyframes = 3
            if len(sg.keyframes) != 0:
                keyframe_ids = np.array(list(sg.keyframes.keys()))[::-1]
                latest_keyframe_ids = keyframe_ids[:num_latest_keyframes]
            else:
                latest_keyframe_ids = []
            
            latest_keyframes = {}
            for id in latest_keyframe_ids:
                latest_keyframes[id] = sg.keyframes.get_single(id)
            
            latest_keyframes = Keyframes(latest_keyframes)
                        
            if current_keyframe is not None and self.occupancy_grid is not None:
                path_xy = copy.deepcopy(self.path_xy)
                if path_xy is None:
                    return
                current_keyframe.get_movable_points(self.occupancy_grid,
                                    style=self.path_annotation_style, node_name=self.node_name,
                                    suffix="_total_path_history", path_history=path_xy)
                latest_keyframes["-1"] = current_keyframe
                                
                self.path_evaluation_queue.append(latest_keyframes)
                self.path_evaluation_processing_event.set()
            
            # pose = current_keyframe.pose
            # self.logger.loginfo(f"Pose: {pose}")
            # if current_keyframe is not None and self.occupancy_grid is not None:
            #     path_xy = copy.deepcopy(self.path_xy)
            #     self.logger.loginfo(f"Path xy: {path_xy}")
            #     if path_xy is None:
            #         return
            #     agent_position = pose['position']
            #     current_gx, current_gy = self.occupancy_grid.world_to_grid(agent_position[0], agent_position[1])
            #     current_keyframe._visualize_movable_points_on_occupancy_grid(self.occupancy_grid, None, [current_gx, current_gy], 
            #                                                     self.path_annotation_style['occupancy_grid'], self.node_name, "_total_path_history", 
            #                                                     None, path_xy)

                # self.path_evaluation_queue.append(Keyframes({'-1': current_keyframe}))
                # self.logger.loginfo(f"Added to path_evaluation_queue")
        except Exception as e:
            self.logger.logerr(f"Error in add_path_evaluation_queue: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
 
    def validation_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            if self.validation_running.is_set():
                rate.sleep()
                continue
                        
            if self.status != Status.PROCESSING or not self.inference_ready_event.is_set():
                rate.sleep()
                continue
            
            self.logger.loginfo(f"<validation_loop.1> Validation running set.")
            self.validation_running.set()
            
            try:
                self.logger.loginfo(f"<validation_loop.1> Validating...")
                self.validate()
            finally:
                self.logger.loginfo(f"<validation_loop.1> Validation running cleared.")
                self.validation_running.clear()

            rate.sleep()    
 
    def validate(self):
        with self.inference_result_lock:
            if self.inference_results.empty():
                self.logger.logwarn("<validate.1> No inference result found")
                return
            inference_result = self.inference_results.peek()
            
        self.logger.loginfo(f"<validate.1> Validation inference result: {inference_result}")
            
        if not inference_result.has_candidate:
            self.logger.loginfo(f"<validate.1> No candidate found. Skip validation.")
            return
            
        if inference_result.confidence >= self.confidence_threshold[1]:
            self.logger.loginfo(f"<validate.1> Confidence >= {self.confidence_threshold[1]}. Skip validation.")
            return
            
        if inference_result.validation_count >= self.validation_count_threshold:
            self.logger.loginfo(f"<validate.1> Validation count >= {self.validation_count_threshold}. Skip validation.")
            return

        # Annotate keyframes with selected objects
        styles = {
            'candidate': {'show': True, 'color': 'green', 'ids': inference_result.objects.ids},  # Highlight selected objects
        }
        with self.sg_lock:
            for etype in self.etypes:
                self.sg.keyframes.annotate(
                    styles, node_name=self.node_name, suffix=f"_annotated_inference_{etype}", etype=etype)

        # Select keyframes
        try:
            keyframes = self.select_keyframes(entity_type=self.etypes[0], target_eids=inference_result.objects.ids)
        except Exception as e:
            self.logger.logerr(f"<validate.1> select_keyframes exception: {e}")
            return

        if len(keyframes) == 0:
            self.logger.logwarn("<validate.1>No keyframes provided for inference")
            return

        # Query LVLM with the selected keyframes
        data = {
            'keyframes': keyframes,
            'options': self.default_validate_options,
            'target_name': self.target_name,
            'candidate_object_ids': inference_result.objects.ids,
            'inference_result': inference_result,
        }
        
        try:
            def handle_response(response):
                self.logger.loginfo(f"Validation response: {response.query_id} (processing time: {response.processing_time:.2f}s, total time: {response.total_time:.2f}s)")
                if self.status != Status.PROCESSING:
                    self.logger.loginfo(f"Ignoring validation response - status changed to {self.status}")
                    return
                
                if response.status != QueryStatus.COMPLETED:
                    self.logger.logerr(f"Validation failed: {response.error}")
                    return
                
                if response.result['confidence'] is not None and response.result['target_name'] != self.target_name:
                    self.logger.logwarn(f"Validation target name mismatch: {response.result['target_name']} != {self.target_name}")
                    return
                
                with self.query_count_lock:
                    self.query_count['validation']['response'] += 1
                    self.logger.loginfo(f"Validation query count: {self.query_count['validation']['response']}")
                
                if response.result is None:
                    self.logger.logwarn(f"Validation result verification failed")
                    return
                
                obs_confidence = response.result['confidence']
                
                with self.inference_result_lock:
                    for i, (_, result) in enumerate(self.inference_results._heap):
                        if result == response.result['inference_result']:
                            result.update_confidence(obs_confidence)
                            self.logger.loginfo(f"Validation updated: {result}")
                            break
            
            query_id = self.query_manager.submit_query(
                query_type='validation_query',
                data=data,
                callback=handle_response
            )
            self.logger.loginfo(f"<validate.1> Validation query submitted with ID: {query_id}")
        except Exception as e:
            self.logger.logerr(f"<validate.1> Failed to submit validation query: {e}")
            return

    def generate_path(self, keyframes, agent_pose):
        self.logger.loginfo(f"Generate path with {len(keyframes)} keyframes and agent pose: {agent_pose}")
        if self.inference_ready_event.is_set():
            self.logger.loginfo(f"Inference ready event is set. Skip path generation.")
            return
        
        if len(keyframes) == 0:
            self.logger.logwarn("No keyframes provided for path generation")
            return
        
        data = {
            'keyframes': keyframes,
            'options': self.default_path_generation_options,
            'target_name': self.target_name,
            'agent_pose': agent_pose
        }
        
        try:
            def handle_response(response):
                try:
                    self.logger.loginfo(f"Path generation response: {response.query_id} (processing time: {response.processing_time:.2f}s, total time: {response.total_time:.2f}s)")
                    if self.status != Status.PROCESSING:
                        self.logger.loginfo(f"Ignoring path generation response - status changed to {self.status}")
                        return
                    
                    if response.status != QueryStatus.COMPLETED:
                        self.logger.logerr(f"Path generation failed: {response.error}")
                        return
                    
                    if response.result['path_points'] is not None and response.result['target_name'] != self.target_name:
                        self.logger.logwarn(f"Path generation target name mismatch: {response.result['target_name']} != {self.target_name}")
                        return
                    
                    # If agent pose is different, skip path generation
                    query_agent_pose = response.result['agent_pose']
                    current_agent_pose = self.agent_pose
                    dist = math.hypot(current_agent_pose['position'][0] - query_agent_pose['position'][0], current_agent_pose['position'][1] - query_agent_pose['position'][1])
                    if dist > self.agent_pose_threshold:
                        self.logger.logwarn(f"Path generation agent pose mismatch: dist {dist} > {self.agent_pose_threshold}, {current_agent_pose['position']} != {query_agent_pose['position']}")
                        with self.query_count_lock:
                            self.query_count['path_generation']['request'] = 0
                        return
                    
                    with self.query_count_lock:
                        if self.query_count['path_generation']['request'] == 0:
                            self.logger.loginfo(f"Path generation request is 0. Skip path generation response.")
                            return
                        
                        self.query_count['path_generation']['response'] += 1
                        self.logger.loginfo(f"Path generation query count: {self.query_count['path_generation']['response']}")
                    
                    if response.result is None:
                        self.logger.logwarn(f"Path generation result verification failed")
                        return
                    
                    with self.path_generation_result_lock:
                        self.generated_path_points = response.result['path_points']
                        self.path_generation_ready_event.set()
                        self.logger.loginfo(f"Path generation ready event set.")
                except Exception as e:
                    self.logger.logerr(f"Error in handle_response (path generation): {e}")
            
            query_id = self.query_manager.submit_query(
                query_type='path_generation_query',
                data=data,
                callback=handle_response
            )
            self.logger.loginfo(f"Path generation query submitted with ID: {query_id}")
        except Exception as e:
            self.logger.logerr(f"Failed to submit path generation query: {e}")
            return

    def evaluate_path(self, history_keyframes):
        if len(history_keyframes) == 0:
            self.logger.logwarn("No history keyframes provided for path evaluation")
            return
        
        data = {
            'keyframes': history_keyframes,
            'options': self.default_path_evaluation_options,
            'target_name': self.target_name,
        }
        
        try:
            def handle_response(response):
                self.logger.loginfo(f"Path evaluation response: {response.query_id} (processing time: {response.processing_time:.2f}s, total time: {response.total_time:.2f}s)")
                if self.status != Status.PROCESSING:
                    self.logger.loginfo(f"Ignoring path evaluation response - status changed to {self.status}")
                    return
                
                if response.status != QueryStatus.COMPLETED:
                    self.logger.logerr(f"Path evaluation failed: {response.error}")
                    return
                
                if response.result['mission_status'] is not None and response.result['target_name'] != self.target_name:
                    self.logger.logwarn(f"Path evaluation target name mismatch: {response.result['target_name']} != {self.target_name}")
                    return
                
                with self.query_count_lock:
                    self.query_count['path_evaluation']['response'] += 1
                    self.logger.loginfo(f"Path evaluation query count: {self.query_count['path_evaluation']['response']}")
                
                if response.result is None:
                    self.logger.logwarn(f"Path evaluation result verification failed")
                    return
                
                mission_status = response.result['mission_status']
                if mission_status == 1:
                    self.mission_completed_event.set()
                    self.logger.loginfo(f"Mission completed.")
                else:
                    self.logger.loginfo(f"Mission not completed.")
                self.path_evaluation_processing_event.clear()
            
            query_id = self.query_manager.submit_query(
                query_type='path_evaluation_query',
                data=data,
                callback=handle_response
            )
            self.logger.loginfo(f"Path evaluation query submitted with ID: {query_id}")
        except Exception as e:
            self.logger.logerr(f"Failed to submit path evaluation query: {e}")
            return

    def update_path_points(self) -> None:
        if self.subtask.action == "avoid":
            return
        
        path_points = np.empty((0, 3))
        if self.inference_ready_event.is_set():
            with self.inference_result_lock:
                inference_result = self.inference_results.peek()
                
            with self.sg_lock:
                try:
                    objects_list = list(inference_result.objects.values())
                    if len(objects_list) == 0:
                        self.logger.logwarn("<update_path_points.1> No objects found in inference result.")
                        return
                    
                    if self.default_inference_options['prompt']['action'] == 'inference_follow_between' and len(objects_list) == 2:
                        with self._traversable_lock:
                            traversable_points = self.traversable_points
                            
                        object1 = objects_list[0]
                        object2 = objects_list[1]
                        path_points = self.get_path_between_objects_pass_through(traversable_points, inference_result.keyframes, object1, object2, entry_offset=0.2, exit_offset=0.2)
                    else:
                        object = objects_list[0]
                        kfs = inference_result.keyframes.get_entity_ids(object.id, etype=self.etypes[0])
                        kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
            
                        with self._traversable_lock:
                            traversable_points = self.traversable_points
                            
                        path_points = object.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
                    if path_points is None:
                        self.logger.loginfo("<update_path_points.1> No path points found.")
                        return
                except Exception as e:
                    self.logger.logerr(f"<update_path_points.1> Error occurs: {e}")
                    import traceback
                    self.logger.logerr(f"Traceback: {traceback.format_exc()}")
                    return
        else:
            try:
                if not self.path_generation_ready_event.is_set():
                    self.logger.loginfo("<update_path_points.1> Path generation ready event is not set. Skip update_path_points.")
                    return
                
                with self.path_generation_result_lock:
                    generated_path_points = copy.deepcopy(self.generated_path_points)
                
                if generated_path_points is None:
                    return
                
                generated_path_points = np.array(generated_path_points)
                if self.traversable_points is None:
                    self.logger.loginfo("<update_path_points.1> No traversable points available.")
                    return
                nearest_points = find_closest_point(generated_path_points, self.traversable_points)
                if nearest_points.shape[1] == 2:
                    nearest_points = np.hstack([nearest_points, np.zeros((len(nearest_points), 1))])

                path_points = np.concatenate((path_points, nearest_points), axis=0)
            except Exception as e:
                self.logger.logerr(f"<update_path_points.1> Error occurs: {e}")
                import traceback
                self.logger.logerr(f"Traceback: {traceback.format_exc()}")
                return
        
        if len(path_points) == 0:
            return

        try:
            current_path_points = None
            if len(path_points) > 0:
                # Convert list to numpy array for filter_close_points
                path_points_array = np.array(path_points)
                new_filtered_path_points = filter_close_points(path_points_array, self.min_point_spacing)
                kept_mask, wps_keep = filter_waypoints_by_path(new_filtered_path_points, self.path_xy, self.radius)
                active_waypoints = []
                for i, keep in enumerate(kept_mask):
                    if keep:
                        active_waypoints.append(new_filtered_path_points[i])
                        
                current_path_points = np.array(active_waypoints)
                
                if len(current_path_points) == 0 and not self.active_clients.is_running:
                    self.logger.loginfo("<update_path_points.2> No current path points available and active clients are not running. Skip update_path_points.")
                    return
                
                with self.navigation_lock:
                    self.path_points = current_path_points
            self.logger.logrich(f"path_points: #total={len(path_points)}, #valid={len(current_path_points) if current_path_points is not None else 0}", name="path_points")
            self.logger.loginfo(f"<update_path_points.2> path_points: #total={len(path_points)}, #valid={len(current_path_points) if current_path_points is not None else 0}", name="path_points")

        except Exception as e:
            self.logger.logerr(f"<update_path_points.2> Error occurs: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
            return

    def update_resource(self, **kwargs):
        self.logger.loginfo(f"<update_resource.1> Starting scene graph update...")
        try:
            self.scene_graph_clients.update_scene_graph(**kwargs)
            self.logger.loginfo(f"<update_resource.1> Scene graph update completed.")
        except Exception as e:
            self.logger.logerr(f"<update_resource.1> Error in scene graph update: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
            raise

        self.logger.loginfo(f"<update_resource.2> Starting annotation...")
        styles = {
            'candidate': {'show': True, 'color': 'green'},
            'reference': {'show': True, 'color': 'red'},
        }
        try:
            with self.sg_lock:
                for etype in self.etypes:
                    self.sg.keyframes.annotate(styles, node_name=self.node_name, suffix=f"_annotated_global_{etype}", etype=etype)
            self.logger.loginfo(f"<update_resource.2> Annotation completed.")
        except Exception as e:
            self.logger.logerr(f"<update_resource.2> Error in annotation: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
            raise
        
        self.updated_resource = True
        self.logger.loginfo(f"<update_resource.3> Resource update completed successfully.")

    def navigate(self):
        # Avoid navigation if the subtask is "avoid"
        self.logger.loginfo(f"<navigate.1> Subtask action: {self.subtask.action}")
        if self.subtask.action == "avoid":
            return
        
        if self.inference_processing_event.is_set():
            self.logger.loginfo(f"<navigate.1> Inference processing event is set. Skip navigation.")
            return
        
        if self.path_evaluation_processing_event.is_set():
            self.logger.loginfo(f"<navigate.1> Path evaluation processing event is set. Skip navigation.")
            return
        
        with self.navigation_lock:
            current_path_points = self.path_points
        
        with self.query_count_lock:
            query_count = self.query_count
        
        self.logger.loginfo(f"<navigate.1> current_path_points: {len(current_path_points) if current_path_points is not None else 'None'}")
        
        if current_path_points is None:
            if self.arrived_event.is_set():
                self.logger.loginfo(f"<navigate.1> Arrived event is set. Skip navigation.")
                return
            
            # Reset path generation query count if frontier arrived
            if self.latest_exploration_strategy == 'semantic_frontier' \
                and self.exploration_status == 'frontier_arrived':
                self.pub_exploration_strategy('vg_first')
                query_count['path_generation']['request'] = 0
                query_count['path_generation']['response'] = 0
                self.logger.loginfo(f"<navigate.1> Frontier arrived. Reset path generation query count.")
                with self.query_count_lock:
                    self.query_count = query_count
            elif self.latest_exploration_strategy == 'semantic_frontier' \
                and self.exploration_status == 'no_frontier':
                self.pub_exploration_strategy('vg_first')
                query_count['path_generation']['request'] = 0
                query_count['path_generation']['response'] = 0
                self.logger.loginfo(f"<navigate.1> No frontier. Reset path generation query count.")
                with self.query_count_lock:
                    self.query_count = query_count
                    
            self.logger.loginfo(f"<navigate.1> query_count: {query_count}")
            if query_count['inference']['response'] >= 1 and query_count['path_generation']['response'] >= 1:
                self.logger.loginfo(f"<navigate.1> No path points available, but inference and path generation queries have been completed one or more times. Switching to semantic_frontier strategy.")
                
                if self.exploration_status == 'no_frontier':
                    self.mission_completed_event.set()
                    self.logger.loginfo(f"<navigate.1> No frontier and failed to generate path. Mission completed.")
                    return
                
                if self.active_clients.is_running:
                    self.active_clients.end()
                    self.logger.logrich(f"Active clients: ended", name='navigation')
                    self.logger.loginfo(f"<navigate.1> Active clients: ended")
                
                if self.answer_result is None:
                    if self.arrived_time is None:
                        self.logger.loginfo(f"<navigate.1> No arrived time. Publishing semantic_frontier.")
                        self.pub_exploration_strategy('semantic_frontier')
                    elif self.arrived_time is not None and time.time() - self.arrived_time > self.semantic_frontier_pub_duration:
                        self.logger.loginfo(f"<navigate.1> Arrived time is not None and {self.semantic_frontier_pub_duration} seconds has passed. Publishing semantic_frontier.")
                        self.pub_exploration_strategy('semantic_frontier')
            return

        if len(current_path_points) > 0:
            try:
                if self.active_clients.start():
                    self.logger.logrich(f"<navigate.2> Active clients: running", name='navigation')
                    # self.logger.loginfo(f"self.active_clients starts!!")
                    self.pub_exploration_strategy('vg_first')
                    self.arrived_time = None
                    path_points_marker = make_marker_array_from_points(
                        current_path_points, ns="path_points", color=(0.5, 0.5, 0.5, 0.5), frame_id=self.frame_id)
                    self.path_points_pub.publish(path_points_marker)
                    self.logger.loginfo(f"<navigate.2> publish path_points: #total={len(current_path_points)}, #valid={len(current_path_points) if current_path_points is not None else 0}")
                else:
                    self.logger.logrich(f"Active clients: does not run", name='navigation')
                    # self.logger.loginfo(f"self.active_clients doesn't start!!")
                    self.pub_exploration_strategy('semantic_frontier')
            except Exception as e:
                self.logger.logerror(f"<navigate.2> Error in navigation active_clients: {e}")
        else:
            if self.active_clients.is_running:
                self.logger.loginfo(f"<navigate.2> len(self.path_points) <= 0: {len(current_path_points)}")
                try:
                    success = self.active_clients.end()
                    self.arrived_event.set()
                    self.arrived_time = time.time()
                    self.path_points = None # reset path_points
                    self.logger.logrich(f"Active clients: ended", name='navigation')
                    self.logger.loginfo(f"<navigate.2> Active clients: ended")
                    # self.pub_exploration_strategy('semantic_frontier')
                    # self.logger.loginfo(f"Active clients ended and inference signal put.")
                except Exception as e:
                    self.logger.logerror(f"<navigate.2> Error ending active_clients: {e}")
            else:
                self.logger.loginfo(f"<navigate.2> Active clients: not running")

    def block_objects(self, traversable_points, keyframes, objects):
        if traversable_points is None:
            self.logger.loginfo("No traversable points available to block objects.")
            return
            
        objects_list = list(objects.values())
        
        if len(objects_list) not in [1, 2]:
            self.logger.logwarn(f"Must provide 1 or 2 object IDs to block path. Got {len(objects_list)}.")
            return
        
        if len(objects_list) == 1: # Expand the bbox of the object
            obj = objects_list[0]
            self.logger.logwarn(f"Object {obj.id} is_object: {obj.is_object}")
            if obj.is_object:
                min_bbox = np.array(obj.min_bbox, dtype=float)
                max_bbox = np.array(obj.max_bbox, dtype=float)
            else:
                kfs = keyframes.get_entity_ids(obj.id, etype=self.etypes[0])
                kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
                center = obj.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
                if center is None:
                    self.logger.logwarn("Failed to get closest traversable point for object, skipping bbox blocking")
                    return
                min_bbox = [center[0] - 0.7, center[1] - 0.7]
                max_bbox = [center[0] + 0.7, center[1] + 0.7]
            self.block_inflated_bbox(min_bbox, max_bbox)
        elif len(objects_list) == 2: # Block the segment between the two objects
            obj1 = objects_list[0]
            obj2 = objects_list[1]
            self.logger.logwarn(f"Object {obj1.id} is_object: {obj1.is_object}")
            self.logger.logwarn(f"Object {obj2.id} is_object: {obj2.is_object}")
            if obj1.is_object:
                center1 = np.array(obj1.center[:2], dtype=float)
            else:
                kfs = keyframes.get_entity_ids(obj1.id, etype=self.etypes[0])
                kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
                center1 = obj1.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
                if center1 is None:
                    self.logger.logwarn("Failed to get closest traversable point for object1, skipping segment blocking")
                    return
            if obj2.is_object:
                center2 = np.array(obj2.center[:2], dtype=float)
            else:
                kfs = keyframes.get_entity_ids(obj2.id, etype=self.etypes[0])
                kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
                center2 = obj2.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
                if center2 is None:
                    self.logger.logwarn("Failed to get closest traversable point for object2, skipping segment blocking")
                    return
            self.block_segment_with_radius(center1, center2)
            
    def block_inflated_bbox(self, min_bbox, max_bbox, inflation=0.8):
        min_bbox = [float(min_bbox[0]), float(min_bbox[1])]
        max_bbox = [float(max_bbox[0]), float(max_bbox[1])]
        try:
            self.logger.loginfo("[TA] calling apply_bbox ...")
            resp = self.trav_bbox_cli(
                min_bbox=min_bbox,
                max_bbox=max_bbox,
                inflation=float(inflation)
            )
            if resp.success:
                self.logger.loginfo(f"[TA] block_bbox OK: {resp.message}")
            else:
                self.logger.logwarn(f"[TA] block_bbox FAIL: {resp.message}")
        except rospy.ServiceException as e:
            self.logger.logerr(f"[TA] block_bbox service error: {e}")
            
    def block_segment_with_radius(self, center1, center2, radius=0.4):
        c1 = [float(center1[0]), float(center1[1])]
        c2 = [float(center2[0]), float(center2[1])]
        try:
            resp = self.trav_segment_cli(
                center1=c1,
                center2=c2,
                radius=float(radius)
            )
            if resp.success:
                self.logger.loginfo(f"[TA] block_segment OK: {resp.message}")
            else:
                self.logger.logwarn(f"[TA] block_segment FAIL: {resp.message}")
        except rospy.ServiceException as e:
            self.logger.logerr(f"[TA] block_segment service error: {e}")

    def get_path_between_objects_pass_through(self, traversable_points, keyframes, object1, object2,
                                            min_dist=0.20,
                                            entry_offset=0.20,
                                            exit_offset=0.20):
        """
        두 물체 중심을 잇는 선(센터선) 위의 traversable 포인트만 사용해
        각 물체에 가장 가까운 점 2개를 고르고, 그 중간에 있는 traversable 포인트를 기준(mid)으로
        경로를 생성함. 기본 방향은 센터선 수직, 좁은 통로가 감지되면 통로 방향(주성분)으로 교체.
        반환: [(x,y,z), ...]  (entry → mid → exit)
        """
        if traversable_points is None or len(traversable_points) == 0:
            self.logger.loginfo("No traversable points available.")
            return None

        obj1 = object1
        obj2 = object2

        # Center(2D)
        if obj1.is_object:
            c1 = np.array(obj1.center[:2], dtype=float)
        else:
            kfs = keyframes.get_entity_ids(obj1.id, etype=self.etypes[0])
            kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
            
            c1 = obj1.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
            if c1 is None:
                self.logger.logwarn("Failed to get closest traversable point for object1")
                return None
                
        if obj2.is_object:
            c2 = np.array(obj2.center[:2], dtype=float)
        else:
            kfs = keyframes.get_entity_ids(obj2.id, etype=self.etypes[0])
            kf_id, kf = kfs.get_closest_keyframe(self.agent_pose['position'])
                        
            c2 = obj2.get_closest_traversable_point(traversable_points, kf_id=kf_id, kf=kf, agent_position=self.agent_pose['position'])
            if c2 is None:
                self.logger.logwarn("Failed to get closest traversable point for object2")
                return None
        u = c2 - c1
        L = float(np.linalg.norm(u))
        if L < 1e-9:
            # If the two objects are almost at the same position, use an arbitrary axis
            u = np.array([1.0, 0.0], dtype=float); L = 1.0
        u /= L  # unit vector along the center line

        # Traversable points(2D)
        T = np.asarray(traversable_points, dtype=float)
        T2 = T[:, :2]

        # Calculate the projection of each traversable point onto the center line
        # q = c1 + t*u, t ∈ [0, L]
        v = T2 - c1[None, :]
        t = (v @ u)                 # (M,)
        perp = v - t[:, None] * u[None, :]
        d_perp = np.linalg.norm(perp, axis=1)  # distance to the center line

        # Filter points that are around the center line
        tol_line = 0.06
        mask = (d_perp <= tol_line) & (t >= -0.02*L) & (t <= 1.02*L)
        if not np.any(mask):
            self.logger.loginfo("No traversable points on the center line.")
            return None

        t_in = t[mask]
        P_in = T2[mask]  # (N_in, 2)

        # The most closest points to the objects on the center line
        idx1 = int(np.argmin(np.abs(t_in - 0.0)))
        idx2 = int(np.argmin(np.abs(t_in - L)))
        p1_line = P_in[idx1]   # object1
        p2_line = P_in[idx2]   # object2

        # Midpoint of the two closest points on the center line
        mid_geometric = 0.5 * (p1_line + p2_line)

        # Select the point closest to the geometric midpoint
        mid_idx = int(np.argmin(np.linalg.norm(P_in - mid_geometric[None, :], axis=1)))
        center_mid = P_in[mid_idx]  # (2,)

        # Default direction is perpendicular to the center line
        n = np.array([-u[1], u[0]], dtype=float)

        # 통로 방향 추정: center_mid 주변의 traversable 포인트들로 PCA
        # 이방성(lambda_max / lambda_min) 높을 때 주성분 방향으로 교체
        R_nbh = 0.8       # 이웃 반경(m)
        aniso_thresh = 2.0  # 이방성 임계
        d_mid = np.linalg.norm(T2 - center_mid[None, :], axis=1)
        Nbh = T2[d_mid <= R_nbh]
        if len(Nbh) >= 8:
            X = Nbh - Nbh.mean(axis=0, keepdims=True)
            C = (X.T @ X) / max(1, len(X) - 1)
            # 2x2 공분산의 고유분해
            w, V = np.linalg.eigh(C)   # w 오름차순, V[:,i] 해당 고유벡터
            lam_min, lam_max = float(w[0]), float(w[-1])
            aniso = lam_max / max(lam_min, 1e-9)
            dir_corr = V[:, -1]  # 주성분(통로 축)
            dir_corr = dir_corr / (np.linalg.norm(dir_corr) + 1e-12)

            if aniso >= aniso_thresh:
                # 통로가 뚜렷하면 통로 축과 "평행"하게 이동하도록 방향 교체
                # self.logger.loginfo(f"Anisotropy detected: {aniso:.2f} (threshold={aniso_thresh}), using corridor direction.")
                n = dir_corr

        # Calculate entry/exit points based on object bbox edges
        # Get bbox dimensions for each object (with safety check)
        
        if obj1.is_object:
            obj1_bbox_half = np.array([obj1.width, obj1.depth], dtype=float) * 0.5
        else:
            obj1_bbox_half = np.array([0.5, 0.5], dtype=float)  # 1m x 1m default
            
        if obj2.is_object:
            obj2_bbox_half = np.array([obj2.width, obj2.depth], dtype=float) * 0.5
        else:
            obj2_bbox_half = np.array([0.5, 0.5], dtype=float)  # 1m x 1m default
        
        # Calculate bbox edge points in the perpendicular direction (n)
        # Project bbox dimensions onto the perpendicular direction
        obj1_edge_dist = np.abs(np.dot(obj1_bbox_half, n))
        obj2_edge_dist = np.abs(np.dot(obj2_bbox_half, n))
        
        # Use the smaller edge distance plus base offset
        # min_edge_dist = min(obj1_edge_dist, obj2_edge_dist)
        mean_edge_dist = (obj1_edge_dist + obj2_edge_dist) / 2
        dynamic_entry_offset = float(entry_offset) + mean_edge_dist
        dynamic_exit_offset = float(exit_offset) + mean_edge_dist
        
        # entry / exit
        entry = center_mid - n * max(float(entry_offset), dynamic_entry_offset)
        exit_ = center_mid + n * max(float(exit_offset), dynamic_exit_offset)

        W = np.array([
            [entry[0],      entry[1],      0.0],
            [center_mid[0], center_mid[1], 0.0],
            [exit_[0],      exit_[1],      0.0],
        ], dtype=np.float32)

        # If the distance between the first two waypoints is too small, remove the middle one
        # def _dist(a, b):
        #     return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))
        # if _dist(W[0], W[1]) < min_dist or _dist(W[1], W[2]) < min_dist:
        #     W = np.array([W[0], W[2]], dtype=np.float32)

        # Get the closest traversable points to the waypoints
        W = find_closest_point(W, traversable_points).astype(np.float32)

        # Sort waypoints by distance to the agent position
        agent_xy = np.array(self.agent_pose['position'][:2], dtype=float)
        W = sorted(W, key=lambda pt: np.linalg.norm(pt[:2] - agent_xy))

        return W

    def start_threads(self):
        """Start all background threads"""
        if self.executor is not None:
            self.logger.logwarn("Threads are already running. Stopping them first.")
            self.stop_threads()
        
        # Clear the stop event for new threads
        self.stop_event.clear()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.main_thread.start()
        
        self.executor.submit(self.inference_loop, 2.0)
        self.executor.submit(self.validation_loop, 3.0)
        self.executor.submit(self.navigation_loop, 5.0)
        
        self.logger.loginfo("All threads started successfully.")

    def stop_threads(self):
        """Stop all background threads"""
        self.logger.loginfo("Setting stop event...")
        self.stop_event.set()
        
        if self.executor is not None:
            self.logger.loginfo("Stopping executor...")
            self.executor.shutdown(wait=True)
            self.executor = None
            
        if self.main_thread is not None and self.main_thread.is_alive():
            self.logger.loginfo("Waiting for main thread to finish...")
            self.main_thread.join(timeout=2.0)  # Wait up to 2 seconds
            if self.main_thread.is_alive():
                self.logger.logwarn("Main thread did not stop gracefully, but continuing...")
            
        self.logger.loginfo("All threads stopped successfully.")

    def main_loop(self):
        """Override main_loop to support stop_event"""
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            # Check if already processing to prevent duplicate execution
            if self.main_running.is_set():
                self.logger.loginfo(f"<main_loop> Main is already running. Let's sleep..")
                rate.sleep()
                continue

            self.logger.loginfo(f"<main_loop> Main is not running. Let's process..")
            self.main_running.set()
            self.timer_callback(None)
            self.main_running.clear()
            self.logger.loginfo(f"<main_loop> Main is cleared. Let's sleep..")
            rate.sleep()

    def navigation_loop(self, hz):
        """Override navigation_loop to support stop_event"""
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            self.logger.logrich(f"<navigation_loop.1> Resource status: {'Updated' if self.updated_resource else 'Not yet'}", name="resource_status")
            try:
                if not self.updated_resource:
                    rate.sleep()
                    self.logger.loginfo(f"<navigation_loop.1> Resource is not updated. Let's sleep..")
                    continue

                if self.subtask == None:
                    rate.sleep()
                    self.logger.loginfo(f"<navigation_loop.1> Subtask is None. Let's sleep..")
                    continue

                if self.navigation_running.is_set():
                    rate.sleep()
                    self.logger.loginfo(f"<navigation_loop.1> Navigation is already running. Let's sleep..")
                    continue

                if self.status != Status.PROCESSING:
                    rate.sleep()
                    self.logger.loginfo(f"<navigation_loop.1> Status is not processing. Let's sleep..")
                    continue

                self.logger.loginfo(f"<navigation_loop.1> Navigation is not running. Let's process..")
                self.navigation_running.set()
                self.update_path_points()
                self.logger.loginfo(f"<navigation_loop.2> Updated path_points: #={len(self.path_points) if self.path_points is not None else 'None'}")
                
                if self.node_active_signal:
                    self.navigate()

                self.navigation_running.clear()
                self.logger.loginfo(f"<navigation_loop.1> Navigation is cleared. Let's sleep..")
                rate.sleep()
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.1> Error in navigation: {e}")
                import traceback
                self.logger.logerr(f"Traceback: {traceback.format_exc()}")
                self.navigation_running.clear()
                rate.sleep()

    def _quat_to_rotation_matrix(self, quat):
        """
        Convert quaternion [x, y, z, w] to 3x3 rotation matrix using numpy
        """
        x, y, z, w = quat
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return np.eye(3)
        
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        rotation_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return rotation_matrix

if __name__ == "__main__":
    if use_rospy:
        # ROS version
        rospy.init_node('visual_grounding')
        node_name = rospy.get_name()
        node_name = node_name.strip('/')
        logger_cfg = LoggerConfig(
            quiet=False, prefix=f"VisualFollower{node_name.split('_')[-1]}",
            log_path=os.path.join(LOG_DIR, f'{node_name}.log'),
            no_intro=False
        )
        vg = VisualFollower(node_name=node_name, logger_cfg=logger_cfg)
        
        # Start all background threads
        vg.start_threads()
        
        rospy.spin()
    else:
        # Python version
        logger_cfg = LoggerConfig(
            quiet=False, prefix='Test for VisualFollower',
            log_path=os.path.join(LOG_DIR, 'visual_follower.log'),
            no_intro=False
        )
        vg = VisualFollower(
            candidate_names=['coffee table', 'path', 'sofa'], reference_names=[],
            node_name="test", logger_cfg=logger_cfg, use_ros=False
        )

        DATA_DIR = "/ws/external/test_data/offline_map"
        dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, d))]
        dir_sorted = sorted(dirs, key=os.path.getmtime)

        for dir in dir_sorted:
            vg.timer_callback(None, dir=dir)
            time.sleep(0.1)


import os
import sys
sys.path.append('/ws/external/')
import cv2
import glob
import time
import random
import threading, queue, logging, traceback, ctypes
import numpy as np
import operator
from functools import reduce
from collections import defaultdict
from enum import Enum
from PIL import Image
from datetime import datetime
from typing import List
from scipy.optimize import minimize
import concurrent.futures
import json
import copy
import queue
import hashlib

from ai_module.src.utils.logger import Logger
from ai_module.src.utils.utils import (pointcloud2_to_xy_array, is_equal, find_closest_point, \
    filter_waypoints_by_path, make_marker_array_from_points, min_distance)
from ai_module.src.utils.refine_bbox import refine_bbox
from ai_module.src.visual_grounding.scripts.utils.utils_message import object_to_marker, point_3d_to_marker
from ai_module.src.utils.visualizer import _color_palette
from ai_module.src.visual_grounding.scripts.models.base_model import BaseModel
from ai_module.src.visual_grounding.scripts.services.scene_graph_clients import SceneGraphClients
from ai_module.src.visual_grounding.scripts.services.active_clients import ActiveClients
from ai_module.src.visual_grounding.scripts.structures.occupancy_grid import CustomOccupancyGrid
from ai_module.src.visual_grounding.scripts.structures.inference_result import InferenceResult, InferenceResults, get_confidence
from ai_module.src.visual_grounding.scripts.structures.aggregated_result import AggregatedResult
from ai_module.src.visual_grounding.scripts.structures.hull_grouper import GridGrouper

# VLMS
from ai_module.src.visual_grounding.scripts.vlms.loaders.vision_client import VisionLlmClient, _is_retryable_llm_error
from ai_module.src.visual_grounding.scripts.vlms.prompt_renderer import PromptRenderer, SystemInstructionRenderer
from ai_module.src.visual_grounding.scripts.vlms.utils.helpers import parse_json
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframes, Keyframe
from ai_module.src.visual_grounding.scripts.structures.entity import Entity
from ai_module.src.visual_grounding.scripts.structures.answer import Answer
from ai_module.src.visual_grounding.scripts.structures.scene_graph import NodeLevel
from ai_module.src.visual_grounding.scripts.structures.bbox import BBoxes
from ai_module.src.utils.visualizer import Visualizer
from ai_module.src.utils.utils_traversability import filter_disconnected_traversable, load_pcd_ascii_with_fields, save_pcd_ascii_with_header
from ai_module.src.utils.utils_pose import theta_from_agent_pose
from ai_module.src.utils.utils_visualize import _polyline_xy_to_ribbon_mesh3d, build_ribbon_mesh

import rerun as rr

try:
    import rospy
except:
    sys.path.append("/ws/external/ai_module/src/utils/debug")
    import ai_module.src.utils.debug
    import rospy
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Int32, Empty
from rosgraph_msgs.msg import Clock
from visualization_msgs.msg import Marker, MarkerArray
from visual_grounding.srv import SetSubplans, SetSubplansResponse
from std_srvs.srv import Trigger, TriggerResponse
from ai_module.src.utils.rr_logger import RRLogger, rotmat_to_quat_xyzw
from cv_bridge import CvBridge


def _stable_hash(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


ANSWER_TYPE = {'find': Marker, 'count': Int32}
ANSWER_TOPIC_NAME = {'find': 'selected_object_marker', 'count': '/numerical_response'}

class EntityType:
    OBJECT = "object"
    DETECTION = "detection"
    IMAGE = "image"
    @classmethod
    def values(cls): return {cls.OBJECT, cls.DETECTION, cls.IMAGE}


make_error_etype = lambda etype: f"entity_type must be in {list(EntityType.values())}, but {etype} was given."


def fmt(v, placeholder="-"):
    if v is None:
        return placeholder
    s = str(v)
    return s if s.strip() else placeholder

def save_path_xy(path_xy: np.ndarray, base_dir="/ws/external/offline_map", name="path_xy"):
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        return
    latest_dir = max(subdirs, key=os.path.getmtime)

    # 저장 파일 이름 (timestamp 기반)
    filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    save_path = os.path.join(latest_dir, filename)
    np.save(save_path, path_xy)
    return save_path


def save_pose(position, orientation, base_dir="/ws/external/offline_map"):
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        return
    latest_dir = max(subdirs, key=os.path.getmtime)

    # 저장 파일 이름 (timestamp 기반)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(latest_dir, f"position_{timestamp}.npy")
    np.save(save_path, position)
    save_path = os.path.join(latest_dir, f"orientation_{timestamp}.npy")
    np.save(save_path, orientation)
    return save_path

class Status(str, Enum):
    WAITING = "Waiting"
    STANDBY = "Standby"
    PROCESSING = "Processing"
    COMPLETED = "Completed"


class PriorityDispatcher:
    """
    - high_queue: (func, args, kwargs, done_event, err_holder)
    - normal_queue: same
    """
    def __init__(self, name="prio-dispatcher", normal_workers=0):
        self.high_queue = queue.Queue()
        self.normal_queue = queue.Queue()
        self._alive = threading.Event(); self._alive.set()

        # 1) 고우선 워커 1개 (필수)
        self.high_thread = threading.Thread(
            target=self._worker_loop, name=f"{name}-HIGH", args=(self.high_queue, True), daemon=True
        )
        self.high_thread.start()

        # 2) (선택) 일반 워커 n개 — 필요시 사용
        self.normal_threads = []
        for i in range(normal_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"{name}-NORM-{i}", args=(self.normal_queue, False), daemon=True
            )
            t.start()
            self.normal_threads.append(t)

        # (선택) 리눅스에서 고우선 스레드 실시간 우선순위 부여
        try:
            self._set_realtime_priority(self.high_thread, priority=80)  # 1~99 (root 필요)
        except Exception:
            pass  # 권한/환경에 따라 실패할 수 있음. 실패해도 기능은 동작.

    def stop(self):
        self._alive.clear()

    def submit_high(self, func, *args, block=False, **kwargs):
        done = threading.Event()
        err = []
        self.high_queue.put((func, args, kwargs, done, err))
        if block:
            done.wait()
            if err:
                raise err[0]
        return done

    def submit_normal(self, func, *args, block=False, **kwargs):
        done = threading.Event()
        err = []
        self.normal_queue.put((func, args, kwargs, done, err))
        if block:
            done.wait()
            if err:
                raise err[0]
        return done

    def _worker_loop(self, q: queue.Queue, is_high: bool):
        log = logging.getLogger(__name__)
        while self._alive.is_set():
            try:
                func, args, kwargs, done, err = q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                func(*args, **kwargs)
            except Exception as e:
                log.error("[%s] task crashed: %s\n%s",
                          threading.current_thread().name, e, traceback.format_exc())
                err.append(e)
            finally:
                done.set()
                q.task_done()

    # ===== Linux 전용: 파이썬 스레드 -> pthread_t 매핑 후 우선순위 부여 =====
    def _set_realtime_priority_safe(priority=80, policy="SCHED_FIFO"):
        """
        현재 호출한 '같은' 스레드의 스케줄링 속성만 안전하게 조절합니다.
        - POSIX에서만 시도, ctypes 시그니처 명시, errno 확인
        - 권한 없으면 경고만 출력하고 정상 진행
        - 잘못된 매개변수로 인한 세그폴트를 방지
        """
        log = logging.getLogger(__name__)
        if os.name != "posix":
            log.info("[rt] non-POSIX, skip")
            return False

        # 정책 상수
        POLICIES = {"SCHED_OTHER": 0, "SCHED_FIFO": 1, "SCHED_RR": 2}
        policy_val = POLICIES.get(policy.upper(), 1)  # default FIFO

        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)

            # typedef struct { int sched_priority; } sched_param;
            class SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]

            # pthread_t pthread_self(void);
            pthread_self = libc.pthread_self
            # pthread_t는 glibc에서 보통 unsigned long
            pthread_self.restype = ctypes.c_ulong
            pthread_self.argtypes = []

            # int pthread_setschedparam(pthread_t, int policy, const struct sched_param*);
            pthread_setschedparam = libc.pthread_setschedparam
            pthread_setschedparam.restype = ctypes.c_int
            pthread_setschedparam.argtypes = [ctypes.c_ulong, ctypes.c_int,
                                              ctypes.POINTER(SchedParam)]

            # 현재 스레드 핸들
            th = pthread_self()

            # 파라미터 구성
            param = SchedParam(int(priority))

            # 호출
            ret = pthread_setschedparam(th, policy_val, ctypes.byref(param))
            if ret != 0:
                err = ctypes.get_errno()  # glibc set errno
                # 일반적으로 CAP_SYS_NICE 없으면 EPERM(1) 납니다.
                log.warning("[rt] pthread_setschedparam failed ret=%d errno=%d (need CAP_SYS_NICE?)", ret, err)
                return False

            log.info("[rt] set realtime priority ok: policy=%s prio=%d", policy, priority)
            return True

        except Exception as e:
            # ctypes 시그니처 오류 등으로 인한 크래시 방지
            log.exception("[rt] exception while setting RT prio: %s", e)
            return False


class BaseVisualGrounder(BaseModel):
# INIT
    def __init__(self, node_name=None, is_real_world=False, logger=None, *args, **kwargs):
        # Load Configuration
        self.config = None
        config_path = rospy.get_param('~config', "/ws/external/ai_module/src/visual_grounding/config/rover_3225.json")
        with open(config_path, "r") as f:
            self.config = config = json.load(f)

        self.rr_logger = RRLogger(name=config['rr_name'])

        super().__init__(logger=logger, *args, **kwargs)

        self.debug = self.config.get("debug", self.debug)
        self.offline_map_dir = os.environ.get(
            "offline_map_dir", self.config.get("OFFLINE_MAP_DIR", "/ws/external/offline_map"))
        self.frame_id = self.config.get("frame_id", "world" if is_real_world else "map")
        self.wo_query = self.config.get(
            "wo_query", rospy.get_param('~wo_query', False) or
                        (os.environ.get("WO_QUERY", "false").lower() == 'true'))

        """ Core """
        self.node_name = node_name if node_name else rospy.get_name()
        self.time_limit = rospy.Duration(self.config.get("time_limit", 600))  # seconds
        self.is_real_world = is_real_world
        
        """ Scheduling """
        self._dispatcher = PriorityDispatcher(name=node_name, normal_workers=0)
   
        """ Aggregation """
        self.aggregated_results_cfg = {
            'min_query': 5,
            'inference_cfg': {
                'method': 'logit_pool',
                'prior': 0.5,
                'keep_top_k': 4,
            }
        }  # TODO: Need to tune   
        
        """ VLM options """
        self.max_llm_concurrency = getattr(self, "max_llm_concurrency", 2)  # 필요시 조절
        self._llm_sema = threading.Semaphore(self.max_llm_concurrency)
        
        self.default_options = {
            'image': {
                'suffix': "",
                'preprocess': 'original',
            },
            'prompt': {
                'rtype': None,      # ['inference', 'validate']
                'action': None,     # ['select_box', 'select_point']
                'atype': None,      # ['object_box_id', 'object_region_id', 'point_id', 'none']
                'hint': None,       # ['reference_object', 'none']
                'is_plural': None,  # [None, True, False]
                'previous_history': "",
            },
            'construct_message': {
                'resize': [1024, 1024],
                'detail': 'high',
            },
            'get_response': {
                'reasoning_effort': 'medium',
                'temperature': 0.0,
            }
        }
        self.default_inference_options = copy.deepcopy(self.default_options)
        self.default_validate_options = copy.deepcopy(self.default_options)
        
        """ Prompt rendering """
        self.prompt_renderer = None
        self.system_instruction_renderer = None        
        
        """ System time """
        self.system_start_ros = None            # from the manager (rospy.Time)
        self.system_start_received = False
        
        """ Initialization """
        self._init_all(*args, **kwargs)

        self.current_agent_message = ""
        self.agent_arrow_len = 0.5

        self.objects_prev = self.objects_curr = []

        self.yolo_sub = rospy.Subscriber("/debug/yolo_image", RosImage, self._yolo_callback, queue_size=1)

    def _init_all(self, *args, **kwargs):
        self._init_vars(*args, **kwargs)
        self._init_services(*args, **kwargs)
        self._init_clients(*args, **kwargs)
        self._init_subscribers(*args, **kwargs)
        self._init_publishers(*args, **kwargs)

    def _init_vars(self, *args, **kwargs) -> None:
        """ High-level state """
        self.updated_resource = False
        self.ready = False
        
        """ Answer """
        self.answer = ""
        self.answer_result = None
        
        """ Subtask """
        self.subtask = None
        
        """ Keyframe selection """
        self.kf_counts = {}  # {kf_id: count}
        
        """ Node active signal """
        self.node_active_signal = False
        
        """ Inference """
        # Inference queue
        self.inference_signal_queue = queue.Queue(maxsize=3)
        self.inference_queue = queue.Queue(maxsize=10)
        self.inference_queue_lock = threading.Lock()
        
        # Aggregated results
        self.agg_results = AggregatedResult(**self.aggregated_results_cfg)
        self.agg_results_lock = threading.Lock()
        
        # Events        
        self.main_running = threading.Event()

        """ Validation """
        # Events        
        self.validation_running = threading.Event()
        
        """ Time """
        self.start_time = None
        self.processing_start_time = 0.0

    def _init_services(self, *args, **kwargs) -> None:
        """ Scene graph """
        self.sg_lock = threading.Lock()
        if not 'logger' in kwargs:
            kwargs.update({'logger': self.logger})
        self.scene_graph_clients = SceneGraphClients(**kwargs)
        
        """ Management services """
        self.srv_subplans_server = rospy.Service(self.node_name + "/set_subplans", SetSubplans, self._set_task_callback)
        self.srv_reset_server = rospy.Service(self.node_name + "/reset", Trigger, self._reset_callback)
        self.srv_status_server = rospy.Service(self.node_name + "/status", Trigger, self._status_callback)
        # self.srv_node_active_signal_client = rospy.ServiceProxy(self.node_name + "/active_signal", Trigger)

    def _init_clients(self, use_ros=True, *args, **kwargs) -> None:
        with open("/ws/external/ai_module/src/config.json", "r") as f:
            config = json.load(f)

        if use_ros:
            api_keys = self._wait_for_keys(param_name="~api_keys", check_hz=5.0) # Wait for the api keys from manager
            clients = []
            for i in range(len(api_keys)):
                clients.append(
                    VisionLlmClient(model_name=config['MODEL_NAME'], api_key=api_keys[i])
                )
            self.clients = clients
        else:
            self.clients = [VisionLlmClient(model_name="gpt-4o", api_key=config['OPENAI_API_KEY0'])]
        self.max_workers = config['MAX_WORKERS']

    def _init_subscribers(self, *args, **kwargs):
        # Subscribe to active nodes topic from manager    
        self.active_nodes_sub = rospy.Subscriber("/active_nodes", String, self._active_nodes_callback, queue_size=1)
        self.logger.loginfo(f"Init subscribers: /active_node")
        # self.rgb_sub = rospy.Subscriber("/UGV4/camera/color/image_raw", Image, self._img_callback, queue_size=1)
        self.rgb_sub = rospy.Subscriber("/UGV4/camera/color/image_raw", RosImage, self._img_callback, queue_size=1)
        self.logger.loginfo(f"Init subscribers: /UGV4/camera/color/image_raw")

        # Subscribe to system start time (latched)
        self.system_start_time_sub = rospy.Subscriber(
            "/system_start_time", Clock, self._system_start_time_callback, queue_size=1
        )
        
        self.force_answer_signal = False
        self.force_answer_sub = rospy.Subscriber(
            "/force_answer", Empty, lambda _msg: setattr(self, "force_answer_signal", True), queue_size=1
        )
    
    def _init_publishers(self, use_ros=True, *args, **kwargs):
        # Visualizer
        if use_ros:
            self.marker_pub = rospy.Publisher("/visual_grounding/markers", Marker, queue_size=50)
            self.marker_pub_orig = rospy.Publisher("/visual_grounding/markers_orig", Marker, queue_size=50)

# RESET
    def _reset_vars(self):
        self._init_vars()

# CALLBACKS
    def _set_task_callback(self, req, candidate_names=[], reference_names=[], *args, **kwargs):
        if self.status == Status.WAITING:
            subtask = req.current_step
            self.start_time = req.start_time
            relation_graph = subtask.entity.relation_graph

            # related_names = []
            # candidate_names = []
            for node in relation_graph.nodes:
                # related_names.append(node.name)
                if node.is_target:
                    if node.name == 'path':
                        for edge in relation_graph.edges:
                            if edge.source_id == node.id:
                                target_ids = edge.target_ids
                                for _node in relation_graph.nodes:
                                    if _node.id in target_ids:
                                        candidate_names.append(_node.name)
                    candidate_names.append(node.name)
                else:
                    reference_names.append(node.name)
            self.related_names = list(set(reference_names + candidate_names))
            self.candidate_names = list(set(candidate_names))
            self.reference_names = list(set(reference_names))
            self.subtask = subtask
            self.agg_results.action = self.action
            self.agg_results.etype = self.etypes[0] # TODO: only one etype is supported for now.

            self.logger.loginfo(f"================================================")
            self.logger.logrich(f"Instruction: \"{req.text_instruction}\"", name='instruction')
            self.logger.logrich(f"Action: \"{subtask.action}\"", name='action')
            self.logger.logrich(f"Target Name: \"{subtask.entity.target_name}\"", name='target_name')
            self.logger.loginfo(f"Candidate names: {self.candidate_names}")
            self.logger.loginfo(f"Reference names: {self.reference_names}")
            self.logger.loginfo(f"Related names: {self.related_names}")
            
            self.rr_log(f"Instruction: \"{req.text_instruction}\"", panel=['default', 'summary/task'])
            self.rr_log(f"Action: \"{subtask.action}\"", panel=['default', 'summary/task'])
            self.rr_log(f"Target Name: \"{subtask.entity.target_name}\"", panel=['default', 'summary/task'])
            self.rr_log(f"Candidate names: {self.candidate_names}", panel=['default', 'summary/task'])
            self.rr_log(f"Reference names: {self.reference_names}", panel=['default', 'summary/task'])

            self.rr_logger.log({"given/instructoin": rr.TextDocument(
                f"## {req.text_instruction}", media_type=rr.MediaType.MARKDOWN)})
            return SetSubplansResponse(success=True, message=self.status)
        else:
            return SetSubplansResponse(success=False, message=self.status)

    def _reset_callback(self, req):
        self._reset_vars()
        
        self.logger.logrich(f"Instruction: ", name='instruction')
        self.logger.logrich(f"Action: ", name='action')
        self.logger.logrich(f"Target Name: ", name='target_name')
        self.logger.logrich(f"Inference: ", name='inference')
        self.logger.log("Visual grounding node has been reset.")
        self.rr_log("Visual grounding node has been reset.", panel=['default', 'summary/task'])
        
        return TriggerResponse(success=True, message="Visual grounding node has been reset.")

    def _status_callback(self, req):
        status = self.status
        # self.logger.log(f"Current status: {status}")
        return TriggerResponse(success=True, message=status.value)

    def _active_nodes_callback(self, msg):
        """
        Callback for active nodes topic.
        Parses the active nodes string and sets node_active_signal based on whether
        current node is in the active list.
        """
        try:
            active_nodes_str = msg.data.strip()
            self.logger.loginfo(f"Received active nodes: {active_nodes_str}")
            
            if not active_nodes_str:
                # Empty string means no active nodes
                self.node_active_signal = False
                self.logger.loginfo(f"Node {self.node_name} is not active (empty active nodes)")
                return
            
            # Parse the active nodes string (format: "node1/node2/node3")
            active_nodes = active_nodes_str.split(',')
            
            # Check if current node is in the active list            
            self.logger.loginfo(f"Active nodes: {active_nodes}")
            
            current_node_name = '/' + self.node_name if self.node_name is not None and '/' != self.node_name[0] else self.node_name
            
            self.logger.loginfo(f"Current node_name: {current_node_name}")
            self.node_active_signal = current_node_name in active_nodes
            
            if self.node_active_signal:
                self.logger.loginfo(f"Node {current_node_name} is ACTIVE")
            else:
                self.logger.loginfo(f"Node {current_node_name} is NOT active")
                
        except Exception as e:
            self.logger.logerr(f"Error in _active_nodes_callback: {e}")
            self.node_active_signal = False

    def _system_start_time_callback(self, msg: Clock):
        self.system_start_ros = msg.clock
        self.system_start_received = True

    def _yolo_callback(self, msg):
        bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.rr_logger.log({'det/rgb': rr.Image(image=rgb)})

    def _img_callback(self, msg):
        try:
            # self.log(f"IMAGE CALLBACK) COMPRESSED?")
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except:
            # self.log(f"IMAGE CALLBACK) RAW?")
            bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.rr_logger.log({'obs/rgb': rr.Image(image=rgb)})
        if self.debug:
            subdirs = [d for d in glob.glob(os.path.join(self.offline_map_dir, "*")) if os.path.isdir(d)]
            if not subdirs:
                return
            latest_dir = max(subdirs, key=os.path.getmtime)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(latest_dir, f"rgb_{timestamp}.jpg")
            cv2.imwrite(save_path, rgb)
# PROPERTIES
    @property
    def confidence_threshold(self):
        action = self.action
        if action == 'find':    return (0.40, 0.70)
        elif action == 'count': return (0.20, 0.70)
        else: return (0.20, 0.50)

    @property
    def sg(self):
        return self.scene_graph_clients.sg

    @property
    def status(self):
        if self.subtask is None:
            return Status.WAITING
        elif self.answer_result is not None:
            return Status.COMPLETED
        elif self.subtask and not self.ready:
            return Status.STANDBY
        elif self.subtask and self.ready:
            return Status.PROCESSING
        else:
            return None

    @property
    def target_name(self):
        return self.subtask.entity.target_name

    @property
    def action(self):
        if hasattr(self, 'subtask'):
            return getattr(self.subtask, 'action', None)
        return None

    @property
    def etypes(self):
        action = self.subtask.action
        if action == 'find':
            return ['object'] # , 'image']
        elif action == 'count':
            return ['object']  # 'object',
        else:
            return ['all']

# MAIN LOOP
    def main_loop(self):
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            # Check if already processing to prevent duplicate execution
            if self.main_running.is_set():
                self.logger.loginfo(f"<main_loop> Main is already running. Let's sleep..")
                rate.sleep()
                continue

            self.logger.loginfo(f"<main_loop> Main is not running. Let's process..")
            self.main_running.set()
            self.spin_once(None)
            self.main_running.clear()
            self.logger.loginfo(f"<main_loop> Main is cleared. Let's sleep..")
            rate.sleep()

    def log_status(self):
        self.rr_logger.log({
            'VG/status': self.status,
            'VG/#inference_queue': len(self.inference_queue.queue),
            'VG/answer': self.answer,
            'VG/MinQuery': self.agg_results.min_query,
        })

    def spin_once(self, event, **kwargs):
        try:
            self.log_status()
        except Exception as e:
            self.logger.logerr(f"<spin_once.1> Error occurs: {e}")
        
        # current_main_state = ""
        # current_main_state += f"Status: {self.status} | "
        # current_main_state += f"#inference_queue={len(self.inference_queue.queue)} | "
        # current_main_state += f"Answer: {self.answer} | "
        # current_main_state += f"MinQuery: {self.agg_results.min_query}"

        status = fmt(self.status)
        nq = fmt(self.agg_results.min_query)
        ans = fmt(self.answer)

        n_infer = len(self.inference_queue.queue)

        current_main_state = (
            f"Status: {status:<10} | "
            f"#inferQ: {n_infer:>4d} | "
            f"Answer: {ans:<10}"
        )
        try:
            self.rr_log(current_main_state, panel=['main', 'summary/status'])
            if self.status == Status.STANDBY:
                self.log_status()
                self.standby()
                self.current_agent_message = f"Let's {self.action} {self.target_name}"
            if self.status == Status.PROCESSING:
                self.log_status()
                # self.logger.logrich(f"Status: {self.status} | #inference_queue={len(self.inference_queue.queue)}", name='status')
                self.process(**kwargs)

            if self.status == Status.COMPLETED:
                self.log_status()
                self.answer_the_question(self.answer_result)
        except Exception as e:
            self.logger.logerr(f"<spin_once.2> Error occurs: {e}")

    def standby(self, **kwargs):
        self.scene_graph_clients.start(
            candidate_names=self.candidate_names,
            reference_names=self.reference_names,
            **kwargs
        ) # TODO: FIX

        action = self.action
        self.answer_pub = rospy.Publisher(ANSWER_TOPIC_NAME.get(action, '/answer'), ANSWER_TYPE.get(action, String), queue_size=1)

        """ LLM Client """
        self.default_inference_options['prompt'].update({'action': action, 'rtype': 'inference'})
        self.default_inference_options['image']['suffix'] = '_annotated_global'
        self.default_validate_options['prompt'].update({'action': action, 'rtype': 'validate'})
        self.default_validate_options['image']['suffix'] = '_annotated_inference'

        self.prompt_renderer = PromptRenderer(description=self.target_name)
        self.system_instruction_renderer = SystemInstructionRenderer()

        self.ready = True
        if self.vis_traversable_points:
            self.traversable_points += self.sg.ground_offset
            self.risky_points += self.sg.ground_offset
            self.rr_logger.log({
                'SG/traversable_points': rr.Points3D(self.traversable_points, colors=[0, 255, 0, 200], radii=0.05)
            }) # non-traversable: red (transparent)
            self.rr_logger.log({
                'SG/non_traversable_points': rr.Points3D(self.risky_points, colors=[255, 0, 0, 80], radii=0.02)
            }) # traversable: green
            self.vis_traversable_points = False

    def log_sg(self, G):
        """
        변경된 노드만 rerun으로 갱신 로깅한다.
        - (level, id)가 과거에 들어왔어도 _attrs가 변하면 fingerprint가 바뀌므로 다시 로깅됨.
        - KEYFRAME의 Pinhole/EncodedImage는 '변경이 있을 때만' 로깅(추가 캐시 포함).
        """
        # --- 캐시 초기화 (처음 한 번만) ---
        if not hasattr(self, "_sg_fp_cache"):
            self._sg_fp_cache = {}      # node_key(level,id) -> fingerprint
        if not hasattr(self, "_sg_logged_pinhole"):
            self._sg_logged_pinhole = set()  # (level,id)
        if not hasattr(self, "_sg_logged_image_path"):
            self._sg_logged_image_path = set()  # image_path

        prefix = "SG/nodes"
        for (level, id), data in G.nodes(data=True):
            entity_path = f"{prefix}/{str(level)}/{id}"
            attrs = data.get('_attrs', {})

            # --- 1) fingerprint 만들기 (가벼운 필드만) ---
            # “이미지 배열” 같은 큰 데이터는 절대 fingerprint에 넣지 마세요.
            if level == str(NodeLevel.OBJECT):
                fp_payload = {
                    "level": level,
                    "id": id,
                    # merge/points 업데이트를 잡기 위해 _attrs 핵심값 포함
                    "name": attrs.get("name"),
                    "centroid": _as_list(attrs.get("centroid")),
                    "extent": _as_list(attrs.get("extent")),
                    "R": _as_list(attrs.get("R")),
                }
            # elif level == str(NodeLevel.KEYFRAME):
            #     # pose/path만으로도 업데이트 감지 가능
            #     fp_payload = {
            #         "level": level,
            #         "id": id,
            #         "pose": _as_list(attrs.get("pose")),
            #         "image_path": attrs.get("image_path", data.get("image_path")),
            #         # K가 바뀌는 경우가 있으면 포함(대부분 고정이라 불필요하지만 안전하게)
            #         "K": _as_list(getattr(self.sg, "rgb_K", None)),
            #     }
            else:
                continue

            fp = _stable_hash(fp_payload)

            # --- 2) 변경이 없으면 로깅 skip ---
            old_fp = self._sg_fp_cache.get((level, id))
            if old_fp == fp:
                continue
            self._sg_fp_cache[(level, id)] = fp

            # --- 3) 변경된 노드만 로깅 ---
            if level == str(NodeLevel.OBJECT):
                centers = np.array([attrs['centroid']], dtype=np.float32) # (1, 3)
                half_sizes = np.array([attrs['extent']], dtype=np.float32) * 0.5
                quaternions = rotmat_to_quat_xyzw(np.array(attrs['R'])).reshape(1, 4)
                palette = self.rr_logger.palette
                palette_idx = int(str(id).split('_')[-1]) % len(palette)
                colors = palette[palette_idx]

                self.rr_logger.log({
                    entity_path: rr.Boxes3D(
                        centers=centers, half_sizes=half_sizes, quaternions=quaternions,
                        colors=colors, labels=f"{attrs['name']}({id})"
                    ),
                })
            # elif level == str(NodeLevel.KEYFRAME):
            #     # Transform은 pose가 바뀌면 갱신되어야 함
            #     pose = np.array(attrs['pose'], dtype=np.float32)
            #     R_b2w, t_b2w = pose[:3, :3], pose[:3, 3]
            #     R_c2w = R_b2w @ self.sg.cam_to_body_R
            #     t_c2w = R_b2w @ self.sg.cam_to_body_t + t_b2w
            #     self.rr_logger.log({
            #         entity_path: rr.Transform3D(
            #             translation=t_c2w, quaternion=rotmat_to_quat_xyzw(R_c2w)
            #         )
            #     })

            #     # Pinhole은 보통 keyframe당 1회면 충분 (K/해상도 고정일 때)
            #     # 단, 위 fingerprint에 K/pose 등이 들어가 있으니 필요하면 매번 다시 찍어도 되지만,
            #     # 비용 절감을 위해 "keyframe당 1회" 캐시로 제한
            #     if (level, id) not in self._sg_logged_pinhole:
            #         # image 해상도: attrs['image']가 있으면 쓰고, 아니면 path에서 1회 로드
            #         img = attrs.get("image", None)
            #         if img is not None:
            #             height, width = img.shape[:2]
            #         else:
            #             image_path = attrs.get("image_path", data.get("image_path"))
            #             if image_path:
            #                 im = cv2.imread(image_path)
            #                 if im is None:
            #                     continue
            #                 height, width = im.shape[:2]
            #             else:
            #                 continue

            #         self.rr_logger.log({
            #             entity_path: rr.Pinhole(
            #                 resolution=[width, height], image_from_camera=self.sg.rgb_K, camera_xyz=rr.ViewCoordinates.RDF,
            #             )
            #         })
            #         self._sg_logged_pinhole.add((level, id))

            #     # EncodedImage는 image_path 기준으로 1회 로깅 (같은 파일이면 재로깅 불필요)
            #     image_path = attrs.get("image_path", data.get("image_path"))
            #     if image_path and (image_path not in self._sg_logged_image_path):
            #         self.rr_logger.log({
            #             entity_path: rr.EncodedImage(path=image_path)
            #         })
            #         self._sg_logged_image_path.add(image_path)


    def update_resource(self, **kwargs):
        # Default styles so they are available even if early steps fail
        styles = {
            # 'reference': {'show': True, 'color': 'blue'},
            'candidate': {'show': True, 'color': 'green'},
        }
        try:
            self.objects_prev = self.objects_curr

            self.scene_graph_clients.update_scene_graph(**kwargs)
            self.log_sg(self.sg.G)

            self.objects_curr = [id for (level, id), e in self.sg.G.nodes(data=True) if level == str(NodeLevel.OBJECT)]
            new_objects = list(set(self.objects_curr) - set(self.objects_prev))
            if len(new_objects) > 0:
                self.current_agent_message = f"New object({', '.join([str(_id) for _id in new_objects])})!"
        except Exception as e:
            self.logger.logerr(f"<update_resource.1> Error occurs: {e}")

        try:
            with self.sg_lock:
                success_path, failed_path = [], []
                visualizer = Visualizer()
                for etype in self.etypes:
                    for (level, id), kf in self.sg.G.nodes(data=True):
                        if level == str(NodeLevel.KEYFRAME):
                            attrs = kf.get('_attrs', {})

                            image = attrs['image'].copy()
                            fname = attrs['fname']
                            detections = attrs['detections']
                            if etype == 'detection':
                                for det in detections:
                                    is_object = det['id'] > 0
                                    if not is_object:
                                        continue

                                    is_candidate = (det['name'] in self.sg.candidate_names)
                                    is_reference = (det['name'] in self.sg.reference_names)
                                    if is_candidate:
                                        style = styles.get('candidate', {'show': False})
                                        if not style['show']:
                                            continue
                                        color = style.get('color', 'green')
                                    elif is_reference:
                                        style = styles.get('reference', {'show': False})
                                        if not style['show']:
                                            continue
                                        color = style.get('color', 'blue')
                                    else:
                                        continue

                                    bbox = det['bbox']
                                    color = visualizer._parse_color(color)
                                    u_min, v_min, u_max, v_max = bbox
                                    u_min, v_min, u_max, v_max = int(u_min), int(v_min), int(u_max), int(v_max)
                                    top_left, bottom_right = (u_min, v_min), (u_max, v_max)

                                    cv2.rectangle(image, top_left, bottom_right, color=color, thickness=2)
                                    cv2.putText(
                                        image, f"{det['id']}", (u_min, v_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                                    )
                            elif etype == 'object':
                                eids = self.sg.pid2eids.get(id, [])
                                for (elevel, eid), entity in self.sg.G.nodes(data=True):
                                    if elevel == str(NodeLevel.OBJECT):
                                        if eid in eids:
                                            bbox = self.sg.project_entity_bbox(entity, kf)
                                            e_attrs = entity.get("_attrs", {})

                                            is_candidate = (e_attrs['name'] in self.sg.candidate_names)
                                            is_reference = (e_attrs['name'] in self.sg.reference_names)

                                            if is_candidate:
                                                style = styles.get('candidate', {'show': False})
                                                if not style['show']:
                                                    continue
                                                color = style.get('color', 'green')
                                            elif is_reference:
                                                style = styles.get('reference', {'show': False})
                                                if not style['show']:
                                                    continue
                                                color = style.get('color', 'blue')
                                            else:
                                                continue

                                            color = visualizer._parse_color(color)
                                            u_min, v_min, u_max, v_max = bbox
                                            u_min, v_min, u_max, v_max = int(u_min), int(v_min), int(u_max), int(v_max)
                                            top_left, bottom_right = (u_min, v_min), (u_max, v_max)

                                            cv2.rectangle(image, top_left, bottom_right, color=color, thickness=2)
                                            cv2.putText(
                                                image, f"{eid}", (u_min, v_min - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                                            )
                            else:
                                raise NotImplementedError("No implementation for other etypes")
                            save_path = self.sg.save_path(etype, fname)
                            success = cv2.imwrite(save_path, image)
                            self.rr_logger.log({f"SG/nodes/NodeLevel.KEYFRAME/{id}": rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))})
                            if success:
                                success_path.append(save_path)
                            else:
                                failed_path.append(save_path)
                    if len(success_path) > 0:
                        txt = '- \n'.join(success_path)
                        self.rr_logger.log({
                            'VG/log': f"Saved images: {txt}"})
                    if len(failed_path) > 0:
                        txt = '- \n'.join(failed_path)
                        self.rr_logger.log({
                            'VG/log': f"Failed to save images: {txt}"}, level='warn')
        except Exception as e:
            self.logger.logerr(f"<update_resource.1> Error occurs: {e}")
        
        self.updated_resource = True

    def select_keyframes(
            self, entity_type='object', w_cov=1.0, w_area=0.1, w_rel=0.5, alpha=0.7, target_eids=None,
            min_kfs=None, max_kfs=10, iter_margin=5, *args, **kwargs
    ):
        with self.sg_lock:
            sg = self.sg
        etype = 'all' if entity_type == 'image' else entity_type

        # --- candidate ids 준비 ---
        related_eids = set([entity['id'][1] for entity in sg.get_related_entities(etype)])
        try:
            if target_eids is None:
                target_eids = set(sg.get_related_entities(etype).ids) # TODO: Need to check
            else:
                target_eids = set(target_eids)
            if not target_eids:
                self.logger.logwarn(f"<select_keyframes.1> target_eids is empty.")
                return sg.keyframes.get([])
            self.log(f"<select_keyframes.1> target_eids: {target_eids}")
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.1> Error occurs: {e}")

        # --- min/max 보정 ---
        max_kfs = max(0, int(max_kfs))
        min_kfs = 0 if min_kfs is None else max(0, min(int(min_kfs), max_kfs))

        kfs = sg.keyframes
        # ---------- 주어진 target entities를 포함하는 keyframes를 구성: kfs_with_targets, pid2target_eids ----------
        try:
            # TODO: Need to check the below code
            eid2pids = sg.eid2pids
            pid2eids = sg.pid2eids

            num_places = len(kfs)
            num_target_entities = len(target_eids)
            use_pid2eids = num_target_entities > max(1, num_places // 8)

            kfs_with_targets = {}
            pid2target_eids = defaultdict(set)
            available_pids = set([kf['id'][1] for kf in kfs])
            if not use_pid2eids:
                for target_eid in target_eids:
                    for pid_with_target in eid2pids.get(target_eid, ()):
                        if pid_with_target not in available_pids:
                            self.log(f"<select_keyframes.3> Warning occurs: PID({pid_with_target}) is not in available PIDs: {available_pids}", level='warn')
                            continue
                        if pid_with_target not in kfs_with_targets:
                            kfs_with_targets[pid_with_target] = kfs[pid_with_target]
                        pid2target_eids[pid_with_target].add(target_eid)
            else:
                for pid in available_pids:
                    eids_here = set(pid2eids.get(pid, ()))
                    if not eids_here:
                        self.logger.logwarn(f"<select_keyframes.3> Warning occurs: PID({pid}) has no any EIDs.")
                        continue
                    target_eids_here = eids_here & target_eids
                    if not target_eids_here:
                        continue
                    kfs_with_targets[pid] = kfs[pid]
                    pid2target_eids[pid] = target_eids_here
            self.log(f"<select_keyframes.2> Target EIDs per each kf: {', '.join([f'{k}: {v}' for k, v  in pid2target_eids.items()])}")

            if len(kfs_with_targets) < min_kfs:
                self.log(f"<select_keyframes.2> #kfs_with_targets={len(kfs_with_targets)} < min_kfs={min_kfs}. Skip.", level='warn')
                return sg.keyframes  # Early Stop
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.2> Error occurs: {e}")

        # Cache: Area
        target_entities = sg.get_related_entities() # target_entities = sg.get_related_entities(etype).get(target_eids)
        per_obj_area = defaultdict(dict)  # {pid: {eid: area}, ...}

        def _entity_area(kf, pid, eid, sg):
            d = per_obj_area[pid]
            if eid in d:
                return d[eid]

            tgt_ent = [entity for entity in target_entities if entity['id'][1] == eid][0]
            if tgt_ent is None:
                d[eid] = 0.0
                return 0.0

            u_min, v_min, u_max, v_max = sg.project_entity_bbox(tgt_ent, kf)

            # top_left, bottom_right = (u_min, v_min), (u_max, v_max)
            # img_vis = kf_attrs['image'].copy()
            # cv2.rectangle(img_vis, top_left, bottom_right, color=(0, 255, 0), thickness=2)
            # cv2.imwrite("/ws/external/vis/tmp.jpg", img_vis)

            area = (u_max - u_min) * (v_max - v_min)
            d[eid] = float(area)
            return d[eid]

        # Select N keyframes which contains target entities (N < max_kfs)
        try:
            selected_pids = []
            uncovered_target_eids = set(target_eids)
            iter_cap = max(1, min(len(kfs_with_targets), max_kfs) + iter_margin)
            iter_cnt = 0
            while uncovered_target_eids and kfs_with_targets and len(selected_pids) < max_kfs:
                iter_cnt += 1
                if iter_cnt > iter_cap:
                    self.log(f"<select_keyframes.4> iter_cap reached. Bail out.", level='warn')
                    break

                num_uncovered_tgts = len(uncovered_target_eids)
                max_area = 0.0
                max_rel = 0
                tmp_stats = {}  # {pid: (c, a, covered_eids_now), ...}
                for pid, kf in kfs_with_targets.items():
                    covered_eids_now = pid2target_eids.get(pid, ()) & uncovered_target_eids
                    if not covered_eids_now:
                        continue
                    c = len(covered_eids_now) / num_uncovered_tgts
                    a = 0.0
                    for eid in covered_eids_now:
                        a += _entity_area(kf, pid, eid, sg=self.sg)
                    if a > max_area:
                        max_area = a
                        
                    eids_here = pid2eids.get(pid, [])
                    related_eids_here = set(eids_here) & set(related_eids)
                    rel_cnt = len(related_eids_here)
                    max_rel = max(max_rel, rel_cnt)
                    tmp_stats[pid] = (c, a, rel_cnt, covered_eids_now)

                if not tmp_stats:
                    self.log(f"<select_keyframes.4> tmp_stats is None", level='warn')
                    break

                best_pid, best_score = None, float("-inf")
                for pid, (c, a, rel_cnt, _) in tmp_stats.items():
                    base = (
                        w_cov * c
                        + (w_area * (a / max_area) if max_area > 0 else 0.0)
                        + (w_rel * (rel_cnt / max_rel) if max_rel > 0 else 0)
                    )
                    cnt = self.kf_counts.get(pid, 0)  # TODO
                    seen = 1.0 / (1.0 + alpha * cnt)
                    score = base * seen
                    if score > best_score:
                        best_score, best_pid = score, pid

                    if best_pid is None:
                        self.log(f"<select_keyframes.4> best_pid is None", level='warn')
                        break

                selected_pids.append(best_pid)
                _, _, _, covered_eids_best = tmp_stats[best_pid]
                uncovered_target_eids.difference_update(covered_eids_best)
                self.kf_counts[best_pid] = self.kf_counts.get(best_pid, 0) + 1
                kfs_with_targets.pop(best_pid, None)
                pid2target_eids.pop(best_pid, None)

            self.log(f"<select_keyframes.3> selected_pids: {selected_pids}")
        except Exception as e:
            self.log(f"<select_keyframes.3> Error occurs: {e}")

        # Add M keyframes which contains target entities (min_kfs < N+M)
        try:
            while (len(selected_pids) < min_kfs) and kfs_with_targets:
                best_pid = max(
                    kfs_with_targets.keys(),
                    key=lambda pid: (-self.kf_counts.get(pid, 0), len(pid2target_eids.get(pid, ())))
                )
                selected_pids.append(best_pid)
                covered_target_eids = pid2target_eids.get(best_pid, ())
                uncovered_target_eids.difference_update(covered_target_eids)
                self.kf_counts[best_pid] = self.kf_counts.get(best_pid, 0) + 1
                kfs_with_targets.pop(best_pid, None)
                pid2target_eids.pop(best_pid, None)
            self.log(f"<select_keyframes.4> selected_pids: {selected_pids}")
        except Exception as e:
            self.log(f"<select_keyframes.4> Error occurs: {e}", level='error')

        selected_kfs = []
        processed_pids = []
        for pid in selected_pids:
            for kf in sg.keyframes:
                if kf['id'][1] == pid:
                    selected_kfs.append(kf)
                    processed_pids.append(pid)
                    break
        if len(selected_kfs) != len(selected_pids):
            self.log("<select_keyframes.4> Warn: len(selected_kfs) != len(selected_pids)", level='warn')

        return selected_kfs

    def process(self, **kwargs):
        self.update_resource(**kwargs)

        # Queue-based inference system
        try:
            signal = self.inference_signal_queue.get_nowait()
            gid, eids = signal.get('group_id'), signal.get('entity_ids')
            is_pending_process = False
            self.logger.loginfo(f"<process.1> New signal is given: {signal}")
        except queue.Empty:
            gids = self.inference_results.pending_gids()  # min_query 기준
            if len(gids) == 0:
                self.logger.loginfo(f"<process.1> There is no pending_gids")
                return
            gid = gids[0]
            eids = self.inference_results.group_eids.get(gid, set())
            is_pending_process = True
            self.logger.loginfo(f"<process.1> There are pending_gids: {gid} (eids:{eids})")

        try:
            with self.sg_lock:
                candidate_object_ids = self.sg.get_candidate_entities('object').ids
                candidate_detection_ids = self.sg.get_candidate_entities('detection').ids
            self.logger.loginfo(f"<process.2> Candidate objects & detections: {candidate_object_ids} + {candidate_detection_ids}")
        except Exception as e:
            self.logger.logerr(f"<process.2> Error occurs: {e}")

        # --- 지금 필요한 개수 계산 (min_query 충족용) ---
        try:
            need = None
            if is_pending_process:
                eff = self.inference_results.effective_count(gid)
                need = max(0, self.inference_results.min_query - eff)
                if need == 0:
                    self.logger.loginfo(f"<process.3> Group({gid}) already satistied (effective={eff})")
                    return
            self.logger.loginfo(f"<process.3> need = {need}")
        except Exception as e:
            self.logger.logerr(f"<process.3> Error occurs: {e}")

        try:
            num_kfs = {}
            data = []
            for etype in self.etypes:
                # 필요 시 max_query를 need로 clamp
                cfg = dict(self.inference_results_cfg)
                if is_pending_process and need is not None:
                    cfg['max_query'] = min(cfg.get('max_query', need), need)
                    # min_query도 과도 예약 방지 용도로 clamp
                    cfg['min_query'] = min(cfg.get('min_query', 0), cfg['max_query'])

                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, **self.inference_results_cfg)
                if etype == 'image':
                    data += [{'keyframes': kfs, 'etype': 'image', 'gid': gid, 'eids': eids} for kfs in keyframes.to_list()]
                else:
                    data += [{'keyframes': keyframes, 'etype': etype, 'gid': gid, 'eids': eids}]
                num_kfs[etype] = len(keyframes)
            self.logger.loginfo(f"<process.4> Selected KFs: {num_kfs}\n"
                                f"  > eids: {eids}")
        except Exception as e:
            self.logger.logerr(f"<process.4> Error occurs: {e}")

        try:
            # 필요 개수보다 많이 뽑혔으면 잘라내기(과도 예약 방지)
            num_data = len(data)
            if is_pending_process and need is not None and len(data) > need:
                data = data[:need]
            self.logger.loginfo(f"<process.5> clipped data: {num_data} -> {len(data)}")
        except Exception as e:
            self.logger.logerr(f"<process.5> Error occurs: {e}")

        try:
            if len(keyframes) == 0:
                self.logger.loginfo(f"<process.6> No keyframes to put: #kfs={len(keyframes)}")
            elif len(data) > 0:
                with self.inference_queue_lock:
                    self.inference_queue.put(data)
                # --- 예약 수 반영 ---
                self.inference_results.schedule(gid, len(data), data=data)
                self.logger.loginfo(f"<process.6> Put data to inference_queue: {data}")
            else:
                self.logger.loginfo(f"<process.6> No data to put: {data}")
        except Exception as e:
            self.logger.logerr(f"<process.6> Error occurs: {e}")

        self.logger.logrich(f"Selected keyframe: #object({num_kfs.get('object', 0)}), #detection({num_kfs.get('detection', 0)}), #image({num_kfs.get('image', 0)})", name="selected_keyframe")

    def _answer_impl(self, answer, block=False):
        if answer is None:
            self.logger.logwarn(f"<answer_the_question> answer is None")
            return

        if isinstance(answer, Answer):
            self.logger.loginfo(f"<answer_the_question.1> Answer is Answer type")
            action = self.action
            self.logger.loginfo(f"<answer_the_question.2> Action is {action}")
            answer_msg = answer.get_answer_msg(action)
            self.logger.loginfo(f"<answer_the_question.3> Answer message is {answer_msg}")

            self.answer_pub.publish(answer_msg)
            self.answer = answer_msg

            # Publish all markers
            self.logger.loginfo(f"<answer_the_question.4> Get answer markers")
            markers = answer.get_answer_vis_msg(action)
            for marker in markers:
                self.marker_pub.publish(marker)
            self.logger.logrich(f"Answer: {answer.get_answer(action)}", name='answer')
        else:
            try:
                if self.action in ['find']:
                    self.logger.loginfo(f"<answer_the_question.5.1> self.action: {self.action}")
                    color = (0.0, 0.0, 1.0, 1.0)
                    eid = int(answer)
                    self.logger.loginfo(f"<answer_the_question.5.2> Get single EID={eid}...")
                    # answer_entity = self.sg.entities.get_single(eid)
                    # self.logger.loginfo(f"<answer_the_question.5.2> answer_entity: {answer_entity}")
                    #
                    # # Refinement
                    # self.logger.loginfo(f"<answer_the_question.5.3> Refinement...")
                    # eid2pids = self.sg.keyframes.entity_id2place_ids
                    # pids_answer = eid2pids[answer_entity.id]
                    # kfs_answer = self.sg.keyframes.get(pids_answer)
                    # initial_bbox_3d = answer_entity.corners_3d
                    #
                    # self.logger.loginfo(f"<answer_the_question.5.4> Let's minimize")
                    # refined_result = minimize(
                    #     refine_bbox,
                    #     initial_bbox_3d,
                    #     args=(answer_entity, kfs_answer,),
                    #     method='Nelder-Mead',
                    #     options={'disp': True}
                    # )
                    # refined_point_3d = refined_result.x.reshape(-1, 3)
                    #
                    # self.logger.loginfo(f"<answer_the_question.5.5> point_3d_to_marker:\n"
                    #                     f"  > initial_bbox_3d: {initial_bbox_3d}\n"
                    #                     f"  > refined_point_3d: {refined_point_3d}\n")
                    # marker_orig = point_3d_to_marker(initial_bbox_3d, eid, color=color, style='box')
                    # marker = point_3d_to_marker(refined_point_3d, eid, color=color, style='box')
                    # answer_msg = point_3d_to_marker(refined_point_3d, eid, color=color, style='cube')

                    # self.logger.loginfo(f"<answer_the_question.5.6> Publish Answer...")
                    # self.answer_pub.publish(answer_msg)
                    # self.logger.loginfo(f"<answer_the_question.5.6> Publish...")
                    # self.marker_pub.publish(marker)
                    # self.marker_pub_orig.publish(marker_orig)
                    # self.logger.loginfo(f"<answer_the_question.5.6> Publish... Done")
                elif self.action in ['count']:
                    count = int(answer)
                    answer_msg = Int32(count)
                else:
                    raise NotImplementedError
                self.logger.loginfo(f"<answer_the_question.5.7> Publish answer...")
                self.answer_pub.publish(answer_msg)
                self.answer = answer_msg
            except Exception as e:
                self.logger.logerr(f"<answer_the_question.5> Error occurs: {e}")
                self.answer = answer

        self.logger.loginfo(f"self.answer: {self.answer}")
        self.logger.logrich(f"Answer: {answer}", name='answer')
        self.current_agent_message = f"Answer: {answer}"
        # self.rr_logger.rr.flush()
        # time.sleep(0.5)
        # sys.exit(0)

    def answer_the_question(self, answer, block=False):
        return self._dispatcher.submit_high(self._answer_impl, answer, block=block)

# INFERENCE LOOP
    def inference_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            try:
                # Time check
                if self.start_time:
                    now = rospy.Time.now()
                    try:
                        elapsed = now - self.start_time
                        remaining_time = (self.start_time + self.time_limit) - now
                    except:
                        elapsed = now.secs - self.start_time
                        remaining_time = (self.start_time + self.time_limit.secs) - now.secs
                else:
                    remaining_time = rospy.Duration(60)
                    elapsed = rospy.Time.now()
                try:
                    time_txt = f"{int(elapsed.to_sec())}/{int(self.time_limit.to_sec())}"
                except:
                    time_txt = f"{int(elapsed)}/{int(self.time_limit.secs)}"
                self.log(f"<inference_loop.1> Time: {time_txt} (sec)")
                self.rr_log(f"<inference_loop.1> Time: {time_txt} (sec)", panel='inference')
                self.rr_logger.log({'VG/Time': f"{time_txt} (sec)"})
                
                # Current status check
                if self.status != Status.PROCESSING:
                    rate.sleep()
                    self.logger.loginfo(f"<inference_loop.1> Let's sleep...")
                    continue
                self.log(f"<inference_loop.1> Let's inference!!")
                self.rr_log(f"<inference_loop.1> Let's inference!!", panel='inference')
            except Exception as e:
                self.logger.logerr(f"<inference_loop.1> Error occurs: {e}")

            try:
                with self.agg_results_lock:
                    agg_results = self.agg_results.snapshot()
                self.log(f"<inference_loop.2> AggResults: {self.agg_results}")
                self.rr_log(f"<inference_loop.2> AggResults: {self.agg_results}", panel='inference')
                self.rr_log(f"AggResults: {self.agg_results}", panel='summary/status')
            except Exception as e:
                self.log(f"<inference_loop.2> Error occurs: {e}", level='error')
                self.rr_log(f"<inference_loop.2> Error occurs: {e}", panel='inference', level='error')

            # Ready to answer?
            try:
                (thres_low, thres_high) = self.confidence_threshold

                best_confidence = agg_results.get('best_confidence')
                enough_observation = (self.exploration_status == 'no_frontier')
                all_inference_done = (self.inference_queue.qsize() == 0) and (self.inference_signal_queue.qsize() == 0) # TODO: If inference becomes asynchronous, this logic must be updated.
                try:
                    enough_time_elapsed = (elapsed >= rospy.Duration(2 * 60))
                except:
                    enough_time_elapsed = (elapsed >= 2 * 60)
                has_any_result = (self.agg_results.best_answer is not None)
                if self.action == 'count':
                    if int(self.agg_results.best_answer) == 0:
                        has_any_result = False
                
                # Determine if ready to answer
                ## Option1: High confidence & Enough time elapsed
                ## Option2: Enough observation & All inference is done & Has any result
                ## Option3: Time is almost up
                try:
                    ready_to_answer = False
                    if self.action == 'count':
                        ready_to_answer = (((best_confidence > thres_high and enough_time_elapsed)
                                            or (enough_observation and all_inference_done and has_any_result))
                                        or (remaining_time <= rospy.Duration(30)))  # (sec)
                        ready_to_answer = ready_to_answer and enough_observation
                    else:
                        ready_to_answer = (((best_confidence > thres_high and enough_time_elapsed)
                                            or (enough_observation and all_inference_done and has_any_result))
                                        or (remaining_time <= rospy.Duration(30)))  # (sec)
                    self.logger.loginfo(f"self.action: {self.action},  enough_observation: {enough_observation}")

                    self.log(f"<inference_loop.3.2> Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}")
                    self.rr_log(f"<inference_loop.3.2> Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}", panel='inference')
                    self.rr_log(f"Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}", panel='summary/status')
                except Exception as e:
                    ready_to_answer = False
                    if self.action == 'count':
                        ready_to_answer = (((best_confidence > thres_high and enough_time_elapsed)
                                        or (enough_observation and all_inference_done and has_any_result))
                                       or (remaining_time <= 30))  # (sec)
                        ready_to_answer = ready_to_answer and enough_observation
                    else:
                        ready_to_answer = (((best_confidence > thres_high and enough_time_elapsed)
                                        or (enough_observation and all_inference_done and has_any_result))
                                       or (remaining_time <= 30))  # (sec)
                    self.logger.loginfo(f"self.action: {self.action},  enough_observation: {enough_observation}")
                    self.log(f"<inference_loop.3.2> Time: {int(elapsed)}/{int(self.time_limit.secs)} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}")
                    self.rr_log(f"<inference_loop.3.2> Time: {int(elapsed)}/{int(self.time_limit.secs)} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}", panel='inference')
                    self.rr_log(f"Time: {int(elapsed)}/{int(self.time_limit.secs)} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status} | Inference Status: {all_inference_done}", panel='summary/status')

                if self.force_answer_signal:
                    ready_to_answer = True
                    self.log(f"<inference_loop.3.2> Force answer signal is given.")
                    self.rr_log(f"<inference_loop.3.2> Force answer signal is given.", panel='inference')
                    self.force_answer_signal = False
                
                if ready_to_answer:
                    self.answer_result = self.agg_results.best_answer  # TODO
                    self.answer_the_question(self.answer_result)
                    self.rr_log(f"Answer: {self.answer_result}", panel=['default', 'summary/task'])
                    self.rr_logger.log({"answer/answer": rr.TextDocument(
                        f"## {self.answer_result}", media_type=rr.MediaType.MARKDOWN)})

                    if best_confidence > thres_high:
                        self.log(f"<inference_loop.3.2> Answer the final result. Confidence: {best_confidence} > {thres_high}.")
                        self.rr_log(f"<inference_loop.3.2> Answer the final result. Confidence: {best_confidence} > {thres_high}.", panel='inference')
                        self.rr_log(f"Answer the final result. Confidence: {best_confidence} > {thres_high}.", panel='summary/status')
                        self.current_agent_message = f"Answer {self.answer_result}, with high confidence."
                    elif enough_observation and all_inference_done and has_any_result:
                        self.log(f"<inference_loop.3.2> Answer the final result. Enough observation and all inference done.")
                        self.rr_log(f"<inference_loop.3.2> Answer the final result. Enough observation and all inference done.", panel='inference')
                        self.rr_log(f"Answer the final result. Enough observation and all inference done.", panel='summary/status')
                        self.current_agent_message = f"Answer {self.answer_result}, with sufficient observations."
                    else:
                        self.log(f"<inference_loop.3.2> Answer the final result. Time is almost up {remaining_time.to_sec()} sec left.")
                        self.rr_log(f"<inference_loop.3.2> Answer the final result. Time is almost up {remaining_time.to_sec()} sec left.", panel='inference')
                        self.rr_log(f"Answer the final result. Time is almost up {remaining_time.to_sec()} sec left.", panel='summary/status')
                        self.current_agent_message = f"No time. The answer is {self.answer_result}."
                    return
                else:                    
                    if best_confidence <= thres_high:
                        self.log(f"<inference_loop.3.2> Let's inference. Confidence: {best_confidence} <= {thres_high}.")
                        self.rr_log(f"<inference_loop.3.2> Let's inference. Confidence: {best_confidence} <= {thres_high}.", panel='inference')
                        self.rr_log(f"Let's inference. Confidence: {best_confidence} <= {thres_high}.", panel='summary/status')
                    elif not enough_time_elapsed:
                        self.log(f"<inference_loop.3.2> Let's inference. Not enough time elapsed yet. {int(elapsed.to_sec())} sec passed.")
                        self.rr_log(f"<inference_loop.3.2> Let's inference. Not enough time elapsed yet. {int(elapsed.to_sec())} sec passed.", panel='inference')
                        self.rr_log(f"Let's inference. Not enough time elapsed yet. {int(elapsed.to_sec())} sec passed.", panel='summary/status')
                    else:
                        self.log(f"<inference_loop.3.2> Let's inference.")
                        self.rr_log(f"<inference_loop.3.2> Let's inference.", panel='inference')
                        self.rr_log(f"Let's inference.", panel='summary/status')
            except Exception as e:
                self.log(f"<inference_loop.3.1&2> Error occurs: {e}", level='error')
                self.rr_log(f"<inference_loop.3.1&2> Error occurs: {e}", panel='inference', level='error')

            # Inference
            try:
                with self.inference_queue_lock:
                    inference_queue_size = self.inference_queue.qsize()
                for _ in range(inference_queue_size):
                    try:
                        if self.wo_query:
                            break
                        with self.inference_queue_lock:
                            jobs = self.inference_queue.get_nowait()
                    except queue.Empty:
                        break
                    for item, result in self.run_parallel(self.inference, jobs, max_workers=self.max_workers):
                        gid = item.get('gid')
                        if result is None:
                            self.log(f"<inference_loop.4.2.{_}> result is None.")
                            self.rr_log(f"<inference_loop.4.2.{_}> result is None.", panel='inference')
                            continue

                        try:
                            etype = result.get('entity_type')
                            target_ids = result.get('target_ids', [])
                            answers = []
                            if self.action == 'count':
                                count = len(target_ids)
                                if count == 0:
                                    answer = None
                                    answers.append(answer)
                                if self.etypes == ['object']:
                                    target_entities = []

                                    target_ids = [int(t) for t in target_ids]
                                    for (level, id), data in self.sg.G.nodes(data=True):
                                        if level == str(NodeLevel.OBJECT):
                                            if id in target_ids:
                                                target_entities.append(data)

                                    self.log(f"<inference_loop.4.3.{_}> len(target_entities): {len(target_entities)}")

                                    result['data'].update({'pid2eids': self.sg.pid2eids})
                                    for target_entity in target_entities:
                                        answer = Answer(object=target_entity, data=result['data'])
                                        self.log(f"<inference_loop.4.3.{_}> target_entity: {target_entity}")
                                        self.rr_log(f"<inference_loop.4.3.{_}> target_entity: {target_entity}", panel='inference')
                                        answers.append(answer)

                                else:
                                    answer = Answer(count=count, data=result['data'])
                                    answers.append(answer)
                                self.log(f"<inference_loop.4.3.{_}> Answer(count={count})")
                                self.rr_log(f"<inference_loop.4.3.{_}> Answer(count={count})", panel='inference')
                                self.rr_log(f"Answer(count={count})", panel='details')
                            elif self.action == 'find':
                                if len(target_ids) > 1:
                                    self.log(f"<inference_loop.4.3.{_}> #target_ids={len(target_ids)} > 1", level='warn')
                                    self.rr_log(f"<inference_loop.4.3.{_}> #target_ids={len(target_ids)} > 1", panel='inference', level='warn')
                                    self.rr_log(f"#target_ids={len(target_ids)} > 1", panel='details', level='warn')
                                elif len(target_ids) == 0:
                                    answer = None
                                    self.log(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: X")
                                    self.rr_log(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: X", panel='inference')
                                    self.rr_log(f"Answer: {answer};  target_entity: X", panel='details')
                                else:
                                    target_id = int(target_ids[0])
                                    # candidate_entities = self.sg.get_candidate_entities('all')
                                    # target_entity = self.sg.entities.get_single(target_id)
                                    target_entity = []
                                    for (level, id), data in self.sg.G.nodes(data=True):
                                        if level == str(NodeLevel.OBJECT):
                                            if id == target_id:
                                                target_entity.append(data)
                                    result['data'].update({'pid2eids': self.sg.pid2eids})
                                    answer = Answer(object=target_entity[0], data=result['data'])
                                    self.log(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: {[e['id'] for e in target_entity]}")
                                    self.rr_log(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: {[e['id'] for e in target_entity]}", panel='inference')
                                    self.rr_log(f"Answer: {answer};  target_entity: {target_entity}", panel='details')
                                    answers.append(answer)
                            else:
                                raise NotImplementedError(f"action must be in ['count'], but {self.action} was given.")
                        except Exception as e:
                            self.log(f"<inference_loop.4.3.{_}> Error occurs: {e}", level='error')
                            self.rr_log(f"<inference_loop.4.3.{_}> Error occurs: {e}", panel='inference', level='error')

                        try:
                            for answer in answers:
                                if answer is not None:
                                    self.agg_results.update(gid=gid, answer=answer, confidence=get_confidence(etype))
                                    self.log(f"<inference_loop.4.4.{_}> Update agg_results <- {answer}")
                                else:
                                    self.log(f"<inference_loop.4.4.{_}> No updated agg_results")
                            # if answer is not None:
                            #     self.agg_results.update(gid=gid, answer=answer, confidence=get_confidence(etype))
                            #     self.log(f"<inference_loop.4.4.{_}> Update agg_results <- {answer}")
                                eids = self.agg_results.results_by_entity.results.keys()
                                best_confidence = self.agg_results.best_confidence
                                eids_with_best_confidence = [eid for eid, data in self.agg_results.results_by_entity.results.items()
                                                             if data['confidence'] == best_confidence]
                                if self.action == 'find':
                                    for (level, id), data in self.sg.G.nodes(data=True):
                                        if level == str(NodeLevel.OBJECT):
                                            if id in eids:
                                                entity_path = f"SG/nodes/{str(level)}/{id}"
                                                attrs = data.get('_attrs', {})

                                                agg_result = self.agg_results.results_by_entity.results[id]
                                                confidence = agg_result['confidence']
                                                count = agg_result['count']
                                                order = agg_result['order']

                                                centers = np.array([attrs['centroid']], dtype=np.float32)  # (1, 3)
                                                half_sizes = np.array([attrs['extent']], dtype=np.float32) * 0.5
                                                quaternions = rotmat_to_quat_xyzw(np.array(attrs['R'])).reshape(1, 4)
                                                colors = self.rr_logger.palette[int(str(id).split('_')[-1])]
                                                self.rr_logger.log({
                                                    entity_path: rr.AnyValues(confidence=f"{confidence:.2f}", count=count),
                                                })
                                                label_pos = copy.deepcopy(centers)
                                                label_pos[:, 2] = 1.5
                                                self.rr_logger.log({
                                                    f"{entity_path}/confidence/anchor": rr.Points3D(
                                                        positions=label_pos,
                                                        radii=0.08 if id in eids_with_best_confidence else 0.04,
                                                        colors=colors,
                                                    )
                                                })
                                                self.rr_logger.log({
                                                    f"{entity_path}/confidence/label": rr.Points3D(
                                                        positions=label_pos,
                                                        radii=0.001,
                                                        labels=[f"{confidence:.2f}"],
                                                        colors=(255, 255, 255),  # white
                                                    )
                                                })
                                                strips = [np.stack([c, l], axis=0) for c, l in zip(centers, label_pos)]
                                                rr.log(
                                                    f"{entity_path}/entity-confidence",
                                                    rr.LineStrips3D(
                                                        strips=strips,
                                                        radii=0.01,
                                                        colors=colors,
                                                    )
                                                )
                                elif self.action == 'count':
                                    eids = self.agg_results.results_by_entity_count.results.keys()

                                    for (level, id), data in self.sg.G.nodes(data=True):
                                        if level != str(NodeLevel.OBJECT):
                                            continue
                                        if id not in eids:
                                            continue

                                        entity_path = f"SG/nodes/{str(level)}/{id}"
                                        attrs = data.get("_attrs", {})

                                        # agg
                                        agg_result = self.agg_results.results_by_entity_count.results[id]
                                        confidence = float(agg_result.get("confidence", 0.0))
                                        count = int(agg_result.get("count", 0))
                                        order = agg_result.get("order", None)

                                        centers = np.array([attrs["centroid"]], dtype=np.float32)  # (1, 3)
                                        half_sizes = np.array([attrs["extent"]], dtype=np.float32) * 0.5
                                        quaternions = rotmat_to_quat_xyzw(np.array(attrs["R"])).reshape(1, 4)

                                        colors = self.rr_logger.palette[int(str(id).split("_")[-1])]

                                        # (A) 메타 값 기록
                                        self.rr_logger.log({
                                            entity_path: rr.AnyValues(
                                                confidence=f"{confidence:.2f}",
                                                count=count,
                                                order=order if order is not None else -1,
                                            ),
                                        })
                                        
                                        # (B) 라벨 위치(기존 유지)
                                        label_pos = copy.deepcopy(centers)        # (1,3)
                                        label_pos[:, 2] = 1.5
                                        p = label_pos[0]                          # (3,)

                                        # (C) 표시 크기: count 기반(원/엑스의 "반지름")
                                        mark_r = 0.10 + 0.04 * np.log1p(count)    # 원하는대로 키워도 됨
                                        mark_r = float(np.clip(mark_r, 0.08, 0.25))

                                        # (D) O/X 결정 + 색
                                        is_ok = (confidence >= 0.5)
                                        mark_char = "O" if is_ok else "X"
                                        mark_color = np.array([0, 255, 0] if is_ok else [255, 0, 0], dtype=np.uint8)  # O=초록, X=빨강

                                        # (E) anchor 점(원하는 경우 유지): UI 픽셀 크기로 키우려면 radii를 음수로
                                        #     (음수 radii = UI points; zoom해도 크기 고정) :contentReference[oaicite:1]{index=1}
                                        self.rr_logger.log({
                                            f"{entity_path}/confidence/anchor": rr.Points3D(
                                                positions=label_pos,
                                                radii=-10.0,              # <-- UI points (픽셀 느낌). 더 키우려면 -14, -18 등
                                                colors=mark_color,
                                            )
                                        })

                                        # (F) O/X를 "텍스트"가 아니라 "선"으로 그림 => 크기 완전 제어 가능
                                        if mark_char == "O":
                                            # 원: XY 평면에 원형 폴리라인(원 개수 늘리면 더 매끈)
                                            n = 32
                                            th = np.linspace(0, 2*np.pi, n, endpoint=True)
                                            circle = np.stack([
                                                p[0] + mark_r * np.cos(th),
                                                p[1] + mark_r * np.sin(th),
                                                np.full_like(th, p[2]),
                                            ], axis=1).astype(np.float32)

                                            self.rr_logger.log({
                                                f"{entity_path}/confidence/mark": rr.LineStrips3D(
                                                    strips=[circle],
                                                    radii=0.015,           # 선 굵기(원 두께)
                                                    colors=mark_color,
                                                )
                                            })

                                        else:  # "X"
                                            # X: 두 개의 대각선
                                            a = np.array([ mark_r,  mark_r, 0.0], dtype=np.float32)
                                            b = np.array([ mark_r, -mark_r, 0.0], dtype=np.float32)

                                            x1 = np.stack([p - a, p + a], axis=0).astype(np.float32)
                                            x2 = np.stack([p - b, p + b], axis=0).astype(np.float32)

                                            self.rr_logger.log({
                                                f"{entity_path}/confidence/mark": rr.LineStrips3D(
                                                    strips=[x1, x2],
                                                    radii=0.02,            # 선 굵기
                                                    colors=mark_color,
                                                )
                                            })

                                        # (G) 중심-라벨 연결선도 같은 색으로
                                        strips = [np.stack([c, l], axis=0) for c, l in zip(centers, label_pos)]
                                        self.rr_logger.log({
                                            f"{entity_path}/entity-confidence": rr.LineStrips3D(
                                                strips=strips,
                                                radii=0.02,
                                                colors=mark_color,
                                            )
                                        })
                                                                                

                            else:
                                self.log(f"<inference_loop.4.4.{_}> No updated agg_results")
                        except Exception as e:
                            self.log(f"<inference_loop.4.4.{_}> Error occurs: {e}", level='error')
                            self.rr_log(f"<inference_loop.4.4.{_}> Error occurs: {e}", panel='inference', level='error')    
                        finally:
                            # --- 예약 해제 (성공/실패 무관 1건) ---
                            # if (gid is not None) and (answer is not None): # TODO: fix error case
                            #     self.agg_results.release(gid, 1, eids=answer.eids)
                            #     self.agg_results.inc_queries(gid, 1, eids=answer.eids)
                            for answer in answers:
                                if answer is not None:
                                    self.agg_results.release(gid, 1, eids=answer.eids)
                                    self.agg_results.inc_queries(gid, 1, eids=answer.eids)

                            self.log(f"<inference_loop.4.4.{_}> Release group({gid})")
                            self.log(f"<inference_loop.4.5.{_}> AggResults: {self.agg_results}")
                            self.rr_log(f"<inference_loop.4.5.{_}> AggResults: {self.agg_results}", panel='inference')

            except Exception as e:
                self.log(f"<inference_loop.4> Error occurs: {e}", level='error')
                self.rr_log(f"<inference_loop.4> Error occurs: {e}", panel='inference', level='error')
            rate.sleep()

    def load_and_preprocess_image(self, image_path, preprocess=None):
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            raise FileNotFoundError(f"Invalid image file: {image_path}")
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def get_images(self, keyframes, suffix="", preprocess=None, **kwargs):
        # if isinstance(keyframes, Keyframe):
        #     keyframes = Keyframes({keyframes.id: keyframes})

        images, image_paths = [], []

        # Normalize suffix to always be a list for consistent processing
        suffixes = suffix if isinstance(suffix, list) else [suffix]
        suffix2etype = {
            '_annotated_global_object_object': 'object',
            '_annotated_global_object': 'object'
        } # TODO: Need to check

        for kf in keyframes:
            attrs = kf.get("_attrs", {})
            for s in suffixes:
                if s == "":
                    image_path = attrs['image_path']
                else:
                    image_path = self.sg.save_path(etype=suffix2etype[s], fname=attrs['fname'])

                try:
                    image = self.load_and_preprocess_image(image_path, preprocess)
                except Exception as e:
                    self.log(f"<get_images> Error occurs: {e}\n"
                             f"  > Failed to load and preprocess image: {image_path}\n"
                             f"  > keyframe: {kf}\n"
                             f"  > suffixes: {suffixes}\n", level='error')
                    self.rr_log(f"<get_images> Error occurs: {e}\n"
                                f"  > Failed to load and preprocess image: {image_path}\n"
                                f"  > keyframe: {kf}\n"
                                f"  > suffixes: {suffixes}\n", panel='inference', level='error')
                    continue

                images.append(image)
                image_paths.append(image_path)

        return images, image_paths

    @staticmethod
    def run_parallel(func, input_data, max_workers=3):
        it = iter(input_data)
        client_counter = 0
        in_flight = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 1) 초기 워커만큼 예열(submit)
            for _ in range(max_workers):
                try:
                    item = next(it)
                except StopIteration:
                    break
                fut = executor.submit(func, **item, client_counter=client_counter)
                in_flight[fut] = item
                client_counter += 1

            # 2) 완료되는 대로 결과를 내보내고, 다음 작업을 즉시 투입
            while in_flight:
                # 완료된 future만 순서 무관하게 가져옴
                for fut in concurrent.futures.as_completed(list(in_flight.keys()), timeout=None):
                    item = in_flight.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = e
                    yield (item, result)

                    # 빈 슬롯에 다음 작업 투입
                    try:
                        next_item = next(it)
                    except StopIteration:
                        # 더 이상 넣을 작업이 없으면 넘어감(남은 in_flight만 소진)
                        continue
                    new_fut = executor.submit(func, **next_item, client_counter=client_counter)
                    in_flight[new_fut] = next_item
                    client_counter += 1

    def _get_response_with_retry(self, client, message, get_response_opts,
                                 max_retries: int = 6,
                                 base_delay: float = 0.8,
                                 max_delay: float = 20.0):
        """
        503/429/5xx/타임아웃 등 일시 오류에 대해 지수 백오프 + 지터로 재시도.
        동시성은 self._llm_sema로 제한.
        """
        last_err = None
        for attempt in range(max_retries):
            # 동시성 제한
            with self._llm_sema:
                try:
                    # 실제 호출
                    return client.get_response(message, **get_response_opts)
                except Exception as e:
                    last_err = e
                    if not _is_retryable_llm_error(e):
                        raise  # 재시도 비대상은 바로 실패
                    # 지수 백오프 + 지터
                    delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
                    self.logger.logwarn(
                        f"[LLM] retryable error on attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
            time.sleep(delay)

        # 재시도 모두 실패
        raise last_err

    def query_worker(self, input_data, client_counter):
        phase = 0
        try:
            time.sleep(random.uniform(0, 0.25))

            client = self.clients[client_counter % len(self.clients)]
            self.log(f"<query_worker.0> Get client: {client_counter % len(self.clients)}")

            phase = 1
            keyframes = input_data['keyframes']

            candidate_eids_in_kfs = [] # TODO: Need to check
            self.log(f"<query_worker.0> 1")
            pid2eids = self.sg.pid2eids
            self.log(f"<query_worker.0> 2")
            for kf in keyframes:
                pid = kf['id'][1]
                self.log(f"<query_worker.0> 3")
                if pid is not None:
                    eids = pid2eids.get(pid, None)
                    if eids is None:
                        self.log(f"<query_worker.0> 4.1")
                        continue
                    self.log(f"<query_worker.0> 4.2")
                    filtered_eids = [
                        eid for (level, eid), entity in self.sg.G.nodes(data=True)
                        if (level == str(NodeLevel.OBJECT))
                           and (eid in eids)
                           and (entity.get("_attrs", {})['name'] in self.sg.candidate_names)
                    ]
                    self.log(f"<query_worker.0> 5")
                    candidate_eids_in_kfs += filtered_eids
            candidate_eids_in_kfs = sorted(set(candidate_eids_in_kfs)) 

            options = input_data.get('options', self.default_options)
            previous_history = options['prompt']['previous_history']  # TODO

            # Prepare the prompt and system instruction
            prompt = self.prompt_renderer.render(**options['prompt'], anno_ids=candidate_eids_in_kfs)
            self.log(f"<query_worker.0> 6")
            system_instruction = self.system_instruction_renderer.render(**options['prompt'])
            images, image_paths = self.get_images(keyframes, **options['image']) # TODO: FIX
            self.log(f"<query_worker.1> Prepare the input data")

            # Query the model
            phase = 2
            start_time = time.time()
            with self.sg_lock:
                message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
            end_time = time.time()
            self.log(f"<query_worker.2> Construct message with client")

            phase = 3
            response_text, _, _ = self._get_response_with_retry(
                client=client,
                message=message,
                get_response_opts=options['get_response'],
                max_retries=6,  # 필요시 조정
                base_delay=0.8,  # 0.8s, 1.6, 3.2, 6.4, ...
                max_delay=20.0
            )
            response = parse_json(response_text)
            if response:
                target_ids = response.get('target_ids', [])
                self.log(f"<query_worker.3> Get response")
            else:
                self.log(f"<query_worker.3> Failed to parse. response_text (truncated):\n"
                                    f"{str(response_text)[:1000]}", level='warn')

            # Log the response
            phase = 4
            result_text = "\n====================================================\n"
            result_text += f"Image paths: {image_paths}\n"
            result_text += f"  > {options['image']}\n"
            result_text += f"Rendering type (rtype): {options.get('prompt', {}).get('rtype')}\n"
            result_text += f"Action: {options.get('prompt', {}).get('action')}\n"
            result_text += f"Annotation type (atype): {options['prompt'].get('atype')}\n"
            result_text += f"Hint: {options['prompt'].get('hint')}\n"
            result_text += f"is_plural: {options['prompt'].get('is_plural')}\n"
            result_text += f"previous_history: {options['prompt'].get('previous_history')}\n"
            result_text += f"-----------------------------------------------------------\n"
            result_text += f"Prompt: {prompt}\n"
            result_text += f"-----------------------------------------------------------\n"
            # result_text += f"System instruction: {system_instruction}\n"
            # result_text += f"Response: {response_text}\n"
            result_text += f"Query time: {end_time - start_time:.2f} seconds\n"
            result_text += f"Response: {response}\n"
            result_text += f"====================================================\n\n"
            self.log(result_text)
            self.log(f"<query_worker.4> Print the log")
            
            self.rr_log(result_text, panel=['inference', 'details'])

            # Verify
            phase = 5
            if self.action in ['find']:
                if len(target_ids) > 1:
                    self.log(f"Object ids mismatch: len(target_ids)={len(target_ids)} != 1", level='warn')
                    self.rr_log(f"Object ids mismatch: len(target_ids)={len(target_ids)} != 1", panel=['inference', 'details'], level='warn')
                    return None
            else:
                if self.default_inference_options['prompt']['action'] == 'follow_between' \
                        and self.default_inference_options['prompt']['rtype'] == 'inference' \
                        and len(target_ids) != 2:
                    self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 2")
                    self.rr_log(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 2", panel=['inference', 'details'], level='warn')
                    return None
                elif self.default_inference_options['prompt']['action'] == 'find' \
                        and self.default_inference_options['prompt']['rtype'] == 'inference' \
                        and len(target_ids) != 1:
                    self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 1")
                    self.rr_log(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 1", panel=['inference', 'details'], level='warn')
                    return None
            self.log(f"<query_worker.5> Verify the response")
            return response
        except Exception as e:
            self.logger.logerr(f"<query_worker.{phase}> Error occurs: {e}")
            self.rr_log(f"<query_worker.{phase}> Error occurs: {e}", panel='inference', level='error')

    def inference(self, keyframes, etype='all', atype='object_box_id', client_counter=0, gid=None, *args, **kwargs):
        try:
            if isinstance(keyframes, dict):
                self.log(f"<inference.1> keyframes is dictionary. Use keyframes['keyframes']", level='warn')
                keyframes = keyframes['keyframes']
            if len(keyframes) == 0:
                self.logger.logwarn(f"<inference.1> No keyframes input was given.")
                return
            self.log(f"<inference.1> Annotation type is {atype}; Entity type is {etype};")
        except Exception as e:
            self.log(f"<inference.1> Error occurs: {e}", level='error')
            self.rr_log(f"<inference.1> Error occurs: {e}", panel='inference', level='error')

        try:
            options = copy.deepcopy(self.default_inference_options)
            options['prompt'].update({
                'rtype': 'inference',
                'action': 'select_box' if self.action in ['find', 'count'] else 'select_point',
                'atype': atype,
                'hint': 'none', # TODO: Implement
                'is_plural': False if self.action == 'find' else True, # TODO: Implement
            })
            if (options['prompt']['is_plural'] is True) and (options['prompt']['atype'] == 'none'):
                oldest_id = max([kf['id'][1] for kf in keyframes])
                keyframes = [kf for kf in keyframes if kf['id'][1] == oldest_id]# keyframes.get_single(oldest_id) # TODO: need to check

            suffix = options['image']['suffix']
            options['image'].update({'suffix': f"{suffix}_{etype}"})
            data_list = [{
                'keyframes': keyframes,  # TODO: need to check
                'options': options,
            }]
            self.log(f"<inference.2> options and data_list are ready.")
        except Exception as e:
            self.log(f"<inference.2> Error occurs: {e}")
            self.rr_log(f"<inference.2> Error occurs: {e}", panel='inference', level='error')

        try:
            self.log(f"<inference.3> GID={gid}")
            self.rr_log(f"<inference.3> GID={gid}", panel='inference')
            response = self.query_worker(data_list[0], client_counter=client_counter)
            if response is None:
                self.log(f"<inference.3> response is None.", level='warn')
                self.rr_log(f"<inference.3> response is None.", panel='inference', level='warn')
                return
            self.log(f"<inference.3> Get response from worker({client_counter})")
            self.rr_log(f"<inference.3> Get response from worker({client_counter})", panel='inference')
        except Exception as e:
            self.log(f"<inference.3> Error occurs: {e}", level='error')
            self.rr_log(f"<inference.3> Error occurs: {e}", panel='inference', level='error')

        # Save the result
        try:
            output = {
                'entity_type': etype,
                'data': data_list[0],
                'target_ids': response.get('target_ids'),
                'reason': response.get('reason'),
            }
            self.log(f"<inference.4> Set inference_ready_event, which is not used.")
        except Exception as e:
            self.log(f"<inference.4> Error occurs: {e}", level='error')
            self.rr_log(f"<inference.4> Error occurs: {e}", panel='inference', level='error')
        return output

# OTHERS
    def _wait_for_keys(self, param_name, check_hz=5.0, require_non_empty=True):
        """
        - param_name: Expected as node private parameter (e.g., '~api_keys' → /<node_name>/api_keys)
        - check_hz: Polling frequency
        - require_non_empty: Continue waiting if empty list
        """
        r = rospy.Rate(check_hz)
        last_log_t = rospy.Time(0)
        log_period = rospy.Duration(2.0)

        while not rospy.is_shutdown():
            if rospy.has_param(param_name):
                val = rospy.get_param(param_name)
                # Allowed formats: list or comma/space separated string
                if isinstance(val, str):
                    # Also allow "key1,key2" or "key1 key2"
                    parts = [p for p in val.replace(",", " ").split() if p]
                elif isinstance(val, (list, tuple)):
                    parts = list(val)
                else:
                    parts = []

                parts = [str(p).strip() for p in parts if str(p).strip()]

                if (not require_non_empty) or (len(parts) > 0):
                    self.logger.loginfo(f"{param_name} loaded (n={len(parts)})")
                    return parts
                else:
                    # Parameter exists but is empty → continue waiting
                    pass

            # Log periodically only to prevent log spam
            now = rospy.Time.now()
            if now - last_log_t > log_period:
                self.logger.loginfo(f"Waiting for parameter {param_name} from manager...")
                last_log_t = now

            r.sleep()

        # When node shuts down, reach here
        raise rospy.ROSInterruptException("Shutdown before ~api_keys was set.")

    def _to_sys_sec(self, stamp: rospy.Time) -> float:
        # Convert an absolute ROS time to seconds since system start.
        if not self.system_start_received or self.system_start_ros is None:
            return stamp.to_sec()  # fallback
        return (stamp - self.system_start_ros).to_sec()

class BaseActiveVisualGrounder(BaseVisualGrounder):
# INIT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ Navigation """
        self.num_frontiers = -1
        # self.min_point_spacing = 0.5
        self.radius = self.config.get('path_radius', 0.55) # (m)
        self.group_threshold = self.config.get('group_threshold', 0.5) # (m)
        self._empty_path_since = {}  # {gid: rospy.Time}
        self._empty_path_cooldown = rospy.Duration(3.0)  # 3초

        self.agent_pose = None
        self.hull_grouper = None
        self.path_points = None
        self.is_path_points_updated = False
        
        self.navigation_running = threading.Event()
        self.navigation_lock = threading.RLock()

        self.last_update_time_path_points = rospy.Time.now()
        self.update_interval_path_points = rospy.Duration(5.0)  # (sec)
        self.current_gid = 0
        self.history_eids = []

        self.logger.loginfo(f"=== configuration ===")
        for k, v in self.config.items():
            self.logger.loginfo(f"  {k} : {v}")
        self.logger.loginfo(f"=====================")

    def _init_services(self, *args, **kwargs) -> None:
        super()._init_services(*args, **kwargs)
        
        """ Active clients """
        self.active_clients = ActiveClients(*args, **kwargs)

    def _init_subscribers(self, *args, **kwargs):
        super()._init_subscribers(*args, **kwargs)
        
        """ Robot current state """
        self.log("Odom Sub")
        self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=20)
        self.odom_sub2 = rospy.Subscriber("/Odometry", Odometry, self._odom_callback2, queue_size=20)

        """ Traversable area """
        self.traversable_points = None
        self.vis_traversable_points = False
        self.traversable_path = traversable_path = os.environ.get("TRAVERSABLE_PATH", self.config.get("traversable_path", None))
        if traversable_path is not None:
            cols, arr = load_pcd_ascii_with_fields(traversable_path)
            col2idx = {c: i for i, c in enumerate(cols)}
            xyz = arr[:, [col2idx["x"], col2idx["y"], col2idx["z"]]]

            collision_risk = arr[:, col2idx["collision_risk"]]
            mask = np.ones(arr.shape[0], dtype=bool)
            mask = mask & (collision_risk < 0.1)
            if not 'postprocessed' in traversable_path:
                # filtered_arr, mask = filter_disconnected_traversable(cols, arr, res=0.11, collision_thr=0.1, use_8n=False)
                print("collision_risk:", float(np.nanmin(collision_risk)), float(np.nanmax(collision_risk)))
                def mask_area(aabb):
                    xmin, ymin, xmax, ymax = aabb
                    return ~((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                             (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))
                mask = (mask
                        & mask_area((-1.6, -2.3, 4.3, -0.9))    # workspace
                        & mask_area((3.5, -0.8, 4.2, 3.9))      # TV monitor
                        & mask_area((-2.0, 4.0, 4.0, 7.4))      # storage
                        & mask_area((0.2, 2.2, 2.3, 3.3))       # table
                        & mask_area((-1.8, -0.8, -1.3, 3.5)))   # chairs

                arr[~mask, col2idx['collision_risk']] = np.inf
                # arr[:, col2idx['z']] += 0.8
                arr_new = np.zeros((0,21), dtype=np.float32)
                for _arr in arr:
                    if _arr[col2idx['collision_risk']] < 0.1:
                        arr_new = np.concatenate([arr_new, _arr[None, :]], axis=0)

                save_pcd_ascii_with_header(traversable_path, '/ws/data/VLA/E3_3225_TRIP_postprocessed_p08.pcd', arr)

            self.traversable_points = np.asarray(xyz[mask]) # self.traversable_points = np.asarray(xyz[~mask])
            self.risky_points = np.asarray(xyz[~mask]) # self.risky_points = np.asarray(xyz_risky)
            self.vis_traversable_points = True
        self._traversable_lock = threading.RLock()
        self.traversable_area_sub = rospy.Subscriber(
            "/traversable_area_filtered", PointCloud2, self._traversable_area_callback, queue_size=10)
        
        """ Occupancy grid """
        self.occupancy_grid = None
        self.occupancy_grid_sub = rospy.Subscriber("/occupancy_map", OccupancyGrid, self._occupancy_grid_callback, queue_size=1)

        """ Path history """
        self.path_xy = np.zeros((0, 2), dtype=float)
        self.robot_path_sub = rospy.Subscriber("/path_recorder/path", Path, self._robot_path_callback, queue_size=1)

        """ Exploration status """
        self.exploration_status = None
        self.exploration_status_sub = rospy.Subscriber("/instruction_following_exp_status", String, self._exploration_status_callback, queue_size=1)
        self.logger.loginfo(f"Init self.exploration_status_sub")

        """ Timeout """
        self.timeout_sub = rospy.Subscriber(
            "/timeout", Empty, self._timeout_callback, queue_size=10)

    def _init_publishers(self, *args, **kwargs):
        super()._init_publishers(*args, **kwargs)
        
        """ Active navigation """
        self.path_points_pub = rospy.Publisher("/active_waypoints", MarkerArray, queue_size=1)
        
        # Visualization
        self.previous_path_points = None
        self.path_points_vis_pub = rospy.Publisher("/active_waypoints_vis", MarkerArray, queue_size=1)

        """ Exploration strategy """
        self.exploration_strategy_pub = rospy.Publisher("/exploration_strategy", String, queue_size=1)

# RESET
    def _reset_vars(self):
        super()._reset_vars()
        
        """ Navigation """
        self.path_points = None
        self.navigation_running = threading.Event()
        self.active_clients.end()

    def log_agent(self, agent_pose):
        try:
            theta = theta_from_agent_pose(agent_pose['orientation'])
            pos = agent_pose['position']
            dir_vec = np.array([[np.cos(theta), np.sin(theta), 0.0]], dtype=np.float32)
            if hasattr(self, 'agent_arrow_len'):
                if self.agent_arrow_len is not None:
                    self.rr_logger.log({
                        "SG/agent": rr.Arrows3D(origins=pos, vectors=dir_vec * self.agent_arrow_len,
                                                colors=[0, 0, 255], radii=0.1)
                    })
                    # balloon_pos = pos + 0.3 * dir_vec * self.agent_arrow_len + np.array([[0.0, 0.0, 0.3]])
                    # self.rr_logger.log({f'SG/message': rr.Points3D(
                    #     positions=balloon_pos, labels=[self.current_agent_message], radii=0.001, colors=[255, 255, 255])})
                    self.rr_logger.log({"answer/answer": rr.TextDocument(f"## {self.current_agent_message}", media_type=rr.MediaType.MARKDOWN)})
        except Exception as e:
            self.logger.logerr(f"<log_agent.1> Error occurs: {e}")
        # arrow_tip = pos + 0.3 * dir_vec * self.agent_arrow_len + np.array([[0.0, 0.0, 0.1]])
        # tail = np.concatenate([arrow_tip, balloon_pos], axis=0).astype(np.float32)  # (2,3)
        # self.rr_logger.log({'SG/message_tail': rr.LineStrips3D(tail, colors=[49, 56, 59], radii=0.01)})

# CALLBACKS
    def _odom_callback(self, msg):
        try:
            self.log("_odom_callback")
            self.agent_pose = agent_pose = {
                "position": np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]),
                "orientation": np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]),
            }
            self.log_agent(agent_pose)

            if hasattr(self, 'path_xy'):
                self.path_xy = path_xy = np.concatenate([self.path_xy, [agent_pose['position'][:2]]], axis=0)
                if len(path_xy) > 1 and getattr(self, 'sg'):
                    if hasattr(self.sg, 'z_const'):
                        v0, t0, c0 = build_ribbon_mesh(path_xy, z=self.sg.z_const - 0.01, width=self.radius * 2,
                                                    rgb_u8=[0, 255, 255])
                        # if (v0 is not None) and (t0 is not None):
                        #     self.rr_logger.log(
                        #         {'SG/agent/path_xy_range': rr.Mesh3D(vertex_positions=v0, triangle_indices=t0, vertex_colors=c0)})
                        v1, t1, c1 = build_ribbon_mesh(path_xy, z=self.sg.z_const, width=0.05, rgb_u8=[0, 0, 255])
                        if (v1 is not None) and (t1 is not None):
                            self.rr_logger.log(
                                {'SG/agent/path_xy': rr.Mesh3D(vertex_positions=v1, triangle_indices=t1, vertex_colors=c1)})

            if self.debug:  # TODO: debug: Save the path_xy
                _ = save_path_xy(agent_pose['position'], base_dir=self.offline_map_dir, name="position")
                _ = save_path_xy(agent_pose['orientation'], base_dir=self.offline_map_dir, name="orientation")
        except Exception as e:
            self.logger.logerr(f"<_odom_callback.1> Error occurs: {e}")

    def _odom_callback2(self, msg):
        try:
            self.log("_odom_callback2")
            self.agent_pose = agent_pose = {
                "position": np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]),
                "orientation": np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]),
            }
            self.log_agent(agent_pose)

            if hasattr(self, 'path_xy'):
                self.path_xy = path_xy = np.concatenate([self.path_xy, [agent_pose['position'][:2]]], axis=0)
                if len(path_xy) > 1 and getattr(self, 'sg'):
                    if hasattr(self.sg, 'z_const'):
                        v0, t0, c0 = build_ribbon_mesh(path_xy, z=self.sg.z_const - 0.01, width=self.radius * 2,
                                                    rgb_u8=[0, 255, 255])
                        # if (v0 is not None) and (t0 is not None):
                        #     self.rr_logger.log(
                        #         {'SG/agent/path_xy_range': rr.Mesh3D(vertex_positions=v0, triangle_indices=t0, vertex_colors=c0)})
                        v1, t1, c1 = build_ribbon_mesh(path_xy, z=self.sg.z_const, width=0.05, rgb_u8=[0, 0, 255])
                        if (v1 is not None) and (t1 is not None):
                            self.rr_logger.log(
                                {'SG/agent/path_xy': rr.Mesh3D(vertex_positions=v1, triangle_indices=t1, vertex_colors=c1)})

            if self.debug:  # TODO: debug: Save the path_xy
                _ = save_path_xy(agent_pose['position'], base_dir=self.offline_map_dir, name="position")
                _ = save_path_xy(agent_pose['orientation'], base_dir=self.offline_map_dir, name="orientation")
        except Exception as e:
            self.logger.logerr(f"<_odom_callback.1> Error occurs: {e}")

    def _traversable_area_callback(self, msg) -> None:
        if self.traversable_points is None:
            traversable_pts, _ = pointcloud2_to_xy_array(msg)
            with self._traversable_lock:
                self.traversable_points = traversable_pts
                self.vis_traversable_points = True

    def _occupancy_grid_callback(self, msg):
        self.occupancy_grid = CustomOccupancyGrid(msg)
        if self.debug:
            filename = f"occupancy_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            save_path = os.path.join("/ws/external/offline_map/", filename)
            self.occupancy_grid.save_npz(save_path)

    def _robot_path_callback(self, msg):
        pts = []
        for ps in msg.poses:
            x = ps.pose.position.x
            y = ps.pose.position.y
            pts.append([x, y])
        if len(pts) > 0:
            self.path_xy = np.array(pts, dtype=float)
        # else:
        #     self.path_xy = np.zeros((0, 2), dtype=float)

        if self.debug:  # TODO: debug: Save the path_xy
            _ = save_path_xy(self.path_xy, base_dir="/ws/external/offline_map", name="path_xy")

    def _exploration_status_callback(self, msg):
        self.exploration_status = msg.data
        self.logger.loginfo(f">>>>>>>> Exploration status: {self.exploration_status}")
        if self.exploration_status == 'no_frontier':
            (thres_low, thres_high) = self.confidence_threshold
            with self.agg_results_lock:
                agg_results = self.agg_results.snapshot()
            best_confidence = agg_results.get('best_confidence')
            if (best_confidence > thres_high):
                if self.start_time:
                    now = rospy.Time.now()
                    elapsed = now - self.start_time
                else:
                    elapsed = rospy.Time.now()
                self.logger.logrich(f"<inference_loop.3.2> Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status}", name='time')
                self.answer_result = self.agg_results.best_answer
                self.answer_the_question(self.answer_result)
                self.logger.loginfo(f"<inference_loop.3.2> Answer the final result. Confidence: {best_confidence} >= {thres_high}.")
            else:
                self.logger.loginfo(f"<inference_loop.3.2> Answer the final result. Confidence: {best_confidence} < {thres_high}.")
            return

    def _timeout_callback(self, msg) -> None:
        if self.status != Status.COMPLETED:
            self.logger.loginfo("<timeout_callback> Timeout signal received #############")
            self.answer_result = self.agg_results.best_answer
            if self.answer_result is None:
                if self.action == 'find':
                    self.answer_result = random.choice(list(self.sg.get_candidate_entities('object', include_untracked=False).ids))
                elif self.action == 'count':
                    self.answer_result = random.randint(2, 6)
                elif self.action is None:
                    self.logger.logwarn(f"action is None")
                else:
                    raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")
            self.answer_the_question(self.answer_result)
        else:
            self.answer_the_question(self.answer_result)
            self.logger.loginfo("<timeout_callback> Finished :) #############")
        return

    def log(self, text, level='info'):
        if level == 'warn':
            self.logger.logwarn(text)
        elif level == 'error':
            self.logger.logerr(text)
        else:
            self.logger.loginfo(text)
        
    def rr_log(self, text, panel, level='info'):
        panels = panel if isinstance(panel, (list, tuple)) else [panel]

        payload = {f'VG/{p}': text for p in panels}
        self.rr_logger.log(payload, level=level.upper())

# MAIN LOOP
    def process(self, **kwargs):
        try:
            self.log(f"<process.0> Start")
            self.update_resource(**kwargs)
        except Exception as e:
            self.logger.logerr(f"<process.0> Error occurs: {e}")
        if False: #  'dir' in kwargs:
            def load_latest_data(dir_path: str, name: str='agent_pose_'):
                pose_files = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.startswith(name)
                ]
                if not pose_files:
                    return None, None

                latest_path = max(pose_files, key=os.path.getmtime)  # 가장 오래된 파일
                if latest_path.endswith("jpg"):
                    data = cv2.imread(latest_path)
                else:
                    data = np.load(latest_path)
                return data, latest_path

            # self.agent_pose, _ = load_latest_agent_pose(kwargs['dir'])
            position, _ = load_latest_data(kwargs['dir'], name='position')
            orientation, _ = load_latest_data(kwargs['dir'], name='orientation')
            self.agent_pose = agent_pose = {'position': position, 'orientation': orientation}
            self.log(f"self.agent_pose: {agent_pose}")
            self.log_agent(agent_pose)

            rgb, _ = load_latest_data(kwargs['dir'], name='rgb')
            if rgb is not None:
                self.rr_logger.log({'obs/rgb': rr.Image(rgb)})
            self.log(f"<process.0> Read agent_pose from {_}")

            self.path_xy = path_xy = np.concatenate([self.path_xy, [position[:2]]], axis=0)
            if len(path_xy) > 1 and getattr(self, 'sg'):
                if hasattr(self.sg, 'z_const'):
                    v0, t0, c0 = build_ribbon_mesh(path_xy, z=self.sg.z_const - 0.01, width=self.radius * 2, rgb_u8=[0, 255, 255])
                    if (v0 is not None) and (t0 is not None):
                        self.rr_logger.log({'SG/agent/path_xy_range': rr.Mesh3D(vertex_positions=v0, triangle_indices=t0, vertex_colors=c0)})
                    v1, t1, c1 = build_ribbon_mesh(path_xy, z=self.sg.z_const, width=0.05, rgb_u8=[0, 0, 255])
                    if (v1 is not None) and (t1 is not None):
                        self.rr_logger.log({'SG/agent/path_xy': rr.Mesh3D(vertex_positions=v1, triangle_indices=t1, vertex_colors=c1)})

        # Select Group ID
        num_queries_required = 1 # self.agg_results.min_query
        try:
            (gid, eids) = self.inference_signal_queue.get_nowait()
            # self.agg_results.generate(gid=gid)  # 모든 related_entity는 어떤 group에 할당됨
            self.logger.loginfo(f"<process.1> New EIDs are given: {eids} (GID={gid})")
            self.history_eids += eids
        except queue.Empty:
            if self.action == 'find':
                # NOTE: Only etype=='object' is supported.
                # TODO: Need to check
                candidate_eids = list(set(self.sg.get_candidate_entities()) & set(self.history_eids))
                pending_eids = sorted(
                    [
                        eid for eid in candidate_eids
                        if self.agg_results.results_by_entity.num_queries.get(eid, 0) < self.agg_results.min_query
                    ],
                    key=lambda eid: self.agg_results.results_by_entity.num_queries.get(eid, 0)
                )
                # pending_eids = []
                gid = None
            elif self.action == 'count':
                if self.etypes == ['object']:
                    # NOTE: Only etype=='object' is supported.
                    candidate_eids = self.sg.get_candidate_entities()
                    pending_eids = sorted(
                        [
                            eid for eid in candidate_eids
                            if self.agg_results.results_by_entity.num_queries.get(eid, 0) < self.agg_results.min_query
                        ],
                        key=lambda eid: self.agg_results.results_by_entity_count.num_queries.get(eid, 0)
                    )
                    gid = None
                else:
                    pending_gids = self.agg_results.get_pending_ids()  # GIDs
                    if len(pending_gids) == 0:
                        self.logger.loginfo(f"<process.1> There is no pending_gids")
                        return
                    gid = pending_gids[0]
                    # pending_eids = self.agg_results.results.get(gid, set()) # error
                    pending_eids = self.hull_grouper.groups(gid)
                    self.logger.loginfo(f"<process.1> There are pending_gids: {gid} (pending_eids:{pending_eids})")
            else:
                self.logger.logerr(f"<process.1> Error occurs: self.action must be in ['find', 'count'], bug {self.action} was given.")

            if len(pending_eids) == 0:
                self.log(f"<process.1> Don't need to more inference")
                self.rr_log(f"<process.1> Don't need to more inference", panel='main')
                return
            eids = pending_eids
            self.log(f"<process.1> Pending EIDs are selected: {eids}")
            self.rr_log(f"<process.1> Pending EIDs are selected: {eids}", panel='main')

        # Get Keyframes of the group gid
        try:
            # eids = self.hull_grouper.groups(gid)
            candidate_eids = self.sg.get_candidate_entities()
            candidate_eids_in_group = list(set(eids) & set(candidate_eids))
            if len(candidate_eids_in_group) == 0:
                data, pids = [], []
            else:
                data, pids = self.build_batch_for_gid(
                    eids=candidate_eids_in_group, num_queries_required=num_queries_required
                )
            self.log(f"<process.2> Selected KFs: {list(set(pids))}")
            self.rr_log(f"<process.2> Selected KFs: {list(set(pids))}", panel='main')
        except Exception as e:
            self.log(f"<process.2> Error occurs: {e}", level='error')
            self.rr_log(f"<process.2> Error occurs: {e}", panel='main', level='error')

        # Put the data to inference_queue
        try:
            if len(data) > 0:
                try:
                    self.inference_queue.put_nowait(data)
                    self.agg_results.schedule(gid, n=len(data), data=data)
                    self.log(
                        "<process.3> Put data to inference_queue:\n" +
                        "\n".join([f"  > pids: {', '.join(map(str, [kf['id'][1] for kf in d['keyframes']]))}, etype: {d['etype']}, atype: {d['atype']}, eids: {d['eids']}" for d in data])
                    )
                    self.rr_log(
                        "<process.3> Put data to inference_queue:\n" +
                        "\n".join([f"  > pids: {', '.join(map(str, [kf['id'][1] for kf in d['keyframes']]))}, etype: {d['etype']}, atype: {d['atype']}, eids: {d['eids']}" for d in data]),
                        panel='main'
                    )
                except queue.Full:
                    self.log("<process.3> inference_queue is full. Will retry next tick.", level='warn')
                    self.rr_log("<process.3> inference_queue is full. Will retry next tick.", panel='main', level='warn')
            else:
                self.log(f"<process.3> No data to put: {data}", level='error')
                self.rr_log(f"<process.3> No data to put: {data}", panel='main', level='error')
        except Exception as e:
            self.logger.logerr(f"<process.3> Error occurs: {e}")
            self.rr_log(f"<process.3> Error occurs: {e}", panel='main', level='error')

# NAVIGATION LOOP
    def navigation_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            self.log(f"<navigation_loop.1> Resource status: {'Updated' if self.updated_resource else 'Not yet'}")
            self.rr_log(f"<navigation_loop.1> Resource status: {'Updated' if self.updated_resource else 'Not yet'}", panel='nav')
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
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.1> Error occurs: {e}")

            self.navigation_running.set()

            try:
                self.update_path_points()
                self.log(f"<navigation_loop.2> Updated path_points: #={len(self.path_points) if self.path_points is not None else 'None'}")
                self.rr_log(f"<navigation_loop.2> Updated path_points: #={len(self.path_points) if self.path_points is not None else 'None'}", panel='nav')
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.2> Error occurs: {e}")

            try:
                # self._get_node_active_signal()
                # if self.node_active_signal:
                if self.node_active_signal:
                    self.navigate()
                self.log(f"<navigation_loop.3> Navigate")
                self.rr_log(f"<navigation_loop.3> Navigate", panel='nav')
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.3> Error occurs: {e}")
            finally:
                self.navigation_running.clear()
                self.log(f"<navigation_loop.3> Clear navigation_running")
                self.rr_log(f"<navigation_loop.3> Clear navigation_running", panel='nav')
            rate.sleep()

    def update_path_points(self) -> None:
        # TODO: Detection-based update_path_points
        try:
            with self.sg_lock:
                related_objects = self.sg.get_related_entities() # NOTE: Don't support etype!='object'
            self.log(f"<update_path_points.1> #related_objects = {len(related_objects)}")
            self.rr_log(f"<update_path_points.1> #related_objects = {len(related_objects)}", panel='nav')
        except Exception as e:
            self.log(f"<update_path_points.1> Error occurs: {e}", level='error')
            self.rr_log(f"<update_path_points.1> Error occurs: {e}", panel='nav', level='error')

        try:
            self.log(f"<update_path_points.2> 1")
            group_hulls_bef = None
            if self.hull_grouper is None:
                self.hull_grouper = GridGrouper(threshold=self.group_threshold).fit(related_objects) # threshold 높을수록 넓은 범위까지 group으로 인정
                self.log(f"<update_path_points.2> 1.1")
            else:
                group_hulls_bef = self.hull_grouper.group_hulls()
                self.log(f"<update_path_points.2> 1.2.1")
                self.hull_grouper.update(related_objects)
                self.log(f"<update_path_points.2> 1.2.2")
            self.log(f"<update_path_points.2> 2")

            self.hull_grouper.update_visibility(self.agent_pose, sg=self.sg, max_range=8.0)
            self.log(f"<update_path_points.2> 3")

            now = rospy.Time.now()
            updated_time_diff = now - self.last_update_time_path_points
            group_hulls = self.hull_grouper.group_hulls()
            self.log(f"<update_path_points.2> 4")

            is_equal_group_hulls = is_equal(group_hulls_bef, group_hulls)
            self.log(f"<update_path_points.2> 5")
            if is_equal_group_hulls and (updated_time_diff < self.update_interval_path_points):
                self.log(f"<update_path_points.2> No update group_hulls: #GIDs={len(group_hulls)}")
                self.rr_log(f"<update_path_points.2> No update group_hulls: #GIDs={len(group_hulls)}", panel='nav')
                return
            self.log(f"<update_path_points.2> Updated group_hulls: #GIDs={len(group_hulls)}")
            self.rr_log(f"<update_path_points.2> Updated group_hulls: #GIDs={len(group_hulls)}", panel='nav')
        except Exception as e:
            self.log(f"<update_path_points.2> Error occurs: {e}", level='error')
            self.rr_log(f"<update_path_points.2> Error occurs: {e}", panel='nav', level='error')

        try:
            path_points, log_data = {}, {}
            for _, group in enumerate(group_hulls): # TODO: Add traversable_area
                hull_xy = group['hull']
                self.logger.log(f"hull_xy: {hull_xy.shape}")
                # hull_xyz = np.hstack([hull_xy, np.ones((6, 1)) * self.sg.z_const]).shape
                gid = group['gid']

                # Visualize visibility
                edge_sigs, edge_visible = group['edge_sigs'], group['edge_visible']
                visible_strips, invisible_strips = [], []
                for (p0, p1), visible in zip(edge_sigs, edge_visible):
                    strip = np.array([[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0]], dtype=np.float32)
                    (visible_strips if visible else invisible_strips).append(strip)
                if visible_strips:
                    self.rr_logger.log({
                        f'SG/group/{gid}/visible_hull':
                            rr.LineStrips3D(visible_strips, colors=[[0, 255, 0]], radii=0.02)})
                if invisible_strips:
                    self.rr_logger.log({
                        f'SG/group/{gid}/invisible_hull':
                            rr.LineStrips3D(invisible_strips, colors=[[255, 0, 0]], radii=0.02)})

                # Visualize edges (OBJECT <-> GROUP)
                member_eids = group['members']
                members = [self.sg.G.nodes[('NodeLevel.OBJECT', eid)] for eid in member_eids]
                members_centroids = []
                for mem in members:
                    centroid = np.array(mem.get("_attrs", {}).get('centroid', None))
                    if centroid is not None:
                        members_centroids.append(centroid)
                members_centroids = np.stack(members_centroids, axis=0)
                if len(members_centroids) > 0:
                    group_xyz = np.mean(members_centroids, axis=0)
                    group_xyz[2] = 3.0
                    self.rr_logger.log({f"SG/nodes/NodeLevel.GROUP/{gid}": rr.Points3D(group_xyz, colors=[155, 155, 130], radii=0.15, labels=gid)})

                    for eid in group['members']:
                        mem = self.sg.G.nodes[('NodeLevel.OBJECT', eid)]
                        mem_xyz = np.array(mem.get("_attrs", {})['centroid'])
                        seg = np.stack([mem_xyz, group_xyz], axis=0)  # shape (2,3)
                        self.rr_logger.log({
                            f"SG/edges/GROUP{gid}/OBJECT{eid}": rr.LineStrips3D(seg, colors=[[155, 155, 130]], radii=0.02)
                        })

                # Compute active_waypoints
                nearest_points = find_closest_point(hull_xy, self.traversable_points)
                if nearest_points.shape[1] == 2:
                    nearest_points = np.hstack([nearest_points, np.zeros((len(nearest_points), 1))])
                self.rr_logger.log({f'SG/active_waypoints/{gid}': rr.Points3D(nearest_points, colors=[255, 0, 255, 200], radii=0.1)}) # pink
                # new_filtered_path_points = filter_close_points(nearest_points, self.min_point_spacing)
                new_filtered_path_points = nearest_points  # Now, we don't need to sampling the points.
                kept_mask, wps_keep = filter_waypoints_by_path(new_filtered_path_points, self.path_xy, self.radius)
                active_waypoints = []
                for i, keep in enumerate(kept_mask):
                    if keep:
                        active_waypoints.append(new_filtered_path_points[i])

                current_path_points = np.array(active_waypoints)
                with self.navigation_lock:
                    path_points.update({gid: current_path_points})
                log_data.update({gid: (len(current_path_points), len(nearest_points))})

                path_points_marker = make_marker_array_from_points(
                    nearest_points, ns=f"path_points_all", color=(0.5, 0.5, 0.5, 0.5), frame_id=self.frame_id)
                self.path_points_vis_pub.publish(path_points_marker)
                self.log(f"<update_path_points.3.1.{_}> Updated path_points for GID({gid}): #={len(current_path_points)}")
                self.rr_log(f"<update_path_points.3.1.{_}> Updated path_points for GID({gid}): #={len(current_path_points)}", panel='nav')

            self.path_points = path_points
            self.is_path_points_updated = True
            self.last_update_time_path_points = now
            # self.log(f"<update_path_points.3> log_data:")
            self.log(f"<update_path_points.3> path_points (#valid/#total): {{{', '.join([f'{gid}: ({valid}/{total})' for gid, (valid, total) in log_data.items()])}}}")
            self.rr_log(f"<update_path_points.3> path_points (#valid/#total): {{{', '.join([f'{gid}: ({valid}/{total})' for gid, (valid, total) in log_data.items()])}}}", panel='nav')
        except Exception as e:
            self.log(f"<update_path_points.3> Error occurs: {e}", level='error')
            self.rr_log(f"<update_path_points.3> Error occurs: {e}", panel='nav', level='error')

    def navigate(self):
        try:
            with self.navigation_lock:
                path_points_all = self.path_points

            if (path_points_all is None) or (len(path_points_all) == 0):
                self.logger.loginfo(f"<navigate.1> path_points_all is None. Skip this turn.")
                self.rr_log(f"<navigate.1> path_points_all is None. Skip this turn.", panel='nav')
                return
            self.logger.loginfo(f"<navigate.1> self.path_points: #GIDs={len(path_points_all)}")
            self.rr_log(f"<navigate.1> self.path_points: #GIDs={len(path_points_all)}", panel='nav')
        except Exception as e:
            self.logger.logerr(f"<navigate.1> Error occurs: {e}")
            self.rr_log(f"<navigate.1> Error occurs: {e}", penel='nav', level='error')

        try:
            if not self.is_path_points_updated:
                if self.previous_path_points:
                    self.path_points_vis_pub.publish(self.previous_path_points)
                self.logger.loginfo(f"<navigate.2> Publish previous path_points for visualization")
                self.rr_log(f"<navigate.2> Publish previous path_points for visualization", panel='nav')
                return
            self.logger.loginfo(f"<navigate.2> ...")
            self.rr_log(f"<navigate.2> ...", panel='nav')
        except Exception as e:
            self.logger.logerr(f"<navigate.2> Error occurs: {e}")
            self.rr_log(f"<navigate.2> Error occurs: {e}", panel='nav')

        try:
            colors = _color_palette(len(path_points_all), alpha=0.5)
            current_gid = self.current_gid
            self.logger.logrich(f"<navigate.3> Current group ID: {current_gid}", name="gid")
            self.rr_log(f"<navigate.3> Current group ID: {current_gid}", panel='nav')

            # Fail to find current_gid
            if current_gid is None:
                self.logger.loginfo(f"<navigate.3> Current GID is None.")
                self.rr_log(f"<navigate.3> Current GID is None.", panel='nav')
                if self.agent_pose:
                    agent_xy = np.array(self.agent_pose['position'][:2], dtype=float)
                    min_dist = float("inf")
                    min_gid = None
                    for gid, path_points in path_points_all.items():
                        if len(path_points) == 0:
                            continue
                        dist = min_distance(agent_xy, path_points[:, :2])
                        if dist < min_dist:
                            min_dist = dist
                            min_gid = gid
                    self.current_gid = min_gid
                    self.logger.loginfo(f"<navigate.3> Changed GID: {current_gid} -> {self.current_gid}")
                    self.rr_log(f"<navigate.3> Changed GID: {current_gid} -> {self.current_gid}", panel='nav')
                    
                # current_gid = self.current_gid
                # current_path_points = path_points_all.get(current_gid, [])
                # if len(current_path_points) == 0:
                #     now = rospy.Time.now()
                #     t0 = self._empty_path_since.get(current_gid)

                #     if t0 is None:
                #         # 처음 빈 상태 감지 → 타이머 시작
                #         self._empty_path_since[current_gid] = now
                #         self.logger.loginfo(f"<navigate.bump> Start empty-path timer for GID({current_gid})")
                #     else:
                #         candidate_eids1 = self.sg.get_candidate_entities('object').ids
                #         candidate_eids2 = self.sg.get_candidate_entities('detection').ids
                #         candidate_eids = list(set(candidate_eids1 + candidate_eids2))
                #         pending_eids = sorted(
                #             [
                #                 eid for eid in candidate_eids
                #                 if self.agg_results.results_by_entity.num_queries.get(eid, 0) < self.agg_results.min_query
                #             ],
                #             key=lambda eid: self.agg_results.results_by_entity.num_queries.get(eid, 0)
                #         )
                #         pending_candidate_eids = list(set(pending_eids) & set(candidate_eids))

                #         dur = now - t0
                #         if dur > self._empty_path_cooldown and len(pending_candidate_eids) == 0:
                #             before = self.agg_results.min_query
                #             self.agg_results.min_query = min(
                #                 self.agg_results.min_query + 1,
                #                 getattr(self.agg_results, "max_query", self.agg_results.min_query + 1)
                #             )
                #             after = self.agg_results.min_query
                #             self.logger.loginfo(
                #                 f"<navigate.bump> Increase min_query: {before} -> {after} "
                #                 f"(empty-path {dur.to_sec():.3f} sec)"
                #             )

                #             # 2) 이 GID에 대해 즉시 배치 구성 후 enqueue
                #             try:
                #                 num_queries_required = self.agg_results.min_query
                #                 if len(pending_candidate_eids) == 0:
                #                     data, pids = [], []
                #                 else:
                #                     data, pids = self.build_batch_for_gid(
                #                         eids=pending_candidate_eids, num_queries_required=num_queries_required
                #                     )
                #                 self.logger.loginfo(f"<navigate.bump> Selected KFs: {list(set(pids))}")
                #                 if data:
                #                     # 큐에 투입
                #                     try:
                #                         self.inference_queue.put_nowait(data)
                #                     except queue.Full:  # 꽉 차면 다음 턴에 시도 (예약도 건너뜀)
                #                         self.logger.logwarn("<navigate.bump> inference_queue full; will retry later.")
                #                     else:
                #                         # 예약 증가
                #                         self.agg_results.schedule(current_gid, n=len(data), data=data)
                #                         self.logger.loginfo(
                #                             "<navigate.bump> Put data to inference_queue:\n" +
                #                             "\n".join([
                #                                 f"  > pids: {', '.join(map(str, d['keyframes'].ids))}, etype: {d['etype']}, atype: {d['atype']}, eids: {d['eids']}"
                #                                 for d in data
                #                             ])
                #                         )
                #                 else:
                #                     self.logger.loginfo(f"<navigate.bump> No keyframes for GID({current_gid})")
                #             except Exception as e:
                #                 self.logger.logerr(f"<navigate.bump> Error while enqueue: {e}")

                #             # 3) 타이머 리셋(지속적으로 쏟아내지 않도록)
                #             self._empty_path_since[current_gid] = now  # 또는 None으로 초기화도 가능
                #     # ------------------ ✅ 끝 ------------------

                exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                return
            
            # Success to find current_gid
            current_path_points = path_points_all.get(current_gid)
            if current_path_points is None:
                self.logger.loginfo(f"<navigate.3> Current path_points for GID({current_gid}) is None. Selecting next GID.")
                self.rr_log(f"<navigate.3> Current path_points for GID({current_gid}) is None. Selecting next GID.", panel='nav')
                candidate_gids = [gid for gid, pts in path_points_all.items() if pts is not None and len(pts) > 0]
                if not candidate_gids:
                    self.logger.loginfo(f"<navigate.3> No valid path_points to switch.")
                    self.rr_log(f"<navigate.3> No valid path_points to switch.", panel='nav')
                    exp_strategy = "geometric_frontier"
                    self.exploration_strategy_pub.publish(String(exp_strategy))
                    return
                candidate_gids.sort()
                prev_gid = current_gid
                next_gid = None
                for gid in candidate_gids:
                    if gid > current_gid:
                        next_gid = gid
                        break
                if next_gid is None:
                    next_gid = candidate_gids[0]
                self.current_gid = next_gid
                current_gid = next_gid
                current_path_points = path_points_all.get(current_gid)
                self.logger.loginfo(f"<navigate.3> Changed GID: {prev_gid} -> {self.current_gid}")
                self.rr_log(f"<navigate.3> Changed GID: {prev_gid} -> {self.current_gid}", panel='nav')
            exp_strategy = "geometric_frontier"
            self.exploration_strategy_pub.publish(String(exp_strategy))
            self.logger.loginfo(f"<navigate.3> current_path_points: {current_path_points.shape}")
            self.rr_log(f"<navigate.3> current_path_points: {current_path_points.shape}", panel='nav')
            self.current_agent_message = f"Let's observe Group({current_gid})!"
        except Exception as e:
            self.logger.logerr(f"<navigate.3> Error occurs: {e}")
            self.rr_log(f"<navigate.3> Error occurs: {e}", panel='nav', level='error')

        try:
            if len(current_path_points) == 0:
                current_eids = set(self.hull_grouper.dsu.idx.keys()) & set(self.sg.get_candidate_entities())
                self.logger.loginfo(f"<navigate.4.0> current_eids: {current_eids}")
                self.logger.loginfo(f"  >> current_gid: {current_gid}")
                self.logger.loginfo(f"  >> self.hull_grouper.dsu.idx.keys(): {self.hull_grouper.dsu.idx.keys()}")
                self.logger.loginfo(f"  >> self.sg.get_candidate_entities(): {self.sg.get_candidate_entities()}")
                if self.action == 'find':
                    unprocessed_eids = [eid for eid in current_eids
                                        if self.agg_results.results_by_entity.num_queries.get(eid, 0) <= 0]
                elif self.action == 'count':
                    if self.etypes == ['object']:
                        unprocessed_eids = [eid for eid in current_eids
                                        if self.agg_results.results_by_entity_count.num_queries.get(eid, 0) <= 0]
                    else:
                        unprocessed_eids = self.hull_grouper.get_low_count_eids(max_count=0)
                else:
                    raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")

                if len(unprocessed_eids) > 0:
                    self.current_agent_message = f"Group({current_gid}) sufficiently observed. Let's query them."
                    self.inference_signal_queue.put((current_gid, unprocessed_eids))
                    self.logger.loginfo(f"<navigate.4.1> Need inference of EIDs: {unprocessed_eids}. Put group({current_gid}) to inference_signal_queue.")
                    self.rr_log(f"<navigate.4.1> Need inference of EIDs: {unprocessed_eids}. Put group({current_gid}) to inference_signal_queue.", panel='nav')
                else:
                    self.logger.loginfo(f"<navigate.4.1> No valide EIDs. Entities in group({current_gid}) was already processed.")
                    self.rr_log(f"<navigate.4.1> No valide EIDs. Entities in group({current_gid}) was already processed.", panel='nav')

                if self.agent_pose:
                    agent_xy = np.array(self.agent_pose['position'][:2], dtype=float)
                    min_dist = float("inf")
                    min_gid = None
                    for gid, path_points in path_points_all.items():
                        if len(path_points) == 0:
                            continue
                        dist = min_distance(agent_xy, path_points[:, :2])
                        if dist < min_dist:
                            min_dist = dist
                            min_gid = gid
                    self.current_gid = min_gid
                    self.logger.loginfo(f"<navigate.4.2> Changed GID: {current_gid} -> {self.current_gid}")
                    self.rr_log(f"<navigate.4.2> Changed GID: {current_gid} -> {self.current_gid}", panel='nav')
                    if self.current_gid is not None:
                        self.current_agent_message = f"Now observe Group({current_gid})!"
                else:
                    self.logger.logwarn(f"<navigate.4.2> self.agent_pose is not available.")
                    self.rr_log(f"<navigate.4.2> self.agent_pose is not available.", panel='nav', level='warn')
        except Exception as e:
            self.logger.logerr(f"<navigate.4> Error occurs: {e}")
            self.rr_log(f"<navigate.4> Error occurs: {e}", panel='nav', level='error')

        # Visualize
        try:
            path_points_marker = make_marker_array_from_points(
                current_path_points, ns=f"path_points_{current_gid}", color=colors[current_gid], frame_id=self.frame_id)
            self.path_points_pub.publish(path_points_marker)
            self.logger.loginfo(f"<navigate.4.3> Publish /active_waypoints (path_points)")
            self.rr_log(f"<navigate.4.3> Publish /active_waypoints (path_points)", panel='nav')
        except Exception as e:
            self.logger.logerr(f"<navigate.4.3> Error occurs: {e}")
            self.rr_log(f"<navigate.4.3> Error occurs: {e}", panel='nav', level='error')

        try:
            marker_array_all = []
            for gid, path_points in path_points_all.items():
                if len(path_points) == 0:
                    continue
                path_points_marker = make_marker_array_from_points(
                    path_points, ns=f"path_points_{gid}", color=colors[gid], frame_id=self.frame_id)
                marker_array_all.extend(path_points_marker.markers)
            if marker_array_all:
                marker_array = MarkerArray(markers=marker_array_all)
                self.previous_path_points = marker_array
                self.path_points_vis_pub.publish(self.previous_path_points)
                self.is_path_points_updated = False
            self.logger.loginfo(f"<navigate.4.4> Save previous path_points for efficient visualization")
            self.rr_log(f"<navigate.4.4> Save previous path_points for efficient visualization", panel='nav')
        except Exception as e:
            self.logger.logerr(f"<navigate.4.4> Error occurs: {e}")
            self.rr_log(f"<navigate.4.4> Error occurs: {e}", panel='nav', level='error')

        # Set exploration strategy
        try:
            if len(current_path_points) > 0:
                is_running = self.active_clients.start()
                if is_running:
                    if self.active_clients.is_paused:
                        success = self.active_clients.resume()
                if is_running:
                    exp_strategy = "vg_first"
                else:
                    exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                self.logger.logrich(f"<navigate.4.5> Exp Strategy: {exp_strategy}", name='navigation')
                self.rr_log(f"<navigate.4.5> Exp Strategy: {exp_strategy}", panel='nav')
            else:
                if self.active_clients.is_running:
                    if len(marker_array_all) == 0:
                        if not self.active_clients.is_paused:
                            success = self.active_clients.pause()
                        self.path_points = None  # reset path_points
                        exp_strategy = "geometric_frontier"
                        self.exploration_strategy_pub.publish(String(exp_strategy))
                        self.logger.logrich(f"<navigate.4.5> Exp Strategy: Ended ({exp_strategy})", name='navigation')
                        self.rr_log(f"<navigate.4.5> Exp Strategy: Ended ({exp_strategy})", panel='nav')
                        return
                exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                self.logger.logrich(f"<navigate.4.5> Is Not Active ({exp_strategy})", name='navigation')
                self.rr_log(f"<navigate.4.5> Is Not Active ({exp_strategy})", panel='nav')
        except Exception as e:
            self.logger.logerr(f"<navigate.4.5> Error occurs: {e}")
            self.rr_log(f"<navigate.4.5> Error occurs: {e}", panel='nav', level='error')

    def start_coverage_planning(self):
        """
        Trigger coverage path planning exploration strategy.
        This method can be called by derived classes or external logic to initiate coverage planning.
        """
        self.logger.loginfo("Starting coverage path planning strategy")
        self.exploration_strategy_pub.publish(String('coverage_planning'))

# OTHERS
    def build_batch_for_gid(self, eids: List[int], num_queries_required: int):
        self.log(f"<build_batch_for_gid.0> EIDs: {eids}")
        self.rr_log(f"<build_batch_for_gid.0> EIDs: {eids}", panel='main')
        try:
            data, pids = [], []
            budget = max(0, int(num_queries_required))
            self.log(f"<build_batch_for_gid.1> Budget: {budget}")
            self.rr_log(f"<build_batch_for_gid.1> Budget: {budget}", panel='main')

            etype = 'object'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=1, max_kfs=10)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': keyframes, 'etype': etype, 'atype': 'object_box_id', 'eids': eids}]
                pids += [kf['id'][1] for kf in keyframes]
                budget -= 1
            self.log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")
            self.rr_log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})", panel='main')

            etype = 'all'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=1, max_kfs=10)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': keyframes, 'etype': etype, 'atype': 'object_box_id', 'eids': eids}]
                pids += [kf['id'][1] for kf in keyframes]
                budget -= 1
            self.log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")
            self.rr_log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})", panel='main')
        except Exception as e:
            self.log(f"<build_batch_for_gid.1> Error occurs: {e}", level='error')
            self.rr_log(f"<build_batch_for_gid.1> Error occurs: {e}", panel='main', level='error')

        try:
            etype = 'image'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=budget, max_kfs=3)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': keyframes, 'etype': 'image', 'atype': 'none', 'eids': eids}]
                pids += [kf['id'][1] for kf in keyframes]
                budget -= 1
            self.log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")
            self.rr_log(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})", panel='main')
        except Exception as e:
            self.logger.logerr(f"<build_batch_for_gid.2> Error occurs: {e}")

        return data, pids

if __name__ == "__main__":
    logger = Logger()

    SCENE = '2find'
    if SCENE == "arabic_room":
        instruction = "Find the pillow closest to the book on the stool."
        action = 'find'
        target_name = "pillow closest to the book on the stool"
        candidate_names, reference_names = ['pillow'], ['book', 'stool']
    elif SCENE in ['2find', '2find_2025-12-27-06-59-58']:
        instruction = "Find a red chair below the halloween poster"
        action = 'find'
        target_name = "red chair below the halloween poster"
        candidate_names, reference_names = ['chair'], ['doll']
    elif SCENE == '5count':
        instruction = "How many chairs does the doll sit on?"
        action = 'count'
        target_name = "chairs the doll sit on"
        candidate_names, reference_names = ['chair'], ['doll']
    elif SCENE == 'vla_test_2025-12-23-16-23-57':
        # instruction = "Find a blue chair between red chairs"
        # action = 'find'
        # target_name = "blue chair between red chairs"
        # candidate_names, reference_names = ['chair'], []
        instruction = "Find a red chair between blue chairs"
        action = 'find'
        target_name = "red chair between blue chairs"
        candidate_names, reference_names = ['chair'], []
    elif SCENE == 'vla_js_chair_2025-12-17-12-17-43':
        instruction = "Find the chair with a blue seat."
        action = 'find'
        target_name = "chair with a blue seat"
        candidate_names, reference_names = ['chair'], []
    else:
        raise TypeError(f"SCENE must be in ['office_1', 'hotel_room_1', 'chinese_room'], but {SCENE} was given.")

    # DATA_DIR = f"/ws/external/test_data/{SCENE}"
    DATA_DIR = f"/ws/data/demo/20251227/{SCENE}"
    MAP_DIR = os.path.join(DATA_DIR, "offline_map")
    KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")

    tester = BaseActiveVisualGrounder(
        logger=logger, action=action, target_name=target_name,
        candidate_names=candidate_names, reference_names=reference_names,
        use_ros=False
    )
    tester._init_all(logger=tester.logger, use_ros=False)

    class Task:
        def __init__(self, current_step, action, text_instruction):
            self.current_step = current_step
            self.start_time = time.time()
            self.action = action
            self.text_instruction = text_instruction
    class Graph:
        def __init__(self):
            self.nodes = []
    class Entity:
        def __init__(self, target_name=target_name):
            self.relation_graph = Graph()
            self.target_name = target_name

    class SubTask:
        def __init__(self, entity, action):
            self.entity = entity
            self.action = action

    subtask = SubTask(entity=Entity(target_name=target_name), action=action)
    task = Task(current_step=subtask, action=action, text_instruction=instruction)
    tester._set_task_callback(
        task, target_name=target_name,
        candidate_names=candidate_names, reference_names=reference_names
    )
    tester.node_active_signal = True

    map_dirs = [os.path.join(MAP_DIR, d) for d in os.listdir(MAP_DIR)
            if os.path.isdir(os.path.join(MAP_DIR, d))]
    map_dir_sorted = sorted(map_dirs, key=os.path.getmtime)

    for dir in map_dir_sorted:
        tester.spin_once(None, dir=dir)
        tester.navigation_loop(1)
        tester.inference_loop(1)
        # time.sleep(0.1)

    print("Done!")

import os
import sys
sys.path.append('/ws/external/')
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

try:
    import rospy
except:
    sys.path.append("/ws/external/ai_module/src/utils/debug")
    import ai_module.src.utils.debug
    import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Int32, Empty
from visualization_msgs.msg import Marker, MarkerArray
from visual_grounding.srv import SetSubplans, SetSubplansResponse
from std_srvs.srv import Trigger, TriggerResponse

ANSWER_TYPE = {'find': Marker, 'count': Int32}

class EntityType:
    OBJECT = "object"
    DETECTION = "detection"
    IMAGE = "image"
    @classmethod
    def values(cls): return {cls.OBJECT, cls.DETECTION, cls.IMAGE}


make_error_etype = lambda etype: f"entity_type must be in {list(EntityType.values())}, but {etype} was given."


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
    def __init__(self, node_name=None, is_real_world=False, *args, **kwargs):
        self._dispatcher = PriorityDispatcher(name=node_name, normal_workers=0)
        super().__init__(*args, **kwargs)
        self.updated_resource = False
        self.answer = ""
        self.ready = False
        self.answer_result = None
        self.node_name = node_name if node_name else rospy.get_name()

        self.time_limit = rospy.Duration(600)  # (sec)
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
        self.prompt_renderer = None
        self.system_instruction_renderer = None

        """ Subtask """
        self.subtask = None
        rospy.Service(self.node_name + "/set_subplans", SetSubplans, self._set_task)
        
        """ Real World """        
        self.is_real_world = is_real_world # TODO
        self.frame_id = "world" if self.is_real_world else "map"

        """ InferenceResults """
        self.kf_counts = {}  # {kf_id: count}
        self.aggregated_results_cfg = {
            'min_query': 5,
            'inference_cfg': {
                'method': 'logit_pool',
                'prior': 0.5,
                'keep_top_k': 4,
            }
        }  # TODO: Need to tune

        """ VLM """
        self.max_llm_concurrency = getattr(self, "max_llm_concurrency", 2)  # 필요시 조절
        self._llm_sema = threading.Semaphore(self.max_llm_concurrency)

    def _init_all(self, *args, **kwargs):
        self._init_vars(*args, **kwargs)
        self._init_services(*args, **kwargs)
        self._init_clients(*args, **kwargs)
        self._init_subscribers(*args, **kwargs)
        self._init_publishers(*args, **kwargs)
        self._init_threads(*args, **kwargs)

    def _init_vars(self, *args, **kwargs) -> None:
        """ Initialize variables """
        self.ready = False
        self.node_active_signal = False
        self.inference_signal_queue = queue.Queue(maxsize=3)
        self.start_time = None
        self.node_active_signal = False

    def _init_services(self, *args, **kwargs) -> None:
        self.sg_lock = threading.Lock()
        if not 'logger' in kwargs:
            kwargs.update({'logger': self.logger})
        self.scene_graph_clients = SceneGraphClients(**kwargs)
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

    def _init_publishers(self, use_ros=True, *args, **kwargs):
        # Visualizer
        if use_ros:
            self.marker_pub = rospy.Publisher("/visual_grounding/markers", Marker, queue_size=50)
            self.marker_pub_orig = rospy.Publisher("/visual_grounding/markers_orig", Marker, queue_size=50)

    def _init_threads(self, *args, **kwargs) -> None:
        """ Initialize threads """
        # Inference
        self.inference_queue = queue.Queue(maxsize=10)

        # Aggregated results
        self.agg_results = AggregatedResult(**self.aggregated_results_cfg)
        self.agg_results_lock = threading.Lock()

        # Validate
        self.main_running = threading.Event()
        self.validation_running = threading.Event()

        # Answer
        self.answer_result = None

        # Time
        self.processing_start_time = 0.0

    @property
    def confidence_threshold(self):
        action = self.action
        if action == 'find':    return (0.40, 0.70)
        elif action == 'count': return (0.20, 0.70)
        else: return (0.20, 0.50)

    def _reset_vars(self):
        self._init_vars()
        self.inference_queue = queue.Queue(maxsize=10)
        self.agg_results = AggregatedResult(**self.aggregated_results_cfg, action=self.action)

        self.main_running = threading.Event()
        self.validation_running = threading.Event()
        self.answer = ""
        self.answer_result = None
        self.ready = False
        self.subtask = None

    def _set_task(self, req, *args, **kwargs):
        if self.status == Status.WAITING:
            subtask = req.current_step
            self.start_time = req.start_time
            relation_graph = subtask.entity.relation_graph

            related_names = []
            candidate_names = []
            for node in relation_graph.nodes:
                related_names.append(node.name)
                if node.is_target:
                    if node.name == 'path':
                        for edge in relation_graph.edges:
                            if edge.source_id == node.id:
                                target_ids = edge.target_ids
                                for _node in relation_graph.nodes:
                                    if _node.id in target_ids:
                                        candidate_names.append(_node.name)
                    candidate_names.append(node.name)
            self.related_names = list(set(related_names))
            self.candidate_names = list(set(candidate_names))
            self.reference_names = list(set(related_names) - set(candidate_names))
            self.subtask = subtask
            self.agg_results.action = self.action

            self.logger.loginfo(f"================================================")
            self.logger.logrich(f"Instruction: \"{req.text_instruction}\"", name='instruction')
            self.logger.logrich(f"Action: \"{subtask.action}\"", name='action')
            self.logger.logrich(f"Target Name: \"{subtask.entity.target_name}\"", name='target_name')
            self.logger.loginfo(f"Candidate names: {self.candidate_names}")
            self.logger.loginfo(f"Reference names: {self.reference_names}")
            self.logger.loginfo(f"Related names: {self.related_names}")
            return SetSubplansResponse(success=True, message=self.status)
        else:
            return SetSubplansResponse(success=False, message=self.status)

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

    def _reset_callback(self, req):
        self._reset_vars()
        
        self.logger.logrich(f"Instruction: ", name='instruction')
        self.logger.logrich(f"Action: ", name='action')
        self.logger.logrich(f"Target Name: ", name='target_name')
        self.logger.logrich(f"Inference: ", name='inference')
        self.logger.log("Visual grounding node has been reset.")
        
        return TriggerResponse(success=True, message="Visual grounding node has been reset.")

    def _status_callback(self, req):
        status = self.status
        # self.logger.log(f"Current status: {status}")
        return TriggerResponse(success=True, message=status.value)


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
        return self.subtask.action

    @property
    def etypes(self):
        action = self.subtask.action
        if action == 'find':
            return ['object'] # , 'image']
        elif action == 'count':
            return ['image']  # 'object',
        else:
            return ['all']

    def update_resource(self, **kwargs):
        self.scene_graph_clients.update_scene_graph(**kwargs)

        styles = {
            # 'reference': {'show': True, 'color': 'blue'},
            'candidate': {'show': True, 'color': 'green'},
        }
        with self.sg_lock:
            for etype in self.etypes:
                self.sg.keyframes.annotate(styles, node_name=self.node_name, suffix=f"_annotated_global_{etype}", etype=etype)
        self.updated_resource = True

    def standby(self, **kwargs):
        self.scene_graph_clients.start(
            candidate_names=self.candidate_names,
            reference_names=self.reference_names,
            **kwargs
        )

        action = self.action
        self.answer_pub = rospy.Publisher('/answer', ANSWER_TYPE.get(action, String), queue_size=1)

        """ LLM Client """
        self.default_inference_options['prompt'].update({'action': action, 'rtype': 'inference'})
        self.default_inference_options['image']['suffix'] = '_annotated_global'
        self.default_validate_options['prompt'].update({'action': action, 'rtype': 'validate'})
        self.default_validate_options['image']['suffix'] = '_annotated_inference'

        self.prompt_renderer = PromptRenderer(description=self.target_name)
        self.system_instruction_renderer = SystemInstructionRenderer()

        self.ready = True

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
                self.inference_queue.put(data)
                # --- 예약 수 반영 ---
                self.inference_results.schedule(gid, len(data), data=data)
                self.logger.loginfo(f"<process.6> Put data to inference_queue: {data}")
            else:
                self.logger.loginfo(f"<process.6> No data to put: {data}")
        except Exception as e:
            self.logger.logerr(f"<process.6> Error occurs: {e}")

        self.logger.logrich(f"Selected keyframe: #object({num_kfs.get('object', 0)}), #detection({num_kfs.get('detection', 0)}), #image({num_kfs.get('image', 0)})", name="selected_keyframe")

    def timer_callback(self, event, **kwargs):
        self.logger.logrich(f"Status: {self.status} | #inference_queue={len(self.inference_queue.queue)}", name='status')
        self.logger.logrich(f"Answer: {self.answer} | MinQuery: {self.agg_results.min_query}", name='answer')

        if self.status == Status.STANDBY:
            self.standby()

        if self.status == Status.PROCESSING:
            self.logger.logrich(f"Status: {self.status} | #inference_queue={len(self.inference_queue.queue)}", name='status')
            self.process(**kwargs)

        if self.status == Status.COMPLETED:
            self.answer_the_question(self.answer_result)

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
            self.timer_callback(None)
            self.main_running.clear()
            self.logger.loginfo(f"<main_loop> Main is cleared. Let's sleep..")
            rate.sleep()

    def _answer_impl(self, answer, block=False):
        if answer is None:
            self.logger.logwarb(f"<answer_the_question> answer is None")
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
                    answer_entity = self.sg.entities.get_single(eid)
                    self.logger.loginfo(f"<answer_the_question.5.2> answer_entity: {answer_entity}")

                    # Refinement
                    self.logger.loginfo(f"<answer_the_question.5.3> Refinement...")
                    eid2pids = self.sg.keyframes.entity_id2place_ids
                    pids_answer = eid2pids[answer_entity.id]
                    kfs_answer = self.sg.keyframes.get(pids_answer)
                    initial_bbox_3d = answer_entity.corners_3d

                    self.logger.loginfo(f"<answer_the_question.5.4> Let's minimize")
                    refined_result = minimize(
                        refine_bbox,
                        initial_bbox_3d,
                        args=(answer_entity, kfs_answer,),
                        method='Nelder-Mead',
                        options={'disp': True}
                    )
                    refined_point_3d = refined_result.x.reshape(-1, 3)

                    self.logger.loginfo(f"<answer_the_question.5.5> point_3d_to_marker:\n"
                                        f"  > initial_bbox_3d: {initial_bbox_3d}\n"
                                        f"  > refined_point_3d: {refined_point_3d}\n")
                    marker_orig = point_3d_to_marker(initial_bbox_3d, eid, color=color, style='box')
                    marker = point_3d_to_marker(refined_point_3d, eid, color=color, style='box')
                    answer_msg = point_3d_to_marker(refined_point_3d, eid, color=color, style='cube')

                    self.logger.loginfo(f"<answer_the_question.5.6> Publish Answer...")
                    self.answer_pub.publish(answer_msg)
                    self.logger.loginfo(f"<answer_the_question.5.6> Publish...")
                    self.marker_pub.publish(marker)
                    self.marker_pub_orig.publish(marker_orig)
                    self.logger.loginfo(f"<answer_the_question.5.6> Publish... Done")
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

    def answer_the_question(self, answer, block=False):
        return self._dispatcher.submit_high(self._answer_impl, answer, block=block)

    def load_and_preprocess_image(self, image_path, preprocess=None):
        image = Image.open(image_path)

        if preprocess == "crop":
            # TODO: Implement crop preprocessing
            pass

        return image

    def get_images(self, keyframes, suffix="", preprocess=None, **kwargs):
        if isinstance(keyframes, Keyframe):
            keyframes = Keyframes({keyframes.id: keyframes})

        images, image_paths = [], []

        # Normalize suffix to always be a list for consistent processing
        suffixes = suffix if isinstance(suffix, list) else [suffix]

        for keyframe in keyframes.values():
            for s in suffixes:
                if s == "":
                    image_path = keyframe.image_path
                else:
                    image_path = keyframe.save_path(node_name=self.node_name, suffix=s)

                try:
                    image = self.load_and_preprocess_image(image_path, preprocess)
                except Exception as e:
                    self.logger.logerr(f"<get_images> Error occurs: {e}\n"
                                       f"  > Failed to load and preprocess image: {image_path}\n"
                                       f"  > keyframe: {keyframe}\n"
                                       f"  > suffixes: {suffixes}\n")
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
            self.logger.loginfo(f"<query_worker.0> Get client: {client_counter % len(self.clients)}")

            phase = 1
            keyframes = input_data['keyframes']
            if isinstance(keyframes, Keyframe): candidate_eids_in_kfs = keyframes.entities['candidate'].ids  # image
            else: candidate_eids_in_kfs = reduce(operator.add, [kf.entities['candidate'].ids for pid, kf in keyframes.items()])  # object, detection

            options = input_data.get('options', self.default_options)
            previous_history = options['prompt']['previous_history']  # TODO

            # Prepare the prompt and system instruction
            prompt = self.prompt_renderer.render(**options['prompt'], anno_ids=candidate_eids_in_kfs)
            system_instruction = self.system_instruction_renderer.render(**options['prompt'])
            images, image_paths = self.get_images(keyframes, **options['image'])
            self.logger.loginfo(f"<query_worker.1> Prepare the input data")

            # Query the model
            phase = 2
            start_time = time.time()
            with self.sg_lock:
                message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
            end_time = time.time()
            self.logger.loginfo(f"<query_worker.2> Construct message with client")

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
                self.logger.loginfo(f"<query_worker.3> Get response")
            else:
                self.logger.logwarn(f"<query_worker.3> Failed to parse. response_text (truncated):\n"
                                    f"{str(response_text)[:1000]}")

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
            self.logger.loginfo(result_text)
            self.logger.loginfo(f"<query_worker.4> Print the log")

            # Verify
            phase = 5
            if self.action in ['find']:
                if len(target_ids) > 1:
                    self.logger.logwarn(f"Object ids mismatch: len(target_ids)={len(target_ids)} != 1")
                    return None
            else:
                if self.default_inference_options['prompt']['action'] == 'follow_between' \
                        and self.default_inference_options['prompt']['rtype'] == 'inference' \
                        and len(target_ids) != 2:
                    self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 2")
                    return None
                elif self.default_inference_options['prompt']['action'] == 'find' \
                        and self.default_inference_options['prompt']['rtype'] == 'inference' \
                        and len(target_ids) != 1:
                    self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(target_ids)} != 1")
                    return None
            self.logger.loginfo(f"<query_worker.5> Verify the response")
            return response
        except Exception as e:
            self.logger.logerr(f"<query_worker.{phase}> Error occurs: {e}")

    def inference(self, keyframes, etype='all', atype='object_box_id', client_counter=0, gid=None, *args, **kwargs):
        try:
            if isinstance(keyframes, dict):
                self.logger.logwarn(f"<inference.1> keyframes is dictionary. Use keyframes['keyframes']")
                keyframes = keyframes['keyframes']
            if len(keyframes) == 0:
                self.logger.logwarn(f"<inference.1> No keyframes input was given.")
                return
            self.logger.loginfo(f"<inference.1> Annotation type is {atype}; Entity type is {etype};")
        except Exception as e:
            self.logger.logerr(f"<inference.1> Error occurs: {e}")

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
                oldest_id = max(keyframes.ids)
                keyframes = keyframes.get_single(oldest_id)

            suffix = options['image']['suffix']
            options['image'].update({'suffix': f"{suffix}_{etype}"})
            data_list = [{
                'keyframes': keyframes,  # TODO: need to check
                'options': options,
            }]
            self.logger.loginfo(f"<inference.2> options and data_list are ready.")
        except Exception as e:
            self.logger.logerr(f"<inference.2> Error occurs: {e}")

        try:
            self.logger.loginfo(f"<inference.3> GID={gid}")
            response = self.query_worker(data_list[0], client_counter=client_counter)
            if response is None:
                self.logger.logwarn(f"<inference.3> response is None.")
                return
            self.logger.loginfo(f"<inference.3> Get response from worker({client_counter})")
        except Exception as e:
            self.logger.logerr(f"<inference.3> Error occurs: {e}")

        # Save the result
        try:
            output = {
                'entity_type': etype,
                'data': data_list[0],
                'target_ids': response.get('target_ids'),
                'reason': response.get('reason'),
            }
            self.logger.loginfo(f"<inference.4> Set inference_ready_event, which is not used.")
        except Exception as e:
            self.logger.logerr(f"<inference.4> Error occurs: {e}")
        return output

    def inference_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            try:
                if self.start_time:
                    now = rospy.Time.now()
                    elapsed = now - self.start_time
                    remaining_time = (self.start_time + self.time_limit) - now
                else:
                    remaining_time = rospy.Duration(60)
                    elapsed = rospy.Time.now()
                self.logger.logrich(f"<inference_loop.1> Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)", name='time')
                if self.status != Status.PROCESSING:
                    rate.sleep()
                    self.logger.loginfo(f"<inference_loop.1> Let's sleep...")
                    continue
                self.logger.loginfo(f"<inference_loop.1> Let's inference!!")
            except Exception as e:
                self.logger.logerr(f"<inference_loop.1> Error occurs: {e}")

            try:
                with self.agg_results_lock:
                    agg_results = self.agg_results.snapshot()
                self.logger.logrich(f"<inference_loop.2> AggResults: {self.agg_results}", name='agg_results')
            except Exception as e:
                self.logger.logerr(f"<inference_loop.2> Error occurs: {e}")

            # Ready to answer?
            try:
                (thres_low, thres_high) = self.confidence_threshold

                best_confidence = agg_results.get('best_confidence')
                enough_observation = (self.exploration_status == 'no_frontier')

                if self.start_time:
                    now = rospy.Time.now()
                    elapsed = now - self.start_time
                    remaining_time = (self.start_time + self.time_limit) - now
                else:
                    remaining_time = rospy.Duration(60)
                    elapsed = rospy.Time.now()
                ready_to_answer = (((best_confidence > thres_high) and enough_observation)
                                   or (remaining_time <= rospy.Duration(30)))  # (sec)
                self.logger.logrich(f"<inference_loop.3.2> Time: {int(elapsed.to_sec())}/{int(self.time_limit.to_sec())} (sec)  |  Best Conf: {best_confidence:.2f}  |  Exp Status: {self.exploration_status}", name='time')
                # timeout = 10분
                if ready_to_answer:
                    self.answer_result = self.agg_results.best_answer  # TODO
                    self.answer_the_question(self.answer_result)
                    self.logger.loginfo(f"<inference_loop.3.2> Answer the final result. Confidence: {best_confidence} >= {thres_high}.")
                    return
                else:
                    self.logger.loginfo(f"<inference_loop.3.2> Let's inference. Confidence: {best_confidence} < {thres_high}.")
            except Exception as e:
                self.logger.logerr(f"<inference_loop.3.1&2> Error occurs: {e}")

            # Inference
            try:
                for _ in range(self.inference_queue.qsize()):
                    jobs = self.inference_queue.get_nowait()
                    for item, result in self.run_parallel(self.inference, jobs, max_workers=self.max_workers):
                        gid = item.get('gid')
                        if result is None:
                            self.logger.loginfo(f"<inference_loop.4.2.{_}> result is None.")
                            continue

                        try:
                            etype = result.get('entity_type')
                            target_ids = result.get('target_ids', [])
                            if self.action == 'count':
                                count = len(target_ids)
                                if count == 0:
                                    answer = None
                                else:
                                    answer = Answer(count=count, data=result['data'])
                                self.logger.loginfo(f"<inference_loop.4.3.{_}> Answer(count={count})")
                            elif self.action == 'find':
                                if len(target_ids) > 1:
                                    self.logger.logwarn(f"<inference_loop.4.3.{_}> #target_ids={len(target_ids)} > 1")
                                if len(target_ids) == 0:
                                    answer = None
                                    self.logger.loginfo(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: X")
                                else:
                                    target_id = int(target_ids[0])
                                    # candidate_entities = self.sg.get_candidate_entities('all')
                                    target_entity = self.sg.entities.get_single(target_id)
                                    answer = Answer(object=target_entity, data=result['data'])
                                    self.logger.loginfo(f"<inference_loop.4.3.{_}> Answer: {answer};  target_entity: {target_entity}")
                            else:
                                raise NotImplementedError(f"action must be in ['count'], but {self.action} was given.")
                        except Exception as e:
                            self.logger.logerr(f"<inference_loop.4.3.{_}> Error occurs: {e}")

                        try:
                            if answer is not None:
                                self.agg_results.update(gid=gid, answer=answer, confidence=get_confidence(etype))
                                self.logger.loginfo(f"<inference_loop.4.4.{_}> Update agg_results <- {answer}")
                            else:
                                self.logger.loginfo(f"<inference_loop.4.4.{_}> No updated agg_results")
                        except Exception as e:
                            self.logger.logerr(f"<inference_loop.4.4.{_}> Error occurs: {e}")
                        finally:
                            # --- 예약 해제 (성공/실패 무관 1건) ---
                            if (gid is not None) and (answer is not None):
                                self.agg_results.release(gid, 1, eids=answer.eids)
                                self.agg_results.inc_queries(gid, 1, eids=answer.eids)
                            self.logger.loginfo(f"<inference_loop.4.4.{_}> Release group({gid})")
                            self.logger.logrich(f"<inference_loop.4.5.{_}> AggResults: {self.agg_results}", name='agg_results')

            except Exception as e:
                self.logger.logerr(f"<inference_loop.4> Error occurs: {e}")
            rate.sleep()

    def select_keyframes(
            self, entity_type='object', w_cov=1.0, w_area=1.0, alpha=0.5, target_eids=None,
            min_kfs=None, max_kfs=10, iter_margin=5, *args, **kwargs
    ):
        with self.sg_lock:
            sg = self.sg
        etype = 'all' if entity_type == 'image' else entity_type

        # --- candidate ids 준비 ---
        try:
            if target_eids is None:
                target_eids = set(sg.get_related_entities(etype).ids)
            else:
                target_eids = set(target_eids)
            if not target_eids:
                self.logger.logwarn(f"<select_keyframes.1> target_eids is empty.")
                return sg.keyframes.get([])
            self.logger.loginfo(f"<select_keyframes.1> target_eids: {target_eids}")
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.1> Error occurs: {e}")

        # --- min/max 보정 ---
        max_kfs = max(0, int(max_kfs))
        min_kfs = 0 if min_kfs is None else max(0, min(int(min_kfs), max_kfs))

        kfs = sg.keyframes
        # ---------- 주어진 target entities를 포함하는 keyframes를 구성: kfs_with_targets, pid2target_eids ----------
        try:
            eid2pids = kfs.entity_id2place_ids
            pid2eids = kfs.place_id2entity_ids

            num_places = len(kfs)
            num_target_entities = len(target_eids)
            use_pid2eids = num_target_entities > max(1, num_places // 8)

            kfs_with_targets = {}
            pid2target_eids = defaultdict(set)
            available_pids = set(kfs.keys())
            if not use_pid2eids:
                for target_eid in target_eids:
                    for pid_with_target in eid2pids.get(target_eid, ()):
                        if pid_with_target not in available_pids:
                            self.logger.logwarn(f"<select_keyframes.3> Warning occurs: PID({pid_with_target}) is not in available PIDs: {available_pids}")
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
            self.logger.loginfo(f"<select_keyframes.2> Target EIDs per each kf: {', '.join([f'{k}: {v}' for k, v  in pid2target_eids.items()])}")

            if len(kfs_with_targets) < min_kfs:
                self.logger.logwarn(
                    f"<select_keyframes.2> #kfs_with_targets={len(kfs_with_targets)} < min_kfs={min_kfs}. Skip.")
                return sg.keyframes.get([])  # Early Stop
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.2> Error occurs: {e}")

        # Cache: Area
        target_entities = sg.get_related_entities(etype).get(target_eids)
        per_obj_area = defaultdict(dict)  # {pid: {eid: area}, ...}

        def _entity_area(kf, pid, eid):
            d = per_obj_area[pid]
            if eid in d:
                return d[eid]
            tgt_ent = target_entities.get_single(eid)
            if tgt_ent is None:
                d[eid] = 0.0
                return 0.0
            tgt_bbox = tgt_ent.get_bbox(pose=kf.pose, image_size=kf.image_size, is_real_world=kf.is_real_world, kf_id=kf.id)
            d[eid] = float(tgt_bbox.area)
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
                    self.logger.logwarn(f"<select_keyframes.4> iter_cap reached. Bail out.")
                    break

                num_uncovered_tgts = len(uncovered_target_eids)
                max_area = 0.0
                tmp_stats = {}  # {pid: (c, a, covered_eids_now), ...}
                for pid, kf in kfs_with_targets.items():
                    covered_eids_now = pid2target_eids.get(pid, ()) & uncovered_target_eids
                    if not covered_eids_now:
                        continue
                    c = len(covered_eids_now) / num_uncovered_tgts
                    a = 0.0
                    for eid in covered_eids_now:
                        a += _entity_area(kf, pid, eid)
                    if a > max_area:
                        max_area = a
                    tmp_stats[pid] = (c, a, covered_eids_now)

                if not tmp_stats:
                    self.logger.logwarn(f"<select_keyframes.4> tmp_stats is None")
                    break

                best_pid, best_score = None, float("-inf")
                for pid, (c, a, _) in tmp_stats.items():
                    base = w_cov * c + (w_area * (a / max_area) if max_area > 0 else 0.0)
                    cnt = self.kf_counts.get(pid, 0)  # TODO
                    seen = 1.0 / (1.0 + alpha * cnt)
                    score = base * seen
                    if score > best_score:
                        best_score, best_pid = score, pid

                    if best_pid is None:
                        self.logger.logwarn(f"<select_keyframes.4> best_pid is None")
                        break

                selected_pids.append(best_pid)
                _, _, covered_eids_best = tmp_stats[best_pid]
                uncovered_target_eids.difference_update(covered_eids_best)
                self.kf_counts[best_pid] = self.kf_counts.get(best_pid, 0) + 1
                kfs_with_targets.pop(best_pid, None)
                pid2target_eids.pop(best_pid, None)

            self.logger.loginfo(f"<select_keyframes.3> selected_pids: {selected_pids}")
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.3> Error occurs: {e}")

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
            self.logger.loginfo(f"<select_keyframes.4> selected_pids: {selected_pids}")
        except Exception as e:
            self.logger.logerr(f"<select_keyframes.4> Error occurs: {e}")

        return sg.keyframes.get(selected_pids)
    
    @staticmethod
    def multi_thread_process(func, input_data, max_workers=3):
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
                fut = executor.submit(func, item, client_counter=client_counter)
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
                    new_fut = executor.submit(func, next_item, client_counter=client_counter)
                    in_flight[new_fut] = next_item
                    client_counter += 1


class BaseActiveVisualGrounder(BaseVisualGrounder):
    def __init__(self, *args, **kwargs):
        self.min_point_spacing = 0.5
        self.radius = 0.55 # (m)
        self._empty_path_since = {}  # {gid: rospy.Time}
        self._empty_path_cooldown = rospy.Duration(3.0)  # 3초
        super().__init__(*args, **kwargs)

        """ Navigation """
        self.agent_pose = None
        self.hull_grouper = None
        self.path_points = None
        self.is_path_points_updated = False
        self.navigation_running = threading.Event()
        self.navigation_lock = threading.RLock()

        self.last_update_time_path_points = rospy.Time.now()
        self.update_interval_path_points = rospy.Duration(5.0)  # (sec)
        self.current_gid = 0

    def _init_services(self, *args, **kwargs) -> None:
        super()._init_services(*args, **kwargs)
        self.active_clients = ActiveClients(*args, **kwargs)

    def _init_subscribers(self, *args, **kwargs):
        super()._init_subscribers(*args, **kwargs)
        self.traversable_points = None
        self._traversable_lock = threading.RLock()
        self.path_xy = np.zeros((0, 2), dtype=float)
        self.exploration_status = None

        self.occupancy_grid = None

        self.traversable_area_sub = rospy.Subscriber(
            "/traversable_area_filtered", PointCloud2, self._traversable_area_callback, queue_size=10)
        self.robot_path_sub = rospy.Subscriber("/path_recorder/path", Path, self._robot_path_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=20)
        self.occupancy_grid_sub = rospy.Subscriber("/occupancy_map", OccupancyGrid, self._occupancy_grid_callback, queue_size=1)
        self.exploration_status_sub = rospy.Subscriber("/instruction_following_exp_status", String, self._exploration_status_callback, queue_size=1)

        self.timeout_sub = rospy.Subscriber(
            "/timeout", Empty, self._timeout_callback, queue_size=10)

    def _reset_vars(self):
        super()._reset_vars()
        self.path_points = None
        self.navigation_running = threading.Event()
        self.active_clients.end()

    def _timeout_callback(self, msg) -> None:
        if self.status != Status.COMPLETED:
            self.logger.loginfo("<timeout_callback> Timeout signal received #############")
            self.answer_result = self.agg_results.best_answer
            self.answer_the_question(self.answer_result)
        else:
            self.answer_the_question(self.answer_result)
            self.logger.loginfo("<timeout_callback> Finished :) #############")
        return

    def _traversable_area_callback(self, msg) -> None:
        if self.traversable_points is None:
            traversable_pts, _ = pointcloud2_to_xy_array(msg)
            with self._traversable_lock:
                self.traversable_points = traversable_pts

    def _init_publishers(self, *args, **kwargs):
        super()._init_publishers(*args, **kwargs)
        self.exploration_strategy_pub = rospy.Publisher("/exploration_strategy", String, queue_size=1)
        self.path_points_pub = rospy.Publisher("/active_waypoints", MarkerArray, queue_size=1)
        self.path_points_vis_pub = rospy.Publisher("/active_waypoints_vis", MarkerArray, queue_size=1)
        self.previous_path_points = None

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

    def _odom_callback(self, msg):
        self.agent_pose = {
            "position": np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]),
            "orientation": np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]),
        }
        if self.debug:  # TODO: debug: Save the path_xy
            _ = save_path_xy(self.agent_pose['position'], base_dir="/ws/external/offline_map", name="agent_pose")

    def _occupancy_grid_callback(self, msg):
        self.occupancy_grid = CustomOccupancyGrid(msg)
        if self.debug:
            filename = f"occupancy_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            save_path = os.path.join("/ws/external/offline_map/", filename)
            self.occupancy_grid.save_npz(save_path)

    def _exploration_status_callback(self, msg):
        self.exploration_status = msg.data
        self.logger.loginfo(f"Exploration status: {self.exploration_status}")
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

    def build_batch_for_gid(self, eids: List[int], num_queries_required: int):
        self.logger.loginfo(f"<build_batch_for_gid.0> EIDs: {eids}")
        try:
            data, pids = [], []
            budget = max(0, int(num_queries_required))
            self.logger.loginfo(f"<build_batch_for_gid.1> Budget: {budget}")

            etype = 'object'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=1, max_kfs=10)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': keyframes, 'etype': etype, 'atype': 'object_box_id', 'eids': eids}]
                pids += keyframes.ids
                budget -= 1
            self.logger.loginfo(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")

            etype = 'all'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=1, max_kfs=10)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': keyframes, 'etype': etype, 'atype': 'object_box_id', 'eids': eids}]
                pids += keyframes.ids
                budget -= 1
            self.logger.loginfo(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")
        except Exception as e:
            self.logger.logerr(f"<build_batch_for_gid.1> Error occurs: {e}")

        try:
            etype = 'image'
            while budget > 0 and (etype in self.etypes):
                keyframes = self.select_keyframes(entity_type=etype, target_eids=eids, min_kfs=budget, max_kfs=3)
                if len(keyframes) == 0:
                    break
                data += [{'keyframes': kf, 'etype': 'image', 'atype': 'none', 'eids': eids} for kf in keyframes.to_list()]
                pids += keyframes.ids
                budget -= 1
            self.logger.loginfo(f"<build_batch_for_gid.1> Budget({etype}): {budget} ({'ok' if etype in self.etypes else 'no'})")
        except Exception as e:
            self.logger.logerr(f"<build_batch_for_gid.2> Error occurs: {e}")

        return data, pids

    def process(self, **kwargs):
        self.logger.loginfo(f"<process.0> Start")
        self.update_resource(**kwargs)

        # Select Group ID
        num_queries_required = self.agg_results.min_query
        try:
            (gid, eids) = self.inference_signal_queue.get_nowait()
            # self.agg_results.generate(gid=gid)  # 모든 related_entity는 어떤 group에 할당됨
            self.logger.loginfo(f"<process.1> New EIDs are given: {eids} (GID={gid})")
        except queue.Empty:
            if self.action == 'find':
                candidate_eids = self.sg.get_candidate_entities('object').ids  # TODO:
                pending_eids = sorted(
                    [
                        eid for eid in candidate_eids
                        if self.agg_results.results_by_entity.num_queries.get(eid, 0) < self.agg_results.min_query
                    ],
                    key=lambda eid: self.agg_results.results_by_entity.num_queries.get(eid, 0)
                )
                gid = None
            elif self.action == 'count':
                pending_gids = self.agg_results.get_pending_ids()  # GIDs
                if len(pending_gids) == 0:
                    self.logger.loginfo(f"<process.1> There is no pending_gids")
                    return
                gid = pending_gids[0]
                pending_eids = self.agg_results.results.get(gid, set())
                self.logger.loginfo(f"<process.1> There are pending_gids: {gid} (pending_eids:{pending_eids})")
            else:
                self.logger.logerr(f"<process.1> Error occurs: self.action must be in ['find', 'count'], bug {self.action} was given.")

            if len(pending_eids) == 0:
                self.logger.loginfo(f"<process.1> Don't need to more inference")
                return
            eids = pending_eids
            self.logger.loginfo(f"<process.1> Pending EIDs are selected: {eids}")

        # Get Keyframes of the group gid
        try:
            # eids = self.hull_grouper.groups(gid)
            candidate_eids = self.sg.get_candidate_entities().ids
            candidate_eids_in_group = list(set(eids) & set(candidate_eids))
            if len(candidate_eids_in_group) == 0:
                data, pids = [], []
            else:
                data, pids = self.build_batch_for_gid(
                    eids=candidate_eids_in_group, num_queries_required=num_queries_required
                )
            self.logger.loginfo(f"<process.2> Selected KFs: {list(set(pids))}")
        except Exception as e:
            self.logger.logerr(f"<process.2> Error occurs: {e}")

        # Put the data to inference_queue
        try:
            if len(data) > 0:
                try:
                    self.inference_queue.put_nowait(data)
                    self.agg_results.schedule(gid, n=len(data), data=data)
                    self.logger.loginfo(
                        "<process.3> Put data to inference_queue:\n" +
                        "\n".join([f"  > pids: {', '.join(map(str, d['keyframes'].ids))}, etype: {d['etype']}, atype: {d['atype']}, eids: {d['eids']}" for d in data])
                    )
                except queue.Full:
                    self.logger.logwarn("<process.3> inference_queue is full. Will retry next tick.")
            else:
                self.logger.loginfo(f"<process.3> No data to put: {data}")
        except Exception as e:
            self.logger.logerr(f"<process.3> Error occurs: {e}")

    def update_path_points(self) -> None:
        # TODO: Detection-based update_path_points
        try:
            with self.sg_lock:
                related_objects = self.sg.get_related_entities('object')
            self.logger.loginfo(f"<update_path_points.1> #related_objects = {len(related_objects)}")
        except Exception as e:
            self.logger.logerr(f"<update_path_points.1> Error occurs: {e}")

        try:
            group_hulls_bef = None
            if self.hull_grouper is None:
                self.hull_grouper = GridGrouper(threshold=0.5).fit(related_objects)
            else:
                group_hulls_bef = self.hull_grouper.group_hulls()
                self.hull_grouper.update(related_objects)

            now = rospy.Time.now()
            updated_time_diff = now - self.last_update_time_path_points
            group_hulls = self.hull_grouper.group_hulls()

            is_equal_group_hulls = is_equal(group_hulls_bef, group_hulls)
            if is_equal_group_hulls and (updated_time_diff < self.update_interval_path_points):
                self.logger.loginfo(f"<update_path_points.2> No update group_hulls: #GIDs={len(group_hulls)}")
                return
            self.logger.loginfo(f"<update_path_points.2> Updated group_hulls: #GIDs={len(group_hulls)}")
        except Exception as e:
            self.logger.logerr(f"<update_path_points.2> Error occurs: {e}")

        try:
            path_points, log_data = {}, {}
            for _, group in enumerate(group_hulls):
                hull_xy = group['hull']
                gid = group['gid']

                nearest_points = find_closest_point(hull_xy, self.traversable_points)
                if nearest_points.shape[1] == 2:
                    nearest_points = np.hstack([nearest_points, np.zeros((len(nearest_points), 1))])

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
                self.logger.loginfo(f"<update_path_points.3.1.{_}> Updated path_points for GID({gid}): #={len(current_path_points)}")

            self.path_points = path_points
            self.is_path_points_updated = True
            self.last_update_time_path_points = now
            self.logger.logrich(f"<update_path_points.3> path_points (#valid/#total): {{{', '.join([f'{gid}: ({valid}/{total})' for gid, (valid, total) in log_data.items()])}}}", name="path_points")
        except Exception as e:
            self.logger.logerr(f"<update_path_points.3> Error occurs: {e}")

    def navigation_loop(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
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
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.1> Error occurs: {e}")

            self.navigation_running.set()

            try:
                self.update_path_points()
                self.logger.loginfo(f"<navigation_loop.2> Updated path_points: #={len(self.path_points) if self.path_points is not None else 'None'}")
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.2> Error occurs: {e}")

            try:
                # self._get_node_active_signal()
                # if self.node_active_signal:
                if self.node_active_signal:
                    self.navigate()
                self.logger.loginfo(f"<navigation_loop.3> Navigate")
            except Exception as e:
                self.logger.logerr(f"<navigation_loop.3> Error occurs: {e}")
            finally:
                self.navigation_running.clear()
                self.logger.loginfo(f"<navigation_loop.3> Clear navigation_running")
            rate.sleep()

    def navigate(self):
        try:
            with self.navigation_lock:
                path_points_all = self.path_points

            if (path_points_all is None) or (len(path_points_all) == 0):
                self.logger.loginfo(f"<navigate.1> path_points_all is None. Skip this turn.")
                return
            self.logger.loginfo(f"<navigate.1> self.path_points: #GIDs={len(path_points_all)}")
        except Exception as e:
            self.logger.logerr(f"<navigate.1> Error occurs: {e}")

        try:
            if not self.is_path_points_updated:
                if self.previous_path_points:
                    self.path_points_vis_pub.publish(self.previous_path_points)
                self.logger.loginfo(f"<navigate.2> Publish previous path_points for visualization")
                return
            self.logger.loginfo(f"<navigate.2> ...")
        except Exception as e:
            self.logger.logerr(f"<navigate.2> Error occurs: {e}")

        try:
            colors = _color_palette(len(path_points_all), alpha=0.5)
            current_gid = self.current_gid
            self.logger.logrich(f"<navigate.3> Current group ID: {current_gid}", name="gid")
            if current_gid is None:
                self.logger.loginfo(f"<navigate.3> Current GID is None.")
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

                current_path_points = path_points_all.get(current_gid, [])
                if len(current_path_points) == 0:
                    now = rospy.Time.now()
                    t0 = self._empty_path_since.get(current_gid)

                    if t0 is None:
                        # 처음 빈 상태 감지 → 타이머 시작
                        self._empty_path_since[current_gid] = now
                        self.logger.loginfo(f"<navigate.bump> Start empty-path timer for GID({current_gid})")
                    else:
                        candidate_eids = self.sg.get_candidate_entities('object').ids  # TODO: 'object' -> 'all'
                        pending_eids = sorted(
                            [
                                eid for eid in candidate_eids
                                if self.agg_results.results_by_entity.num_queries.get(eid, 0) < self.agg_results.min_query
                            ],
                            key=lambda eid: self.agg_results.results_by_entity.num_queries.get(eid, 0)
                        )
                        pending_candidate_eids = list(set(pending_eids) & set(candidate_eids))

                        dur = now - t0
                        if dur > self._empty_path_cooldown and len(pending_candidate_eids) == 0:
                            before = self.agg_results.min_query
                            self.agg_results.min_query = min(
                                self.agg_results.min_query + 1,
                                getattr(self.agg_results, "max_query", self.agg_results.min_query + 1)
                            )
                            after = self.agg_results.min_query
                            self.logger.loginfo(
                                f"<navigate.bump> Increase min_query: {before} -> {after} "
                                f"(empty-path {dur.to_sec():.3f} sec)"
                            )

                            # 2) 이 GID에 대해 즉시 배치 구성 후 enqueue
                            try:
                                num_queries_required = self.agg_results.min_query
                                if len(pending_candidate_eids) == 0:
                                    data, pids = [], []
                                else:
                                    data, pids = self.build_batch_for_gid(
                                        eids=pending_candidate_eids, num_queries_required=num_queries_required
                                    )
                                self.logger.loginfo(f"<navigate.bump> Selected KFs: {list(set(pids))}")
                                if data:
                                    # 큐에 투입
                                    try:
                                        self.inference_queue.put_nowait(data)
                                    except queue.Full:  # 꽉 차면 다음 턴에 시도 (예약도 건너뜀)
                                        self.logger.logwarn("<navigate.bump> inference_queue full; will retry later.")
                                    else:
                                        # 예약 증가
                                        self.agg_results.schedule(current_gid, n=len(data), data=data)
                                        self.logger.loginfo(
                                            "<navigate.bump> Put data to inference_queue:\n" +
                                            "\n".join([
                                                f"  > pids: {', '.join(map(str, d['keyframes'].ids))}, etype: {d['etype']}, atype: {d['atype']}, eids: {d['eids']}"
                                                for d in data
                                            ])
                                        )
                                else:
                                    self.logger.loginfo(f"<navigate.bump> No keyframes for GID({current_gid})")
                            except Exception as e:
                                self.logger.logerr(f"<navigate.bump> Error while enqueue: {e}")

                            # 3) 타이머 리셋(지속적으로 쏟아내지 않도록)
                            self._empty_path_since[current_gid] = now  # 또는 None으로 초기화도 가능
                    # ------------------ ✅ 끝 ------------------

                exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                return
            current_path_points = path_points_all.get(current_gid)
            if current_path_points is None:
                self.logger.loginfo(f"<navigate.3> Current path_points for GID({current_gid}) is None.")
                exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                return
            exp_strategy = "geometric_frontier"
            self.exploration_strategy_pub.publish(String(exp_strategy))
            self.logger.loginfo(f"<navigate.3> current_path_points: {current_path_points.shape}")
        except Exception as e:
            self.logger.logerr(f"<navigate.3> Error occurs: {e}")

        try:
            if len(current_path_points) == 0:
                current_eids = set(self.hull_grouper.dsu.idx.keys()) & set(self.sg.get_candidate_entities('all').ids)
                if self.action == 'find':
                    unprocessed_eids = [eid for eid in current_eids
                                        if self.agg_results.results_by_entity.num_queries.get(eid, 0) <= 0]
                elif self.action == 'count':
                    unprocessed_eids = self.hull_grouper.get_low_count_eids(max_count=0)
                else:
                    raise NotImplementedError(f"self.action must be in ['find', 'count'], but {self.action} was given.")

                if len(unprocessed_eids) > 0:
                    self.inference_signal_queue.put((current_gid, unprocessed_eids))
                    self.logger.loginfo(
                        f"<navigate.4.1> Need inference of EIDs: {unprocessed_eids}. Put group({current_gid}) to inference_signal_queue.")
                else:
                    self.logger.loginfo(
                        f"<navigate.4.1> No valide EIDs. Entities in group({current_gid}) was already processed.")

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
                else:
                    self.logger.logwarn(f"<navigate.4.2> self.agent_pose is not available.")
        except Exception as e:
            self.logger.logerr(f"<navigate.4> Error occurs: {e}")

        try:
            path_points_marker = make_marker_array_from_points(
                current_path_points, ns=f"path_points_{current_gid}", color=colors[current_gid], frame_id=self.frame_id)
            self.path_points_pub.publish(path_points_marker)
            self.logger.loginfo(f"<navigate.4.3> Publish /active_waypoints (path_points)")
        except Exception as e:
            self.logger.logerr(f"<navigate.4.3> Error occurs: {e}")

        # Visualize
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
        except Exception as e:
            self.logger.logerr(f"<navigate.4.4> Error occurs: {e}")

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
            else:
                if self.active_clients.is_running:
                    if len(marker_array_all) == 0:
                        if not self.active_clients.is_paused:
                            success = self.active_clients.pause()
                        self.path_points = None  # reset path_points
                        exp_strategy = "geometric_frontier"
                        self.exploration_strategy_pub.publish(String(exp_strategy))
                        self.logger.logrich(f"<navigate.4.5> Exp Strategy: Ended ({exp_strategy})", name='navigation')
                        return
                exp_strategy = "geometric_frontier"
                self.exploration_strategy_pub.publish(String(exp_strategy))
                self.logger.logrich(f"<navigate.4.5> Is Not Active ({exp_strategy})", name='navigation')
        except Exception as e:
            self.logger.logerr(f"<navigate.4.5> Error occurs: {e}")

    def start_coverage_planning(self):
        """
        Trigger coverage path planning exploration strategy.
        This method can be called by derived classes or external logic to initiate coverage planning.
        """
        self.logger.loginfo("Starting coverage path planning strategy")
        self.exploration_strategy_pub.publish(String('coverage_planning'))


if __name__ == "__main__":
    logger = Logger()

    SCENE = "arabic_room"
    if SCENE == "arabic_room":
        instruction = "How many sofas are below a window?"
        action = 'count'
        target_name = "sofas below a window"
        candidate_names, reference_names = ['sofa'], ['window']
    else:
        raise TypeError(f"SCENE must be in ['office_1', 'hotel_room_1', 'chinese_room'], but {SCENE} was given.")

    DATA_DIR = f"/ws/external/test_data/{SCENE}"
    MAP_DIR = os.path.join(DATA_DIR, "offline_map")
    KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")

    tester = BaseVisualGrounder(
        logger=logger, action=action, target_name=target_name,
        candidate_names=candidate_names, reference_names=reference_names,
    )
    tester._init_all(logger=tester.logger, use_ros=False)
    tester._set_task(
        action, target_name=target_name,
        candidate_names=candidate_names, reference_names=reference_names
    )

    map_dirs = [os.path.join(MAP_DIR, d) for d in os.listdir(MAP_DIR)
            if os.path.isdir(os.path.join(MAP_DIR, d))]
    map_dir_sorted = sorted(map_dirs, key=os.path.getmtime)

    for dir in map_dir_sorted:
        tester.timer_callback(None, dir=dir)
        # time.sleep(0.1)

    print("Done!")

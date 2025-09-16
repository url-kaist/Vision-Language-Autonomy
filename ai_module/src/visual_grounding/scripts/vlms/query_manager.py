#!/usr/bin/env python3

import threading
import queue
import time
import uuid
import json
import os
import shutil
from typing import Dict, Any, Optional, Callable, List, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import logging
from abc import ABC, abstractmethod
import rospy
from ai_module.src.utils.logger import Logger


class QueryStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class QueryRequest:
    """쿼리 요청을 나타내는 데이터 클래스"""
    query_id: str
    query_type: str
    data: Dict[str, Any]
    priority: int = 0
    timeout: float = 30.0
    created_at: float = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None  # 추가 메타데이터
    
    def __lt__(self, other):
        """우선순위 큐를 위한 비교 연산자 (낮은 우선순위가 먼저 나오도록)"""
        # None 값 처리: None은 항상 마지막에 정렬되도록 함
        if other is None:
            return True
        if self.priority != other.priority:
            return self.priority < other.priority
        # 우선순위가 같으면 먼저 생성된 것이 먼저 나오도록 (FIFO)
        return self.created_at < other.created_at
    
    def __gt__(self, other):
        """우선순위 큐를 위한 비교 연산자 (낮은 우선순위가 먼저 나오도록)"""
        # None 값 처리: None은 항상 마지막에 정렬되도록 함
        if other is None:
            return False
        if self.priority != other.priority:
            return self.priority > other.priority
        # 우선순위가 같으면 먼저 생성된 것이 먼저 나오도록 (FIFO)
        return self.created_at > other.created_at
    
    def __eq__(self, other):
        """동등성 비교"""
        return self.query_id == other.query_id
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResponse:
    """쿼리 응답을 나타내는 데이터 클래스"""
    query_id: str
    status: QueryStatus
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    total_time: float = 0.0  # 전체 쿼리 처리 시간 (대기 시간 포함)
    completed_at: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = time.time()
        if self.metadata is None:
            self.metadata = {}


class QueryWorker(ABC):
    """쿼리 워커 추상 클래스"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Any:
        """쿼리 처리 메서드 (구현 필요)"""
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """입력 데이터 검증 (선택적 오버라이드)"""
        return True
    
    def get_required_fields(self) -> List[str]:
        """필수 필드 목록 반환 (선택적 오버라이드)"""
        return []
    
    def get_optional_fields(self) -> List[str]:
        """선택적 필드 목록 반환 (선택적 오버라이드)"""
        return []


class QueryWorkerRegistry:
    """쿼리 워커 등록 및 관리"""
    
    def __init__(self):
        self.workers: Dict[str, QueryWorker] = {}
        self.worker_configs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def register_worker(self, query_type: str, worker: QueryWorker, 
                       config: Dict[str, Any] = None):
        """워커 등록"""
        with self.lock:
            self.workers[query_type] = worker
            self.worker_configs[query_type] = config or {}
    
    def register_function(self, query_type: str, func: Callable, 
                         config: Dict[str, Any] = None):
        """함수를 워커로 등록"""
        class FunctionWorker(QueryWorker):
            def __init__(self, func):
                super().__init__(query_type, f"Function worker for {query_type}")
                self.func = func
            
            def process(self, data):
                return self.func(data)
        
        worker = FunctionWorker(func)
        self.register_worker(query_type, worker, config)
    
    def get_worker(self, query_type: str) -> Optional[QueryWorker]:
        """워커 조회"""
        with self.lock:
            return self.workers.get(query_type)
    
    def get_worker_config(self, query_type: str) -> Dict[str, Any]:
        """워커 설정 조회"""
        with self.lock:
            return self.worker_configs.get(query_type, {})
    
    def list_workers(self) -> List[str]:
        """등록된 워커 목록 반환"""
        with self.lock:
            return list(self.workers.keys())
    
    def unregister_worker(self, query_type: str) -> bool:
        """워커 등록 해제"""
        with self.lock:
            if query_type in self.workers:
                del self.workers[query_type]
                del self.worker_configs[query_type]
                return True
            return False


class QueryConfig:
    """쿼리 설정 관리"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # 기본 설정
        return {
            "default_timeout": 30.0,
            "default_priority": 0,
            "max_queue_size": 50,
            "max_workers": 3,
            "query_types": {
                "inference_query": {
                    "timeout": 30.0,
                    "priority": 0,
                    "description": "Visual inference query"
                },
                "validation_query": {
                    "timeout": 30.0,
                    "priority": 1,
                    "description": "Validation query"
                },
                "path_generation_query": {
                    "timeout": 30.0,
                    "priority": 1,
                    "description": "Path generation query"
                },
                "path_evaluation_query": {
                    "timeout": 15.0,
                    "priority": 1,
                    "description": "Path evaluation query"
                }
            }
        }
    
    def get_query_config(self, query_type: str) -> Dict[str, Any]:
        """특정 쿼리 타입 설정 조회"""
        if self.config is None:
            return {}
        return self.config.get("query_types", {}).get(query_type, {})
    
    def get_default_timeout(self) -> float:
        """기본 타임아웃 조회"""
        if self.config is None:
            return 30.0
        return self.config.get("default_timeout", 30.0)
    
    def get_default_priority(self) -> int:
        """기본 우선순위 조회"""
        if self.config is None:
            return 0
        return self.config.get("default_priority", 0)
    
    def add_query_type(self, query_type: str, config: Dict[str, Any]):
        """새로운 쿼리 타입 추가"""
        if "query_types" not in self.config:
            self.config["query_types"] = {}
        self.config["query_types"][query_type] = config
        self._save_config()
    
    def _save_config(self):
        """설정 파일 저장"""
        if self.config_file:
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save config file {self.config_file}: {e}")


class QueryManager:
    def __init__(self, max_workers: int = 3, max_queue_size: int = 50, 
                 config_file: str = None, debug: bool = False):
        try:
            quiet = rospy.get_param('~quiet', False)
        except:
            quiet = False
        self.logger = Logger(
            quiet=quiet, prefix='QueryManager', log_path="/ws/external/log/query_manager.log")
        
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # 디버깅 설정
        self.debug = False
        self.debug_dir = '/ws/external/log/query'
        
        # 설정 관리
        self.config = QueryConfig(config_file)
        
        # 워커 등록소
        self.worker_registry = QueryWorkerRegistry()
        
        # 쿼리 큐
        self.query_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # 응답 저장소
        self.responses: Dict[str, QueryResponse] = {}
        
        # 활성 쿼리 추적
        self.active_queries: Dict[str, QueryRequest] = {}
        self.query_status: Dict[str, QueryStatus] = {}
        
        # 스레드 관리
        self.executor = None
        self.worker_threads = []
        
        # 동기화
        self.lock = threading.RLock()
        self.response_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # 클라이언트 카운터 관리
        self.client_counter = 0
        self.client_counter_lock = threading.RLock()
        
        # 통계
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_processing_time': 0.0,
            'active_workers': 0,
            'query_type_stats': {}
        }
        
        # 디버그 디렉토리 초기화
        if self.debug:
            self._init_debug_directory()
    
    def _init_debug_directory(self):
        """디버그 디렉토리 초기화 (기존 파일이 있으면 재생성)"""
        try:
            # 기존 디렉토리가 있으면 삭제 후 재생성
            if os.path.exists(self.debug_dir):
                shutil.rmtree(self.debug_dir)
                self.logger.loginfo(f"Removed existing debug directory: {self.debug_dir}")
            
            # 새 디렉토리 생성
            os.makedirs(self.debug_dir, exist_ok=True)
            self.logger.loginfo(f"Debug directory initialized: {self.debug_dir}")
        except Exception as e:
            self.logger.logerr(f"Failed to initialize debug directory: {e}")
    
    def _save_debug_info(self, request: QueryRequest, response: QueryResponse):
        """디버그 정보 저장"""
        if not self.debug:
            return
            
        try:
            # 쿼리별 디렉토리 생성
            query_debug_dir = os.path.join(self.debug_dir, request.query_id)
            os.makedirs(query_debug_dir, exist_ok=True)
            
            # Query 정보 추출 (plugin에서 반환된 query_info)
            query_info = self._extract_query_info(response.result)
            
            # 실제 사용된 이미지 경로 추출
            actual_image_paths = self._extract_actual_image_paths_from_query_info(query_info)
            
            # query_info에 직접 이미지가 있는 경우 저장
            if query_info:
                # 'images' 필드 처리 (복수)
                if 'images' in query_info:
                    images = query_info['images']
                    if isinstance(images, list) and images:
                        # 이미지 리스트가 있는 경우
                        for i, image in enumerate(images):
                            if image is not None:
                                try:
                                    # 이미지 파일명 생성
                                    image_filename = f"query_images_{i}.jpg"
                                    image_path = os.path.join(query_debug_dir, image_filename)
                                    
                                    # 이미지 저장 (PIL Image 객체)
                                    if hasattr(image, 'save') and hasattr(image, 'format'):
                                        image.save(image_path)
                                        self.logger.loginfo(f"Saved PIL image {i} to {image_path}")
                                    elif isinstance(image, str) and os.path.exists(image):
                                        # 이미지 경로인 경우 복사
                                        import shutil
                                        shutil.copy2(image, image_path)
                                        self.logger.loginfo(f"Copied query image {i} from {image} to {image_path}")
                                    else:
                                        self.logger.logwarn(f"Unknown image type for image {i}: {type(image)}")
                                except Exception as e:
                                    self.logger.logerr(f"Failed to save query image {i}: {e}")
                    elif images is not None:
                        # 단일 이미지인 경우
                        try:
                            image_filename = "query_images.jpg"
                            image_path = os.path.join(query_debug_dir, image_filename)
                            
                            # 이미지 저장 (PIL Image 객체)
                            if hasattr(images, 'save') and hasattr(images, 'format'):
                                images.save(image_path)
                                self.logger.loginfo(f"Saved PIL image to {image_path}")
                            elif isinstance(images, str) and os.path.exists(images):
                                # 이미지 경로인 경우 복사
                                import shutil
                                shutil.copy2(images, image_path)
                                self.logger.loginfo(f"Copied query image from {images} to {image_path}")
                            else:
                                self.logger.logwarn(f"Unknown image type: {type(images)}")
                        except Exception as e:
                            self.logger.logerr(f"Failed to save query image: {e}")
                
                # 'image' 필드 처리 (단수)
                if 'image' in query_info:
                    image = query_info['image']
                    if image is not None:
                        try:
                            image_filename = "query_image.jpg"
                            image_path = os.path.join(query_debug_dir, image_filename)
                            
                            # 이미지 저장 (PIL Image 객체)
                            if hasattr(image, 'save') and hasattr(image, 'format'):
                                image.save(image_path)
                                self.logger.loginfo(f"Saved PIL image to {image_path}")
                            elif isinstance(image, str) and os.path.exists(image):
                                # 이미지 경로인 경우 복사
                                import shutil
                                shutil.copy2(image, image_path)
                                self.logger.loginfo(f"Copied query image from {image} to {image_path}")
                            else:
                                self.logger.logwarn(f"Unknown image type: {type(image)}")
                        except Exception as e:
                            self.logger.logerr(f"Failed to save query image: {e}")
            
            # 실제 사용된 이미지만 복사 (query_info에서 추출된 이미지들)
            copied_image_paths = []
            if actual_image_paths:
                self._copy_actual_images(actual_image_paths, query_debug_dir, request.data)
                # 복사된 이미지 경로 수집
                copied_image_paths = [path_info.get('copied_path') for path_info in actual_image_paths if path_info.get('copied_path')]
            
            # 디버그 정보 구성
            debug_info = {
                'query_id': request.query_id,
                'query_type': request.query_type,
                'status': response.status.value,
                'created_at': request.created_at,
                'completed_at': response.completed_at,
                'processing_time': response.processing_time,
                'total_time': response.total_time,  # 전체 쿼리 처리 시간 (대기 시간 포함)
                'priority': request.priority,
                'timeout': request.timeout,
                'client_counter': request.data.get('client_counter'),
                'data': self._serialize_data_for_debug(request.data),
                'result': self._serialize_result_for_debug(response.result),
                'query_info': self._serialize_query_info_for_debug(query_info),  # Plugin에서 반환된 query 정보
                'error': str(response.error) if response.error is not None else None,
                'metadata': request.metadata,
                'response_metadata': response.metadata,
                'actual_image_paths': actual_image_paths,  # 추론 시 실제 사용된 이미지
                'copied_image_paths': copied_image_paths   # 복사된 이미지들
            }
            
            # JSON 파일로 저장
            debug_file = os.path.join(query_debug_dir, 'query.json')
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.loginfo(f"Debug info saved: {debug_file}")
            
        except Exception as e:
            self.logger.logerr(f"Failed to save debug info for query {request.query_id}: {e}")
    
    def _copy_query_images(self, request: QueryRequest, query_debug_dir: str) -> List[str]:
        """쿼리에 사용된 이미지들을 복사하고 경로 반환"""
        image_paths = []
        
        try:
            # 이미지 디렉토리 생성
            images_dir = os.path.join(query_debug_dir, 'query_images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 데이터에서 이미지 경로 찾기
            data = request.data
            self.logger.loginfo(f"Searching for images in data keys: {list(data.keys())}")
            
            # 기본 이미지 키들
            image_keys = ['image_path', 'image', 'images', 'keyframe_path', 'keyframes']
            
            for key in image_keys:
                if key in data:
                    value = data[key]
                    self.logger.loginfo(f"Found key '{key}' with type: {type(value)}")
                    
                    # keyframes 객체 처리 (특별 처리)
                    if key == 'keyframes':
                        image_paths.extend(self._extract_images_from_keyframes(value, images_dir, key))
                    
                    # 단일 이미지 경로
                    elif isinstance(value, str) and os.path.exists(value):
                        copied_path = self._copy_single_image(value, images_dir, key)
                        if copied_path:
                            image_paths.append(copied_path)
                    
                    # 이미지 경로 리스트
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, str) and os.path.exists(item):
                                copied_path = self._copy_single_image(item, images_dir, f"{key}_{i}")
                                if copied_path:
                                    image_paths.append(copied_path)
                    
                    # 딕셔너리 형태의 이미지 정보
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and os.path.exists(sub_value):
                                copied_path = self._copy_single_image(sub_value, images_dir, f"{key}_{sub_key}")
                                if copied_path:
                                    image_paths.append(copied_path)
            
            self.logger.loginfo(f"Total images found and copied: {len(image_paths)}")
            
        except Exception as e:
            self.logger.logerr(f"Failed to copy images for query {request.query_id}: {e}")
        
        return image_paths
    
    def _extract_images_from_keyframes(self, keyframes_obj, images_dir: str, prefix: str) -> List[str]:
        """Keyframes 객체에서 이미지 경로들을 추출하고 복사"""
        image_paths = []
        
        try:
            self.logger.loginfo(f"Processing keyframes object: {type(keyframes_obj)}")
            
            # Keyframes 객체가 딕셔너리 형태인 경우
            if hasattr(keyframes_obj, 'items'):
                for keyframe_id, keyframe in keyframes_obj.items():
                    self.logger.loginfo(f"Processing keyframe {keyframe_id}: {type(keyframe)}")
                    
                    # Keyframe 객체에서 image_path 추출
                    if hasattr(keyframe, 'image_path') and keyframe.image_path:
                        if isinstance(keyframe.image_path, str) and os.path.exists(keyframe.image_path):
                            copied_path = self._copy_single_image(keyframe.image_path, images_dir, f"{prefix}_{keyframe_id}")
                            if copied_path:
                                image_paths.append(copied_path)
                                self.logger.loginfo(f"Copied keyframe image: {keyframe.image_path}")
                    
                    # Keyframe 객체가 딕셔너리인 경우
                    elif isinstance(keyframe, dict) and 'image_path' in keyframe:
                        image_path = keyframe['image_path']
                        if isinstance(image_path, str) and os.path.exists(image_path):
                            copied_path = self._copy_single_image(image_path, images_dir, f"{prefix}_{keyframe_id}")
                            if copied_path:
                                image_paths.append(copied_path)
                                self.logger.loginfo(f"Copied keyframe image from dict: {image_path}")
            
            # Keyframes 객체가 리스트인 경우
            elif isinstance(keyframes_obj, list):
                for i, keyframe in enumerate(keyframes_obj):
                    if hasattr(keyframe, 'image_path') and keyframe.image_path:
                        if isinstance(keyframe.image_path, str) and os.path.exists(keyframe.image_path):
                            copied_path = self._copy_single_image(keyframe.image_path, images_dir, f"{prefix}_{i}")
                            if copied_path:
                                image_paths.append(copied_path)
                                self.logger.loginfo(f"Copied keyframe image from list: {keyframe.image_path}")
            
            # 객체의 속성을 직접 확인
            elif hasattr(keyframes_obj, 'image_path') and keyframes_obj.image_path:
                if isinstance(keyframes_obj.image_path, str) and os.path.exists(keyframes_obj.image_path):
                    copied_path = self._copy_single_image(keyframes_obj.image_path, images_dir, prefix)
                    if copied_path:
                        image_paths.append(copied_path)
                        self.logger.loginfo(f"Copied single keyframe image: {keyframes_obj.image_path}")
            
            # 디버깅을 위해 객체의 모든 속성 출력
            if hasattr(keyframes_obj, '__dict__'):
                self.logger.loginfo(f"Keyframes object attributes: {list(keyframes_obj.__dict__.keys())}")
            
            # 재귀적으로 모든 이미지 경로 찾기
            image_paths.extend(self._find_all_image_paths(keyframes_obj, images_dir, prefix))
            
        except Exception as e:
            self.logger.logerr(f"Failed to extract images from keyframes: {e}")
        
        return image_paths
    
    def _find_all_image_paths(self, obj, images_dir: str, prefix: str, depth: int = 0) -> List[str]:
        """재귀적으로 객체에서 모든 이미지 경로를 찾아 복사"""
        image_paths = []
        
        if depth > 5:  # 무한 재귀 방지
            return image_paths
        
        try:
            # 문자열인 경우 이미지 경로인지 확인
            if isinstance(obj, str):
                if os.path.exists(obj) and any(obj.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                    copied_path = self._copy_single_image(obj, images_dir, f"{prefix}_found_{depth}")
                    if copied_path:
                        image_paths.append(copied_path)
                        self.logger.loginfo(f"Found image path recursively: {obj}")
            
            # 딕셔너리인 경우
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if 'image' in key.lower() or 'path' in key.lower():
                        image_paths.extend(self._find_all_image_paths(value, images_dir, f"{prefix}_{key}", depth + 1))
                    else:
                        image_paths.extend(self._find_all_image_paths(value, images_dir, prefix, depth + 1))
            
            # 리스트인 경우
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    image_paths.extend(self._find_all_image_paths(item, images_dir, f"{prefix}_{i}", depth + 1))
            
            # 객체인 경우
            elif hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if 'image' in attr_name.lower() or 'path' in attr_name.lower():
                        image_paths.extend(self._find_all_image_paths(attr_value, images_dir, f"{prefix}_{attr_name}", depth + 1))
            
        except Exception as e:
            self.logger.logerr(f"Error in recursive image search: {e}")
        
        return image_paths
    
    def _extract_query_info(self, result: Any) -> Dict[str, Any]:
        """Plugin에서 반환된 query_info 추출"""
        try:
            if result is None:
                return {}
            
            # 딕셔너리인 경우 query_info 키 확인
            if isinstance(result, dict):
                if 'query_info' in result:
                    return result['query_info']
            
            # 객체인 경우 attributes에서 query_info 확인
            if hasattr(result, '__dict__'):
                if hasattr(result, 'query_info'):
                    return result.query_info
                
                # attributes에서 query_info 찾기
                for attr_name, attr_value in result.__dict__.items():
                    if attr_name == 'query_info' and isinstance(attr_value, dict):
                        return attr_value
            
            return {}
            
        except Exception as e:
            self.logger.logerr(f"Failed to extract query info: {e}")
            return {}
    
    def _extract_actual_image_paths_from_query_info(self, query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """query_info에서 실제 사용된 이미지 경로들을 추출"""
        actual_paths = []
        
        try:
            if not query_info or 'image_paths' not in query_info:
                return actual_paths
            
            image_paths = query_info['image_paths']
            image_options = query_info.get('image_options', {})
            suffix = image_options.get('suffix', '')
            
            # image_paths가 리스트인 경우
            if isinstance(image_paths, list):
                for i, image_path in enumerate(image_paths):
                    if isinstance(image_path, str) and os.path.exists(image_path):
                        actual_paths.append({
                            "index": i,
                            "actual_path": image_path,
                            "suffix": suffix,
                            "exists": True,
                            "source": "query_info"
                        })
            
            # image_paths가 단일 문자열인 경우
            elif isinstance(image_paths, str) and os.path.exists(image_paths):
                actual_paths.append({
                    "index": 0,
                    "actual_path": image_paths,
                    "suffix": suffix,
                    "exists": True,
                    "source": "query_info"
                })
            
            self.logger.loginfo(f"Extracted {len(actual_paths)} actual image paths from query_info")
            
        except Exception as e:
            self.logger.logerr(f"Failed to extract actual image paths from query_info: {e}")
        
        return actual_paths
    
    def _copy_actual_images(self, actual_image_paths: List[Dict[str, Any]], query_debug_dir: str, request_data: Dict[str, Any] = None):
        """실제 사용된 이미지들을 복사 (이름 그대로 유지)"""
        try:
            images_dir = os.path.join(query_debug_dir, 'query_images')
            os.makedirs(images_dir, exist_ok=True)
            
            for i, path_info in enumerate(actual_image_paths):
                actual_path = path_info.get('actual_path')
                if actual_path and isinstance(actual_path, str) and os.path.exists(actual_path):
                    # 실제 사용된 이미지 복사 (원본 파일명 유지)
                    copied_path = self._copy_image_with_original_name(actual_path, images_dir)
                    if copied_path:
                        path_info['copied_path'] = copied_path
                        self.logger.loginfo(f"Copied actual image: {actual_path} -> {copied_path}")
            
        except Exception as e:
            self.logger.logerr(f"Failed to copy actual images: {e}")
    
    def _copy_image_with_original_name(self, source_path: str, dest_dir: str) -> Optional[str]:
        """이미지를 원본 파일명으로 복사"""
        try:
            if not os.path.exists(source_path):
                return None
            
            # 원본 파일명 추출
            filename = os.path.basename(source_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # 파일이 이미 존재하는 경우 번호 추가
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                new_filename = f"{base_name}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, new_filename)
                counter += 1
            
            # 파일 복사
            shutil.copy2(source_path, dest_path)
            
            # 상대 경로로 변환 (디버그 디렉토리 기준)
            rel_path = os.path.relpath(dest_path, self.debug_dir)
            
            self.logger.loginfo(f"Image copied with original name: {source_path} -> {dest_path}")
            return rel_path
            
        except Exception as e:
            self.logger.logerr(f"Failed to copy image with original name {source_path}: {e}")
            return None
    
    def _serialize_query_info_for_debug(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """query_info를 디버그용으로 직렬화"""
        try:
            if not query_info:
                return {}
            
            serialized_query_info = {}
            
            # 기본 정보들
            basic_fields = ['prompt', 'system_instruction', 'prompt_type', 'query_time', 
                          'response_text', 'target_name', 'candidate_object_ids']
            for field in basic_fields:
                if field in query_info:
                    serialized_query_info[field] = query_info[field]
            
            # 이미지 관련 정보
            if 'image_paths' in query_info:
                image_paths = query_info['image_paths']
                if isinstance(image_paths, list):
                    serialized_query_info['image_paths'] = [
                        {
                            'path': path,
                            'exists': os.path.exists(path) if isinstance(path, str) else False
                        } for path in image_paths
                    ]
                elif isinstance(image_paths, str):
                    serialized_query_info['image_paths'] = [{
                        'path': image_paths,
                        'exists': os.path.exists(image_paths)
                    }]
            
            # 이미지 옵션
            if 'image_options' in query_info:
                serialized_query_info['image_options'] = query_info['image_options']
            
            # 파싱된 응답
            if 'parsed_response' in query_info:
                serialized_query_info['parsed_response'] = self._serialize_result_value(query_info['parsed_response'])
            
            return serialized_query_info
            
        except Exception as e:
            self.logger.logerr(f"Failed to serialize query_info for debug: {e}")
            return {"error": f"Query info serialization failed: {str(e)}"}
    
    def _serialize_result_for_debug(self, result: Any) -> Any:
        """worker.process 결과를 디버그용으로 직렬화"""
        try:
            if result is None:
                return None
            
            # 기본 타입들은 그대로 반환
            if isinstance(result, (str, int, float, bool)):
                return result
            
            # 딕셔너리인 경우
            if isinstance(result, dict):
                serialized_result = {}
                for key, value in result.items():
                    serialized_result[key] = self._serialize_result_value(value)
                return serialized_result
            
            # 리스트인 경우
            if isinstance(result, list):
                return [self._serialize_result_value(item) for item in result]
            
            # 객체인 경우 (InferenceResult2 등)
            if hasattr(result, '__dict__'):
                serialized_result = {
                    "type": str(type(result)),
                    "attributes": {}
                }
                
                # 객체의 모든 속성을 직렬화
                for attr_name, attr_value in result.__dict__.items():
                    serialized_result["attributes"][attr_name] = self._serialize_result_value(attr_value)
                
                return serialized_result
            
            # 기타 객체는 문자열로 변환
            return str(result)
            
        except Exception as e:
            self.logger.logerr(f"Failed to serialize result for debug: {e}")
            return {"error": f"Result serialization failed: {str(e)}"}
    
    def _serialize_result_value(self, value: Any) -> Any:
        """결과 값의 개별 항목을 직렬화"""
        try:
            if value is None:
                return None
            
            # 기본 타입들
            if isinstance(value, (str, int, float, bool)):
                return value
            
            # 딕셔너리
            if isinstance(value, dict):
                return {k: self._serialize_result_value(v) for k, v in value.items()}
            
            # 리스트
            if isinstance(value, list):
                return [self._serialize_result_value(item) for item in value]
            
            # 객체 (Entities, Keyframes 등)
            if hasattr(value, '__dict__'):
                return {
                    "type": str(type(value)),
                    "repr": str(value)[:200] if hasattr(value, '__repr__') else "No repr available",
                    "attributes": list(value.__dict__.keys()) if hasattr(value, '__dict__') else []
                }
            
            # 기타는 문자열로 변환
            return str(value)
            
        except Exception as e:
            return {"error": f"Value serialization failed: {str(e)}"}
    
    def _serialize_data_for_debug(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """디버그를 위해 데이터를 직렬화 가능한 형태로 변환"""
        try:
            serialized_data = {}
            
            for key, value in data.items():
                if key == 'keyframes':
                    # keyframes 객체를 상세히 분석
                    serialized_data[key] = self._serialize_keyframes_for_debug(value)
                else:
                    serialized_data[key] = self._serialize_value_for_debug(value)
            
            return serialized_data
            
        except Exception as e:
            self.logger.logerr(f"Failed to serialize data for debug: {e}")
            return {"error": f"Serialization failed: {str(e)}"}
    
    def _serialize_keyframes_for_debug(self, keyframes_obj) -> Dict[str, Any]:
        """keyframes 객체를 디버그용으로 직렬화"""
        try:
            debug_info = {
                "type": str(type(keyframes_obj)),
                "image_paths": [],
            }
            
            for keyframe_id, keyframe in keyframes_obj.items():
                if hasattr(keyframe, 'image_path') and keyframe.image_path:
                    image_info = {
                        "keyframe_id": keyframe_id,
                        "image_path": keyframe.image_path,
                        "exists": os.path.exists(keyframe.image_path) if isinstance(keyframe.image_path, str) else False
                    } 
                    debug_info["image_paths"].append(image_info)
            return debug_info
            
        except Exception as e:
            return {"error": f"Keyframes serialization failed: {str(e)}"}
    
    def _serialize_value_for_debug(self, value) -> Any:
        """값을 디버그용으로 직렬화 (numpy 타입 포함)"""
        try:
            # numpy 타입들 처리
            if hasattr(value, 'dtype'):  # numpy array or scalar
                if hasattr(value, 'tolist'):  # numpy array
                    return value.tolist()
                else:  # numpy scalar
                    return value.item()
            
            # 기본 타입들
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [self._serialize_value_for_debug(item) for item in value]
            elif isinstance(value, dict):
                return {k: self._serialize_value_for_debug(v) for k, v in value.items()}
            else:
                return {
                    "type": str(type(value)),
                    "repr": str(value)[:200] if hasattr(value, '__repr__') else "No repr available",
                    "attributes": list(value.__dict__.keys()) if hasattr(value, '__dict__') else []
                }
        except Exception as e:
            return {"error": f"Value serialization failed: {str(e)}"}
    
    def _copy_single_image(self, source_path: str, dest_dir: str, prefix: str) -> Optional[str]:
        """단일 이미지 파일 복사"""
        try:
            if not os.path.exists(source_path):
                return None
            
            # 파일 확장자 추출
            _, ext = os.path.splitext(source_path)
            if not ext:
                ext = '.jpg'  # 기본 확장자
            
            # 대상 파일명 생성
            filename = f"{prefix}{ext}"
            dest_path = os.path.join(dest_dir, filename)
            
            # 파일 복사
            shutil.copy2(source_path, dest_path)
            
            # 상대 경로로 변환 (디버그 디렉토리 기준)
            rel_path = os.path.relpath(dest_path, self.debug_dir)
            
            self.logger.loginfo(f"Image copied: {source_path} -> {dest_path}")
            return rel_path
            
        except Exception as e:
            self.logger.logerr(f"Failed to copy image {source_path}: {e}")
            return None
    
    def register_worker(self, query_type: str, worker: QueryWorker, 
                       config: Dict[str, Any] = None):
        """워커 등록"""
        self.worker_registry.register_worker(query_type, worker, config)
        
        # 통계 초기화
        if query_type not in self.stats['query_type_stats']:
            self.stats['query_type_stats'][query_type] = {
                'submitted': 0,
                'completed': 0,
                'failed': 0,
                'avg_time': 0.0
            }
    
    def register_function(self, query_type: str, func: Callable, 
                         config: Dict[str, Any] = None):
        """함수를 워커로 등록"""
        self.worker_registry.register_function(query_type, func, config)
        
        # 통계 초기화
        if query_type not in self.stats['query_type_stats']:
            self.stats['query_type_stats'][query_type] = {
                'submitted': 0,
                'completed': 0,
                'failed': 0,
                'avg_time': 0.0
            }
    
    def start(self):
        """쿼리 매니저 시작"""
        if self.executor is not None:
            self.logger.logwarn("Query manager is already running")
            return
            
        self.logger.loginfo(f"Starting QueryManager with {self.max_workers} workers")
        
        # ThreadPoolExecutor 생성
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="QueryWorker"
        )
        
        # 워커 스레드들 시작
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"QueryWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        self.logger.loginfo("QueryManager started successfully")
    
    def stop(self):
        """쿼리 매니저 중지"""
        if self.executor is None:
            return
            
        self.logger.loginfo("Stopping QueryManager...")
        
        # 셧다운 이벤트 설정
        self.shutdown_event.set()
        
        # 새로운 쿼리 제출 중지
        for _ in range(self.max_workers):
            self.query_queue.put(None)  # 종료 신호
        
        # 워커 스레드들이 종료될 때까지 대기
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        # 실행자 종료
        self.executor.shutdown(wait=True)
        
        # 상태 초기화
        self.executor = None
        self.worker_threads = []
        self.shutdown_event.clear()
        
        # 큐와 응답 정리
        while not self.query_queue.empty():
            try:
                self.query_queue.get_nowait()
            except queue.Empty:
                break
        
        with self.response_lock:
            self.responses.clear()
        
        with self.lock:
            self.active_queries.clear()
            self.query_status.clear()
        
        self.logger.loginfo("QueryManager stopped")
    
    def submit_query(self, query_type: str, data: Dict[str, Any], 
                    priority: int = None, timeout: float = None,
                    callback: Optional[Callable] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """
        쿼리 제출
        
        Args:
            query_type: 쿼리 타입 (동적으로 등록된 타입)
            data: 쿼리 데이터
            priority: 우선순위 (None이면 설정에서 가져옴)
            timeout: 타임아웃 (None이면 설정에서 가져옴)
            callback: 완료 시 호출할 콜백 함수
            metadata: 추가 메타데이터
            
        Returns:
            query_id: 제출된 쿼리의 고유 ID
        """
        self.logger.loginfo(f"Submit query: {query_type}")
        # 설정에서 기본값 가져오기
        if self.config is None:
            self.logger.logerr("QueryManager config is None, using default values")
            query_config = {}
            priority = priority if priority is not None else 0
            timeout = timeout if timeout is not None else 30.0
        else:
            query_config = self.config.get_query_config(query_type)
            if query_config is None:
                self.logger.logerr(f"Query config is None for type: {query_type}, using default values")
                query_config = {}
            if priority is None:
                priority = query_config.get('priority', self.config.get_default_priority() if self.config else 0)
            if timeout is None:
                timeout = query_config.get('timeout', self.config.get_default_timeout() if self.config else 30.0)
            
        self.logger.loginfo(f"Priority: {priority}")
        self.logger.loginfo(f"Timeout: {timeout}")
        
        # 클라이언트 카운터 자동 할당
        with self.client_counter_lock:
            current_client_counter = self.client_counter
            self.client_counter += 1
            
        self.logger.loginfo(f"Current client counter: {current_client_counter}")
        
        # data에 client_counter 추가
        data_with_counter = data.copy()
        data_with_counter['client_counter'] = current_client_counter
        
        query_id = str(uuid.uuid4())
        
        request = QueryRequest(
            query_id=query_id,
            query_type=query_type,
            data=data_with_counter,
            priority=priority,
            timeout=timeout,
            callback=callback,
            metadata=metadata or {}
        )
        
        self.logger.loginfo(f"Request created with callback: {callback is not None}")
        if callback:
            self.logger.loginfo(f"Callback function: {callback}")
        
        self.logger.loginfo(f"Request: {request}")
        
        try:
            # 우선순위 큐에 추가
            self.query_queue.put(request, timeout=1.0)
            
            with self.lock:
                self.active_queries[query_id] = request
                self.query_status[query_id] = QueryStatus.PENDING
                self.stats['total_submitted'] += 1
                
                # 쿼리 타입별 통계 업데이트
                if query_type in self.stats['query_type_stats']:
                    self.stats['query_type_stats'][query_type]['submitted'] += 1
            
            self.logger.loginfo(f"Query submitted: {query_id} (type: {query_type}, priority: {priority})")
            return query_id
            
        except queue.Full:
            self.logger.loginfo(f"Query queue is full, cannot submit query: {query_id}")
            raise RuntimeError("Query queue is full")
    
    def get_query_status(self, query_id: str) -> Optional[QueryStatus]:
        """쿼리 상태 조회"""
        with self.lock:
            return self.query_status.get(query_id)
    
    def get_query_result(self, query_id: str, timeout: float = None) -> Optional[QueryResponse]:
        """쿼리 결과 조회 (블로킹)"""
        start_time = time.time()
        
        while True:
            with self.response_lock:
                if query_id in self.responses:
                    response = self.responses[query_id]
                    
                    # 응답이 완료되면 active_queries에서 제거
                    with self.lock:
                        self.active_queries.pop(query_id, None)
                    
                    return response
            
            # 타임아웃 체크
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)  # 짧은 대기
    
    def get_query_result_non_blocking(self, query_id: str) -> Optional[QueryResponse]:
        """쿼리 결과 조회 (논블로킹)"""
        with self.response_lock:
            return self.responses.get(query_id)
    
    def cancel_query(self, query_id: str) -> bool:
        """쿼리 취소"""
        with self.lock:
            if query_id in self.active_queries:
                self.query_status[query_id] = QueryStatus.CANCELLED
                self.active_queries.pop(query_id, None)
                self.logger.loginfo(f"Query cancelled: {query_id}")
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self.lock:
            stats = self.stats.copy()
            stats['active_queries'] = len(self.active_queries)
            stats['queue_size'] = self.query_queue.qsize()
            stats['completed_responses'] = len(self.responses)
            stats['registered_workers'] = self.worker_registry.list_workers()
            
        with self.client_counter_lock:
            stats['current_client_counter'] = self.client_counter
            
        return stats
    
    def list_query_types(self) -> List[str]:
        """등록된 쿼리 타입 목록 반환"""
        return self.worker_registry.list_workers()
    
    def get_query_type_info(self, query_type: str) -> Dict[str, Any]:
        """쿼리 타입 정보 반환"""
        worker = self.worker_registry.get_worker(query_type)
        config = self.worker_registry.get_worker_config(query_type)
        query_config = self.config.get_query_config(query_type)
        
        return {
            'worker': worker.name if worker else None,
            'description': worker.description if worker else None,
            'worker_config': config,
            'query_config': query_config,
            'stats': self.stats['query_type_stats'].get(query_type, {})
        }
    
    def clear_completed_responses(self):
        """완료된 응답들 정리"""
        with self.response_lock:
            # 오래된 응답들 정리 (1시간 이상)
            current_time = time.time()
            to_remove = []
            for query_id, response in self.responses.items():
                if current_time - response.completed_at > 3600:  # 1시간
                    to_remove.append(query_id)
            
            for query_id in to_remove:
                del self.responses[query_id]
            
            if to_remove:
                self.logger.loginfo(f"Cleared {len(to_remove)} old responses")
    
    def _worker_loop(self):
        """워커 스레드 루프"""
        thread_name = threading.current_thread().name
        self.logger.loginfo(f"Worker thread {thread_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # 큐에서 쿼리 가져오기
                item = self.query_queue.get(timeout=1.0)
                if item is None:  # 종료 신호
                    break
                
                self.logger.loginfo(f"Item: {item}")
                request = item
                self.logger.loginfo(f"Request callback: {request.callback is not None}")
                if request.callback:
                    self.logger.loginfo(f"Request callback function: {request.callback}")
                
                # 취소된 쿼리인지 확인
                with self.lock:
                    if self.query_status.get(request.query_id) == QueryStatus.CANCELLED:
                        self.query_queue.task_done()
                        continue
                    self.query_status[request.query_id] = QueryStatus.PROCESSING
                    
                self.logger.loginfo(f"Query status: {self.query_status[request.query_id]}")
                
                # 워커 실행
                start_time = time.time()
                try:
                    worker = self.worker_registry.get_worker(request.query_type)
                    if not worker:
                        raise ValueError(f"No worker registered for query type: {request.query_type}")
                    
                    # 데이터 검증
                    if not worker.validate_data(request.data):
                        raise ValueError(f"Invalid data for query type: {request.query_type}")
        
                    result = worker.process(request.data)
                    
                    processing_time = time.time() - start_time
                    total_time = time.time() - request.created_at  # 전체 시간 (대기 시간 포함)
                                        
                    # 성공 응답
                    response = QueryResponse(
                        query_id=request.query_id,
                        status=QueryStatus.COMPLETED,
                        result=result,
                        processing_time=processing_time,
                        total_time=total_time,
                        metadata={'worker': worker.name}
                    )
                    
                    self.logger.loginfo(f"Result: {result}")
                    
                    with self.lock:
                        self.query_status[request.query_id] = QueryStatus.COMPLETED
                        self.stats['total_completed'] += 1
                        self._update_avg_processing_time(processing_time)
                        
                        # 쿼리 타입별 통계 업데이트
                        if request.query_type in self.stats['query_type_stats']:
                            stats = self.stats['query_type_stats'][request.query_type]
                            stats['completed'] += 1
                            stats['avg_time'] = (
                                (stats['avg_time'] * (stats['completed'] - 1) + processing_time) / stats['completed']
                            )
                    
                    self.logger.loginfo(f"Query completed: {request.query_id} (time: {processing_time:.2f}s)")
                    
                    # Execute callback if provided (콜백을 먼저 실행)
                    if request.callback:
                        self.logger.loginfo(f"Starting callback execution for query: {request.query_id}")
                        threading.Thread(
                            target=self._execute_callback,
                            args=(request.callback, response),
                            daemon=True
                        ).start()
                        self.logger.loginfo(f"Callback thread started for query: {request.query_id}")
                    else:
                        self.logger.loginfo(f"No callback provided for query: {request.query_id}")
                    
                    # 디버그 정보 저장 (콜백 실행 후, 실패해도 콜백에 영향 없음)
                    try:
                        self._save_debug_info(request, response)
                    except Exception as debug_error:
                        self.logger.logerr(f"Debug info save failed (non-critical): {debug_error}")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    total_time = time.time() - request.created_at  # 전체 시간 (대기 시간 포함)
                                        
                    # 실패 응답
                    response = QueryResponse(
                        query_id=request.query_id,
                        status=QueryStatus.FAILED,
                        error=e,
                        processing_time=processing_time,
                        total_time=total_time
                    )
                    
                    with self.lock:
                        self.query_status[request.query_id] = QueryStatus.FAILED
                        self.stats['total_failed'] += 1
                        
                        # 쿼리 타입별 통계 업데이트
                        if request.query_type in self.stats['query_type_stats']:
                            self.stats['query_type_stats'][request.query_type]['failed'] += 1
                    
                    self.logger.logerr(f"Query failed: {request.query_id} - {e}")
                    
                    # Execute callback even for failed queries (콜백을 먼저 실행)
                    if request.callback:
                        self.logger.loginfo(f"Starting callback execution for failed query: {request.query_id}")
                        threading.Thread(
                            target=self._execute_callback,
                            args=(request.callback, response),
                            daemon=True
                        ).start()
                        self.logger.loginfo(f"Callback thread started for failed query: {request.query_id}")
                    
                    # 디버그 정보 저장 (실패한 쿼리도 포함, 실패해도 콜백에 영향 없음)
                    try:
                        self._save_debug_info(request, response)
                    except Exception as debug_error:
                        self.logger.logerr(f"Debug info save failed for failed query (non-critical): {debug_error}")
                
                # 응답 저장
                with self.response_lock:
                    self.responses[request.query_id] = response
                
                # 타임아웃 체크
                if time.time() - request.created_at > request.timeout:
                    self.logger.logwarn(f"Query timeout: {request.query_id}")
                
                # 정상적으로 처리된 경우에만 task_done() 호출
                self.query_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.logerr(f"Error in worker loop: {e}")
    
    def _execute_callback(self, callback: Callable, response: QueryResponse):
        """콜백 함수 실행"""
        try:
            self.logger.loginfo(f"Executing callback for query: {response.query_id}")
            callback(response)
            self.logger.loginfo(f"Callback completed successfully for query: {response.query_id}")
        except Exception as e:
            self.logger.logerr(f"Error executing callback for query {response.query_id}: {e}")
            import traceback
            self.logger.logerr(f"Callback traceback: {traceback.format_exc()}")
    
    def _update_avg_processing_time(self, processing_time: float):
        """평균 처리 시간 업데이트"""
        total_completed = self.stats['total_completed']
        if total_completed > 0:
            current_avg = self.stats['avg_processing_time']
            self.stats['avg_processing_time'] = (
                (current_avg * (total_completed - 1) + processing_time) / total_completed
            )
import os
import importlib
import inspect
from typing import Dict, Any, List
import time

from ai_module.src.visual_grounding.scripts.vlms.query_manager import QueryWorker
from ai_module.src.visual_grounding.scripts.vlms.prompt import PROMPT
from ai_module.src.visual_grounding.scripts.vlms.system_instruction import SYSTEM_INSTRUCTION
from ai_module.src.visual_grounding.scripts.vlms.utils.helpers import parse_json
from ai_module.src.visual_grounding.scripts.structures.inference_result import InferenceResult2
from ai_module.src.utils.logger import Logger

try:
    import rospy
except:
    pass
               
class VisualFollowerPlugin(QueryWorker):
    """Visual Follower용 플러그인 베이스 클래스"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.visual_follower = None  # Visual Follower 인스턴스 참조

    def set_visual_follower(self, vf_instance):
        """Visual Follower 인스턴스 설정"""
        self.visual_follower = vf_instance
        # VisualFollower의 logger 사용
        self.logger = vf_instance.logger
    
    def get_required_fields(self) -> List[str]:
        """기본 필수 필드"""
        return ["keyframes"]
    
    def get_optional_fields(self) -> List[str]:
        """기본 선택적 필드"""
        return ["options", "metadata"]       

class InferencePlugin(VisualFollowerPlugin):
    """추론 플러그인"""
    
    def __init__(self):
        super().__init__("inference", "Visual inference plugin")
    
    def process(self, data: Dict[str, Any]) -> Any:
        """추론 처리"""
        if not self.visual_follower:
            raise ValueError("Visual Follower instance not set")
        
        keyframes = data['keyframes']
        options = data['options']
        target_name = data['target_name']
        candidate_object_ids = data['candidate_object_ids']
        client_counter = data.get('client_counter', 0)
        
        # Query the model
        query_result = self.query_worker({
            'keyframes': keyframes,
            'options': options,
            'target_name': target_name,
            'candidate_object_ids': candidate_object_ids
        }, client_counter)
        
        # 결과에서 response와 query_info 분리
        result = query_result['response']
        query_info = query_result['query_info']

        # Verify
        prompt_type = options['prompt']['action']
        object_ids = result.get('object_ids', [])
        if prompt_type in ["inference_find"]:
            if len(object_ids) != 1:
                self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(object_ids)} != 1")
                final_result = {
                    'inference_result': None,
                    'target_name': target_name,
                    'query_info': query_info
                }
                return final_result
        elif prompt_type in ['inference_count']:
            if len(object_ids) != result.get('count'):
                self.logger.logwarn(f"Count mismatch: len(object_ids)={len(object_ids)} != count={result.get('count')}")
                final_result = {
                    'inference_result': None,
                    'target_name': target_name,
                    'query_info': query_info
                }
                return final_result
        else:
            if prompt_type in ['inference_follow_between'] and len(object_ids) not in [1, 2]:
                self.logger.logwarn(f"Object ids mismatch: len(object_ids)={len(object_ids)} not in [1, 2]")
                final_result = {
                    'inference_result': None,
                    'target_name': target_name,
                    'query_info': query_info
                }
                return final_result
            
        # Return the result
        sg = self.visual_follower.sg
        
        has_candidate = all(obj_id in sg.get_candidate_entities(self.visual_follower.etypes[0]).ids for obj_id in object_ids)
        
        inference_result = InferenceResult2(
            objects=sg.entities(self.visual_follower.etypes[0]).get(object_ids),
            keyframes=keyframes,
            reason=result.get('reason', ""),
            stamp=rospy.Time.now(),
            confidence=0.5,
            validation_count=0,
            has_candidate=has_candidate
        )
        
        final_result = {
            'inference_result': inference_result,
            'target_name': target_name,
            'query_info': query_info
        }
        
        return final_result
    
    def query_worker(self, input_data, client_counter):
        client = self.visual_follower.clients[client_counter % len(self.visual_follower.clients)]

        keyframes = input_data['keyframes']
        target_name = input_data['target_name']
        candidate_object_ids = input_data['candidate_object_ids']

        options = input_data['options']
        prompt_type = options['prompt']['action']
        previous_history = options['prompt']['previous_history'] # TODO

        # Prepare the prompt and system instruction
        prompt = PROMPT[prompt_type].format(target_name, candidate_object_ids)
        system_instruction = SYSTEM_INSTRUCTION[prompt_type]
        images, image_paths = self.visual_follower.get_images(keyframes, **options['image'])

        # Query the model
        start_time = time.time()
        message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
        end_time = time.time()

        response_text, _, _ = client.get_response(message, **options['get_response'])
        response = parse_json(response_text)
        
        object_ids = response.get('object_ids', [])
        
        # Log the response
        result_text = "\n====================================================\n"
        result_text += f"Image paths: {image_paths}\n"
        result_text += f"  > {options['image']}\n"
        result_text += f"Prompt Type: {prompt_type}\n"
        result_text += f"Prompt: {prompt}\n"
        # result_text += f"System instruction: {system_instruction}\n"
        result_text += f"Response: {response_text}\n"
        result_text += f"Response: {response}\n"
        result_text += f"===================================================="
        self.logger.loginfo(result_text)
    
        # Query 정보를 포함한 결과 반환
        query_info = {
            'prompt': prompt,
            # 'system_instruction': system_instruction,
            'image_paths': image_paths,
            'image_options': options['image'],
            'prompt_type': prompt_type,
            'query_time': end_time - start_time,
            'response_text': response_text,
            'parsed_response': response,
            'target_name': target_name,
            'candidate_object_ids': candidate_object_ids
        }
        
        return {
            'response': response,
            'query_info': query_info
        } 
    
    def get_required_fields(self) -> List[str]:
        return ["keyframes", "options", "target_name", "candidate_object_ids"]


class ValidationPlugin(VisualFollowerPlugin):
    """검증 플러그인"""
    
    def __init__(self):
        super().__init__("validation", "Validation plugin")
    
    def process(self, data: Dict[str, Any]) -> Any:
        """검증 처리"""
        try:
            if not self.visual_follower:
                raise ValueError("Visual Follower instance not set")
            
            keyframes = data['keyframes']
            options = data['options']
            target_name = data['target_name']
            candidate_object_ids = data['candidate_object_ids']
            inference_result = data['inference_result']
            client_counter = data.get('client_counter', 0)
            
            # Query the model
            query_result = self.query_worker({
                'keyframes': keyframes,
                'options': options,
                'target_name': target_name,
                'candidate_object_ids': candidate_object_ids,
            }, client_counter)
            
            # 결과에서 response와 query_info 분리
            result = query_result['response']
            query_info = query_result['query_info']
            
            # Verify
            confidence = result.get('confidence', -1)
            if confidence not in [0, 1]:
                self.logger.logwarn(f"Confidence is not 0 or 1: {confidence}")
                final_result = {
                    'confidence': None,
                    'reason': result.get('reason', ""),
                    'target_name': target_name,
                    'inference_result': inference_result,
                    'query_info': query_info
                }
                return final_result
            
            # Return the result
            validation_result = {
                'confidence': confidence,
                'reason': result.get('reason', ""),
                'target_name': target_name,
                'inference_result': inference_result,
                'query_info': query_info
            }
        except Exception as e:
            self.logger.logerr(f"Error in ValidationPlugin.process: {e}")
            import traceback
            self.logger.logerr(f"Traceback: {traceback.format_exc()}")
            return None
        
        return validation_result
    
    def query_worker(self, input_data, client_counter):
        client = self.visual_follower.clients[client_counter % len(self.visual_follower.clients)]

        keyframes = input_data['keyframes']
        target_name = input_data['target_name']
        candidate_object_ids = input_data['candidate_object_ids']

        options = input_data['options']
        prompt_type = options['prompt']['action']
        previous_history = options['prompt']['previous_history'] # TODO

        # Prepare the prompt and system instruction
        prompt = PROMPT[prompt_type].format(target_name, candidate_object_ids)
        system_instruction = SYSTEM_INSTRUCTION[prompt_type]
        images, image_paths = self.visual_follower.get_images(keyframes, **options['image'])

        # Query the model
        start_time = time.time()
        message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
        end_time = time.time()

        response_text, _, _ = client.get_response(message, **options['get_response'])
        response = parse_json(response_text)
        
        # Log the response
        result_text = "\n====================================================\n"
        result_text += f"Image paths: {image_paths}\n"
        result_text += f"  > {options['image']}\n"
        result_text += f"Prompt Type: {prompt_type}\n"
        result_text += f"Prompt: {prompt}\n"
        # result_text += f"System instruction: {system_instruction}\n"
        result_text += f"Response: {response_text}\n"
        result_text += f"Response: {response}\n"
        result_text += f"===================================================="
        self.logger.loginfo(result_text)
        
        # Query 정보를 포함한 결과 반환
        query_info = {
            'prompt': prompt,
            # 'system_instruction': system_instruction,
            'image_paths': image_paths,
            'image_options': options['image'],
            'prompt_type': prompt_type,
            'query_time': end_time - start_time,
            'response_text': response_text,
            'parsed_response': response,
            'target_name': target_name,
            'candidate_object_ids': candidate_object_ids
        }
        
        return {
            'response': response,
            'query_info': query_info
        } 
    
    
    def get_required_fields(self) -> List[str]:
        return ["keyframes", "options", "target_name", "candidate_object_ids"]


class PathGenerationPlugin(VisualFollowerPlugin):
    """경로 생성 플러그인"""
    
    def __init__(self):
        super().__init__("path_generation", "Path generation plugin")
    
    def process(self, data: Dict[str, Any]) -> Any:
        """경로 생성 처리"""
        if not self.visual_follower:
            raise ValueError("Visual Follower instance not set")
        
        keyframes = data['keyframes']
        options = data['options']
        target_name = data['target_name']
        agent_pose = data['agent_pose']
        client_counter = data.get('client_counter', 0)
        
        # Query the model
        query_result = self.query_worker({
            'keyframes': keyframes,
            'options': options,
            'target_name': target_name,
        }, client_counter)
        
        # 결과에서 response와 query_info 분리
        result = query_result['response']
        query_info = query_result['query_info']
        
        path = result.get('selected_numbers', [])
        
        # Verify
        if len(path) == 0:
            self.logger.logwarn(f"Path is empty")
            final_result = {
                'path_points': None,
                'target_name': target_name,
                'agent_pose': agent_pose,
                'query_info': query_info
            }
            return final_result
        elif len(path) > 3:
            self.logger.logwarn(f"Path is too long: {len(path)}")
            final_result = {
                'path_points': None,
                'target_name': target_name,
                'agent_pose': agent_pose,
                'query_info': query_info
            }
            return final_result
        
        # Return the result
        _, keyframe = next(iter(keyframes.items()))
        movable_points = keyframe.movable_points
        path_points = []
        for p_id in path:
            path_points.append(movable_points[p_id])
            
        final_result = {
            'path_points': path_points,
            'target_name': target_name,
            'agent_pose': agent_pose,
            'query_info': query_info
        }
        
        return final_result        
        
    def query_worker(self, input_data, client_counter):
        client = self.visual_follower.clients[client_counter % len(self.visual_follower.clients)]

        keyframes = input_data['keyframes']

        if len(keyframes) != 1:
            self.logger.logwarn(f"Keyframes mismatch: len(keyframes)={len(keyframes)} != 1")
            return None
        _, keyframe = next(iter(keyframes.items()))

        target_name = input_data['target_name']
        
        action = self.visual_follower.subtask.action
        dist_idx2movable_points_indices = keyframe.dist_idx2movable_points_indices

        options = input_data['options']
        prompt_type = options['prompt']['action']
        previous_history = options['prompt']['previous_history']  # TODO

        # Prepare the prompt and system instruction
        if action == 'stop':
            verb = action + " at"
        else:
            verb = action
        prompt = PROMPT[prompt_type].format(verb + " " + target_name,
                                            dist_idx2movable_points_indices[0],
                                            dist_idx2movable_points_indices[1],
                                            dist_idx2movable_points_indices[2])
        system_instruction = SYSTEM_INSTRUCTION[prompt_type]
        images, image_paths = self.visual_follower.get_images(keyframes, **options['image'])

        # Query the model
        start_time = time.time()
        message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
        end_time = time.time()

        response_text, _, _ = client.get_response(message, **options['get_response'])
        response = parse_json(response_text)

        # Log the response
        result_text = "\n====================================================\n"
        result_text += f"Image paths: {image_paths}\n"
        result_text += f"  > {options['image']}\n"
        result_text += f"Prompt Type: {prompt_type}\n"
        result_text += f"Prompt: {prompt}\n"
        # result_text += f"System instruction: {system_instruction}\n"
        result_text += f"Response: {response_text}\n"
        result_text += f"Response: {response}\n"
        result_text += f"===================================================="
        self.logger.loginfo(result_text)
        
        query_info = {
            'prompt': prompt,
            # 'system_instruction': system_instruction,
            'image_paths': image_paths,
            'image_options': options['image'],
            'prompt_type': prompt_type,
            'query_time': end_time - start_time,
            'response_text': response_text,
            'parsed_response': response,
            'target_name': target_name,
            # 'images': images
        }
        
        return {
            'response': response,
            'query_info': query_info
        }

    def _default_path_generation(self, keyframes):
        """기본 경로 생성 로직"""
        return [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
    
    def get_required_fields(self) -> List[str]:
        return ["keyframes", "options", "target_name"]


class PathEvaluationPlugin(VisualFollowerPlugin):
    """경로 평가 플러그인"""
    
    def __init__(self):
        super().__init__("path_evaluation", "Path evaluation plugin")
    
    def process(self, data: Dict[str, Any]) -> Any:
        """경로 평가 처리"""
        if not self.visual_follower:
            raise ValueError("Visual Follower instance not set")
        
        keyframes = data['keyframes']
        options = data['options']
        target_name = data['target_name']
        client_counter = data.get('client_counter', 0)
            
        # Query the model
        query_result = self.query_worker({
            'keyframes': keyframes,
            'options': options,
            'target_name': target_name,
        }, client_counter)
        
        # 결과에서 response와 query_info 분리
        result = query_result['response']
        query_info = query_result['query_info']
        
        mission_status = result.get('mission_status', -1)
        
        # Verify
        if mission_status not in [0, 1]:
            self.logger.logwarn(f"Mission status is not 0 or 1: {mission_status}")
            final_result = {
                'mission_status': None,
                'reason': result.get('reason', ""),
                'target_name': target_name,
                'query_info': query_info
            }
            return final_result
        
        # Return the result
        final_result = {
            'mission_status': mission_status,
            'reason': result.get('reason', ""),
            'target_name': target_name,
            'query_info': query_info
        }
        
        return final_result
        
    def query_worker(self, input_data, client_counter):
        client = self.visual_follower.clients[client_counter % len(self.visual_follower.clients)]
        
        keyframes = input_data['keyframes']
        target_name = input_data['target_name']
        
        action = self.visual_follower.subtask.action
        
        options = input_data['options']
        prompt_type = options['prompt']['action']
        previous_history = options['prompt']['previous_history']  # TODO
        
        # Prepare the prompt and system instruction
        prompt = PROMPT[prompt_type].format(action + " " + target_name)
        system_instruction = SYSTEM_INSTRUCTION[prompt_type]
        
        # current image
        options['image'] = {
            'suffix': ['_total_path_history_occupancy_grid', '_total_path_history_image'],
            'preprocess': 'original',
        }
        images, image_paths = self.visual_follower.get_images(keyframes['-1'], **options['image'])
        
        # previous images
        options['image'] = {
            'suffix': [''],
            'preprocess': 'original',
        }
        for kf_id, kf in keyframes.items():
            if kf_id == "-1":
                continue
            else:
                image, image_path = self.visual_follower.get_images(kf, **options['image'])
                images.append(image[0])
                image_paths.append(image_path[0])
                
        # images, image_paths = self.visual_follower.get_images(keyframes, **options['image'])
    
        # Query the model
        start_time = time.time()
        message = client.construct_message(prompt, images, system_instruction, **options['construct_message'])
        end_time = time.time()
        
        response_text, _, _ = client.get_response(message, **options['get_response'])
        response = parse_json(response_text)
        
        # Log the response
        result_text = "\n====================================================\n"
        result_text += f"Image paths: {image_paths}\n"
        result_text += f"  > {options['image']}\n"
        result_text += f"Prompt Type: {prompt_type}\n"
        result_text += f"Prompt: {prompt}\n"
        # result_text += f"System instruction: {system_instruction}\n"
        result_text += f"Response: {response_text}\n"
        result_text += f"Response: {response}\n"
        result_text += f"===================================================="
        self.logger.loginfo(result_text)
        
        query_info = {
            'prompt': prompt,
            # 'system_instruction': system_instruction,
            'image_paths': image_paths,
            'image_options': options['image'],
            'prompt_type': prompt_type,
            'query_time': end_time - start_time,
            'response_text': response_text,
            'parsed_response': response,
            'target_name': target_name,
            'images': images
        }
        
        return {
            'response': response,
            'query_info': query_info
        }

    
    def get_required_fields(self) -> List[str]:
        return ["keyframes", "options", "target_name"]

class PluginManager:
    """플러그인 관리자"""
    
    def __init__(self):
        self.plugins: Dict[str, VisualFollowerPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, query_type: str, plugin: VisualFollowerPlugin, 
                       config: Dict[str, Any] = None):
        """플러그인 등록"""
        self.plugins[query_type] = plugin
        self.plugin_configs[query_type] = config or {}
    
    def get_plugin(self, query_type: str) -> VisualFollowerPlugin:
        """플러그인 조회"""
        return self.plugins.get(query_type)
    
    def list_plugins(self) -> List[str]:
        """등록된 플러그인 목록"""
        return list(self.plugins.keys())
    
    def load_plugins_from_directory(self, directory: str):
        """디렉토리에서 플러그인 자동 로드"""
        if not os.path.exists(directory):
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, os.path.join(directory, module_name + '.py')
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 플러그인 클래스 찾기
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, VisualFollowerPlugin) and 
                            obj != VisualFollowerPlugin):
                            plugin_instance = obj()
                            self.register_plugin(plugin_instance.name, plugin_instance)
                            
                except Exception as e:
                    print(f"Failed to load plugin {module_name}: {e}")
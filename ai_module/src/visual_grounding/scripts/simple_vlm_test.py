#!/usr/bin/env python3
"""
Simplified VLM testing script for visual grounding.

This script:
1. Loads annotated images from keyframes/visual_grounding_0/ folder
2. Uses 3 image types: global_object, global_detection, raw image
3. Generates prompts based on task description
4. Queries VLM and returns response

Usage:
    python simple_vlm_test.py \
        --keyframes_dir /path/to/keyframes \
        --target_description "red pillow on the sofa" \
        --candidate_names "pillow,cushion" \
        --reference_names "sofa,chair" \
        --action find \
        --selected_keyframes "keyframe_00001,keyframe_00005" \
        --candidate_ids "1,2,3,4,5"
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from PIL import Image
import glob

# Add path for imports
sys.path.append('/ws/external/')
sys.path.append('/ws/external/ai_module/src')

from ai_module.src.utils.logger import Logger
from ai_module.src.visual_grounding.scripts.vlms.loaders.vision_client import VisionLlmClient
from ai_module.src.visual_grounding.scripts.vlms.prompt_renderer import PromptRenderer, SystemInstructionRenderer
from ai_module.src.visual_grounding.scripts.vlms.utils.helpers import parse_json


class SimpleVLMTester:
    def __init__(self):
        self.logger = Logger()
        
        # Load configuration
        with open("/ws/external/ai_module/src/config.json", "r") as f:
            config = json.load(f)
        
        # Initialize VLM client
        self.client = VisionLlmClient(
            model_name=config.get('MODEL_NAME', 'gpt-4o'), 
            api_key=config['OPENAI_API_KEY0']
        )
        
        self.logger.loginfo(f"Initialized VLM client with model: {config.get('MODEL_NAME', 'gemini-2.5-flash')}")
    
    def find_annotated_images(self, keyframes_dir: str, keyframe_name: str) -> Dict[str, str]:
        """
        Find images for a keyframe:
        - object_annotated: annotated with object bounding boxes
        - none: original keyframe image
        """
        vg_dir = os.path.join(keyframes_dir, "visual_grounding_0")
        
        image_types = {}
        
        # Pattern: keyframe_00001_annotated_global_object.jpg
        if os.path.exists(vg_dir):
            object_pattern = f"{keyframe_name}_annotated_global_object.*"
            object_files = glob.glob(os.path.join(vg_dir, object_pattern))
            if object_files:
                image_types['object_annotated'] = object_files[0]
        
        # Raw image (original keyframe)
        raw_pattern = f"{keyframe_name}.*"
        raw_files = glob.glob(os.path.join(keyframes_dir, raw_pattern))
        if raw_files:
            image_types['none'] = raw_files[0]
        
        self.logger.loginfo(f"Found images for {keyframe_name}: {list(image_types.keys())}")
        return image_types
    
    def load_images_for_keyframes(self, keyframes_dir: str, selected_keyframes: List[str]) -> List[Dict[str, Any]]:
        """Load all image types for selected keyframes"""
        keyframe_data = []
        
        for keyframe_name in selected_keyframes:
            image_paths = self.find_annotated_images(keyframes_dir, keyframe_name)
            print(image_paths)
            
            if not image_paths:
                self.logger.logwarn(f"No images found for keyframe: {keyframe_name}")
                continue
            
            images = {}
            for img_type, img_path in image_paths.items():
                try:
                    image = Image.open(img_path)
                    images[img_type] = image
                    self.logger.loginfo(f"Loaded {img_type}: {os.path.basename(img_path)}")
                except Exception as e:
                    self.logger.logerr(f"Failed to load {img_type} image {img_path}: {e}")
            
            if images:
                keyframe_data.append({
                    'keyframe_name': keyframe_name,
                    'images': images,
                    'image_paths': image_paths
                })
        
        return keyframe_data
    
    def test_vlm_prompt(self, 
                       keyframes_dir: str,
                       target_description: str,
                       candidate_names: List[str],
                       reference_names: List[str],
                       selected_keyframes: List[str],
                       action: str = 'find',
                       candidate_ids: List[int] = None,
                       use_image_type: str = 'global_object') -> Dict[str, Any]:
        """
        Test VLM with specified parameters
        
        Args:
            keyframes_dir: Path to keyframes directory
            target_description: Description of target to find/count
            candidate_names: List of candidate object names
            reference_names: List of reference object names  
            selected_keyframes: List of keyframe names (without extension)
            action: 'find' or 'count'
            candidate_ids: List of candidate object IDs
            use_image_type: Which image type to use ('global_object', 'global_detection', 'raw')
        """
        
        # Default candidate IDs
        if candidate_ids is None:
            candidate_ids = [1, 2, 3, 4, 5]
        
        # Load keyframe data
        keyframe_data = self.load_images_for_keyframes(keyframes_dir, selected_keyframes)
        
        if not keyframe_data:
            self.logger.logerr("No valid keyframes loaded")
            return None
        
        # Initialize prompt renderers
        prompt_renderer = PromptRenderer(description=target_description)
        system_instruction_renderer = SystemInstructionRenderer()
        
        # Configure prompt options based on action and image type
        if use_image_type == 'object_annotated':
            atype = 'object_box_id'
        else:  # 'none'
            atype = 'none'
        
        options = {
            'prompt': {
                'rtype': 'inference',
                'action': 'select_box',
                'atype': atype,
                'hint': 'reference_object' if reference_names else 'none',
                'is_plural': (action == 'count'),
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
        
        # Generate prompt and system instruction
        prompt = prompt_renderer.render(**options['prompt'], anno_ids=candidate_ids)
        system_instruction = system_instruction_renderer.render(**options['prompt'])
        
        # Collect images of specified type
        images = []
        image_paths = []
        
        for kf_data in keyframe_data:
            if use_image_type in kf_data['images']:
                images.append(kf_data['images'][use_image_type])
                image_paths.append(kf_data['image_paths'][use_image_type])
            else:
                self.logger.logwarn(f"Image type {use_image_type} not found for {kf_data['keyframe_name']}")
        
        if not images:
            self.logger.logerr(f"No {use_image_type} images found")
            return None
        
        # Print test configuration
        print("\n" + "="*80)
        print("VLM TEST CONFIGURATION")
        print("="*80)
        print(f"Target Description: {target_description}")
        print(f"Candidate Names: {candidate_names}")
        print(f"Reference Names: {reference_names}")
        print(f"Action: {action}")
        print(f"Image Type: {use_image_type}")
        print(f"Candidate IDs: {candidate_ids}")
        print(f"Selected Keyframes: {selected_keyframes}")
        print(f"Images Found: {len(images)}")
        
        # Print system instruction
        print("\n" + "="*80)
        print("SYSTEM INSTRUCTION:")
        print("="*80)
        print(system_instruction)
        
        # Print prompt
        print("\n" + "="*80)
        print("GENERATED PROMPT:")
        print("="*80)
        print(prompt)
        
        # Print image info
        print("\n" + "="*80)
        print("IMAGES SENT TO VLM:")
        print("="*80)
        for i, path in enumerate(image_paths):
            print(f"  {i+1}. {os.path.basename(path)}")
        print("="*80)
        
        # Query VLM
        try:
            print(f"\nQuerying VLM ({self.client.model_name})...")
            message = self.client.construct_message(prompt, images, system_instruction, **options['construct_message'])
            response_text, chat_completion, usage_log = self.client.get_response(message, **options['get_response'])
            
            # Parse response
            response_json = parse_json(response_text)
            
            # Print results
            print("\n" + "="*80)
            print("VLM RESPONSE:")
            print("="*80)
            print("Raw Response:")
            print(response_text)
            print("\nParsed JSON:")
            print(response_json)
            print(f"\nUsage: {usage_log}")
            print("="*80)
            
            # Analyze response
            analysis = self.analyze_response(response_json, candidate_ids, action)
            print(f"\nResponse Analysis: {analysis}")
            
            return {
                'target_description': target_description,
                'action': action,
                'image_type': use_image_type,
                'candidate_ids': candidate_ids,
                'selected_keyframes': selected_keyframes,
                'prompt': prompt,
                'system_instruction': system_instruction,
                'response_text': response_text,
                'response_json': response_json,
                'analysis': analysis,
                'usage_log': usage_log,
                'image_paths': image_paths
            }
            
        except Exception as e:
            self.logger.logerr(f"Error querying VLM: {e}")
            return {
                'error': str(e),
                'prompt': prompt,
                'system_instruction': system_instruction,
                'image_paths': image_paths
            }
    
    def analyze_response(self, response_json: Dict, candidate_ids: List[int], action: str) -> Dict[str, Any]:
        """Analyze VLM response quality"""
        analysis = {
            'valid_format': False,
            'has_target_ids': False,
            'has_reason': False,
            'target_ids_valid': False,
            'response_type': 'unknown'
        }
        
        if not response_json:
            analysis['response_type'] = 'invalid_json'
            return analysis
        
        analysis['valid_format'] = True
        
        # Check for required fields
        if 'target_ids' in response_json:
            analysis['has_target_ids'] = True
            target_ids = response_json['target_ids']
            
            # Validate target IDs
            if isinstance(target_ids, list):
                try:
                    target_ids_int = [int(tid) for tid in target_ids]
                    if all(tid in candidate_ids for tid in target_ids_int):
                        analysis['target_ids_valid'] = True
                except:
                    pass
        
        if 'reason' in response_json:
            analysis['has_reason'] = True
        
        # Determine response type
        if analysis['has_target_ids'] and analysis['target_ids_valid']:
            target_ids = response_json['target_ids']
            if len(target_ids) == 0:
                analysis['response_type'] = 'no_match'
            elif len(target_ids) == 1:
                analysis['response_type'] = 'single_match'
            else:
                analysis['response_type'] = 'multiple_matches'
        else:
            analysis['response_type'] = 'invalid_response'
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description='Simple VLM testing for visual grounding')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                       help='Path to keyframes directory')
    parser.add_argument('--target_description', type=str, required=True,
                       help='Target description (e.g., "red pillow on the sofa")')
    parser.add_argument('--candidate_names', type=str, required=True,
                       help='Candidate object names (comma-separated)')
    parser.add_argument('--reference_names', type=str, default='',
                       help='Reference object names (comma-separated)')
    parser.add_argument('--action', type=str, choices=['find', 'count'], default='find',
                       help='Action type')
    parser.add_argument('--selected_keyframes', type=str, required=True,
                       help='Selected keyframe names without extension (comma-separated, e.g., "keyframe_00001,keyframe_00005")')
    parser.add_argument('--candidate_ids', type=str, default='1,2,3,4,5',
                       help='Candidate object IDs (comma-separated)')
    parser.add_argument('--image_type', type=str, 
                       choices=['object_annotated', 'none'], 
                       default='object_annotated',
                       help='Image type to use for VLM')
    
    args = parser.parse_args()
    
    # Parse arguments
    candidate_names = [name.strip() for name in args.candidate_names.split(',')]
    reference_names = [name.strip() for name in args.reference_names.split(',') if name.strip()] if args.reference_names else []
    selected_keyframes = [name.strip() for name in args.selected_keyframes.split(',')]
    candidate_ids = [int(id.strip()) for id in args.candidate_ids.split(',')]
    
    # Initialize tester
    tester = SimpleVLMTester()
    
    # Run test
    result = tester.test_vlm_prompt(
        keyframes_dir=args.keyframes_dir,
        target_description=args.target_description,
        candidate_names=candidate_names,
        reference_names=reference_names,
        selected_keyframes=selected_keyframes,
        action=args.action,
        candidate_ids=candidate_ids,
        use_image_type=args.image_type
    )
    
    if result and 'error' not in result:
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Print final answer
        if result.get('response_json'):
            if result['action'] == 'find':
                target_ids = result['response_json'].get('target_ids', [])
                if target_ids:
                    print(f"ANSWER: Found target at ID(s): {target_ids}")
                else:
                    print("ANSWER: No target found")
            elif result['action'] == 'count':
                target_ids = result['response_json'].get('target_ids', [])
                print(f"ANSWER: Count = {len(target_ids)}")
    else:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        if result and 'error' in result:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main() 
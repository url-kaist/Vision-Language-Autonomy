#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external")
import rospy
import numpy as np
import torch
import clip
import cv2
import math
from PIL import Image
from sensor_msgs.msg import CompressedImage, Image as RosImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String, Int16, Bool
from cv_bridge import CvBridge
from typing import List
import torch.nn.functional as F

from ai_module.src.utils.logger import Logger


DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "ViT-B/32"
IMAGE_TOPIC   = "/camera/image/compressed"
SCORES_TOPIC  = "/clip/segment_scores"

NUM_SEGMENTS  = 4
OVERLAP_RATIO = 0.2 

model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

TOTAL_TASK = None
CURRENT_TASK = None
TARGET_PROMPT = None
prompt_feature = None

# Multi-task support
ALL_TASK_PROMPTS = []
all_prompt_features = None

bridge = CvBridge()
scores_pub = None 
debug_img_pub = None
ready_pub = None
all_task_scores_pub = None  # Publisher for all task scores

# VG task management
ALL_STEPS = []
CURRENT_STEP_IDX = -1

bridge = CvBridge()
scores_pub = None 
reset_values_pub = None  # Publisher to notify value map reset


def steps_callback(msg: String):
    """Callback for receiving task steps from VG"""
    global ALL_STEPS, ALL_TASK_PROMPTS, all_prompt_features, TOTAL_TASK, CURRENT_STEP_IDX, prompt_feature

    if TOTAL_TASK == msg.data:
        return

    else:
        if msg.data:
            TOTAL_TASK = msg.data

            ALL_STEPS = msg.data.split("/")
            rospy.loginfo(f"[CLIP] Received {len(ALL_STEPS)} task steps: {ALL_STEPS}")
            
            # Generate prompts for all tasks immediately
            ALL_TASK_PROMPTS = []
            for step in ALL_STEPS:
                task_prompt = f"{step.strip()}."
                ALL_TASK_PROMPTS.append(task_prompt)
            
            rospy.loginfo(f"[CLIP] Generated task prompts: {ALL_TASK_PROMPTS}")
            
            # Encode all prompts at once
            with torch.no_grad():
                if ALL_TASK_PROMPTS:
                    prompt_tokens = clip.tokenize(ALL_TASK_PROMPTS).to(DEVICE)
                    all_prompt_features = model.encode_text(prompt_tokens)  # Shape: (num_tasks, feature_dim)
                    rospy.loginfo(f"[CLIP] Encoded {len(ALL_TASK_PROMPTS)} task prompts")
                    rospy.loginfo(f"[CLIP] all_prompt_features shape: {all_prompt_features.shape}")
                    
                    # If we already have a current step index, initialize prompt_feature immediately
                    if 0 <= CURRENT_STEP_IDX < all_prompt_features.shape[0]:
                        prompt_feature = all_prompt_features[CURRENT_STEP_IDX:CURRENT_STEP_IDX+1]
                        rospy.loginfo(f"[CLIP] Immediately initialized prompt_feature for step {CURRENT_STEP_IDX}")
                        rospy.loginfo(f"[CLIP] prompt_feature shape: {prompt_feature.shape}")
                    else:
                        rospy.loginfo(f"[CLIP] Current step index {CURRENT_STEP_IDX} not valid yet")
        else:
            ALL_STEPS = []
            ALL_TASK_PROMPTS = []
            all_prompt_features = None
            prompt_feature = None
            rospy.loginfo(f"[CLIP] No task steps received")


def step_idx_callback(msg):
    """Callback for receiving current step index from VG"""
    global CURRENT_STEP_IDX, CURRENT_TASK, TARGET_PROMPT, prompt_feature, all_prompt_features
    
    if CURRENT_STEP_IDX == msg.data:
        return
    
    new_idx = msg.data
    
    # Check if step index has changed
    if CURRENT_STEP_IDX != new_idx:
        old_idx = CURRENT_STEP_IDX
        CURRENT_STEP_IDX = new_idx
        
        rospy.loginfo(f"[CLIP] Step index changed from {old_idx} to {new_idx}")
        
        # NO LONGER RESET VALUE MAPS - we use pre-computed multi-task maps
        # Signal the frontier mapper to switch to the correct task map
        if reset_values_pub is not None:
            reset_values_pub.publish(Bool(False))  # False = switch task map, don't reset
            rospy.loginfo(f"[CLIP] Published task switch signal")
    
    # Update current task info for compatibility
    if 0 <= CURRENT_STEP_IDX < len(ALL_STEPS):
        new_task = ALL_STEPS[CURRENT_STEP_IDX].strip()
        
        if CURRENT_TASK != new_task:
            CURRENT_TASK = new_task
            rospy.loginfo(f"[CLIP] Current task updated to: '{CURRENT_TASK}'")
            
            # Keep single prompt for compatibility
            TARGET_PROMPT = f"Seems like I can find {CURRENT_TASK} ahead."
            
            # Debug information before setting prompt_feature
            rospy.loginfo(f"[CLIP] DEBUG: all_prompt_features is None: {all_prompt_features is None}")
            if all_prompt_features is not None:
                rospy.loginfo(f"[CLIP] DEBUG: all_prompt_features.shape: {all_prompt_features.shape}")
                rospy.loginfo(f"[CLIP] DEBUG: CURRENT_STEP_IDX: {CURRENT_STEP_IDX}")
                rospy.loginfo(f"[CLIP] DEBUG: Condition check: {0 <= CURRENT_STEP_IDX < all_prompt_features.shape[0]}")
            
            # Try to set prompt_feature, with retry logic if all_prompt_features isn't ready yet
            if all_prompt_features is not None and 0 <= CURRENT_STEP_IDX < all_prompt_features.shape[0]:
                prompt_feature = all_prompt_features[CURRENT_STEP_IDX:CURRENT_STEP_IDX+1]  # Keep batch dimension
                rospy.loginfo(f"[CLIP] prompt_feature initialized for task {CURRENT_STEP_IDX}")
                rospy.loginfo(f"[CLIP] prompt_feature shape: {prompt_feature.shape}")
                ready_pub.publish(True)
                rospy.loginfo(f"[CLIP] Updated to task {CURRENT_STEP_IDX}: '{TARGET_PROMPT}'")
            else:
                rospy.logwarn(f"[CLIP] Could not set prompt_feature for step {CURRENT_STEP_IDX} - will retry")
                if all_prompt_features is None:
                    rospy.logwarn(f"[CLIP] - all_prompt_features is None, scheduling retry...")
                    # Schedule a retry after a short delay
                    rospy.Timer(rospy.Duration(0.1), lambda event: _retry_set_prompt_feature(CURRENT_STEP_IDX), oneshot=True)
                else:
                    rospy.logwarn(f"[CLIP] - all_prompt_features.shape: {all_prompt_features.shape}")
                    rospy.logwarn(f"[CLIP] - CURRENT_STEP_IDX: {CURRENT_STEP_IDX}")
                    ready_pub.publish(False)
        else:
            # Task didn't change, but we should still ensure prompt_feature is set
            if all_prompt_features is not None and 0 <= CURRENT_STEP_IDX < all_prompt_features.shape[0]:
                if prompt_feature is None:
                    prompt_feature = all_prompt_features[CURRENT_STEP_IDX:CURRENT_STEP_IDX+1]
                    rospy.loginfo(f"[CLIP] prompt_feature set for existing task {CURRENT_STEP_IDX}")
                ready_pub.publish(True)
            else:
                rospy.logwarn(f"[CLIP] Cannot set prompt_feature for existing task {CURRENT_STEP_IDX}")
                ready_pub.publish(False)
    else:
        CURRENT_TASK = None
        TARGET_PROMPT = None
        prompt_feature = None
        rospy.logwarn(f"[CLIP] Invalid step index {CURRENT_STEP_IDX} for {len(ALL_STEPS)} steps")


def _retry_set_prompt_feature(step_idx):
    """Retry setting prompt_feature after a delay"""
    global prompt_feature, all_prompt_features
    
    rospy.loginfo(f"[CLIP] Retrying to set prompt_feature for step {step_idx}")
    
    if all_prompt_features is not None and 0 <= step_idx < all_prompt_features.shape[0]:
        prompt_feature = all_prompt_features[step_idx:step_idx+1]
        rospy.loginfo(f"[CLIP] RETRY SUCCESS: prompt_feature set for step {step_idx}")
        rospy.loginfo(f"[CLIP] prompt_feature shape: {prompt_feature.shape}")
        ready_pub.publish(True)
    else:
        rospy.logwarn(f"[CLIP] RETRY FAILED: Still cannot set prompt_feature for step {step_idx}")
        ready_pub.publish(False)


def split_image(img: Image.Image,
                num_segments: int,
                overlap_ratio: float = OVERLAP_RATIO) -> List[Image.Image]:
    w, h  = img.size
    seg_w = w // num_segments
    ov    = int(seg_w * overlap_ratio)
    segs  = []
    for i in range(num_segments):
        c = i * seg_w + seg_w//2
        l = max(0, c - seg_w//2 - ov)
        r = min(w, c + seg_w//2 + ov)
        segs.append(img.crop((l, 0, r, h)))
    return segs

def encode_image_segments(segments: List[Image.Image]) -> torch.Tensor:
    batch = torch.stack([preprocess(s) for s in segments], dim=0).to(DEVICE)
    with torch.no_grad():
        return model.encode_image(batch)  # (N, D)

def compute_target_cosines(image_features: torch.Tensor) -> List[float]:
    """Compute cosines for current task (backward compatibility)"""
    img_norm = F.normalize(image_features, p=2, dim=1)   # (N, D)
    txt_norm = F.normalize(prompt_feature,    p=2, dim=1)   # (1, D)
    cos_sim  = img_norm @ txt_norm.T                    # (N, 1)
    return cos_sim[:,0].cpu().tolist()

def compute_all_task_cosines(image_features: torch.Tensor) -> List[List[float]]:
    global all_prompt_features

    """Compute cosines for all tasks simultaneously"""
    if all_prompt_features is None:
        return []
    
    img_norm = F.normalize(image_features, p=2, dim=1)      # (N, D) - N segments
    txt_norm = F.normalize(all_prompt_features, p=2, dim=1) # (T, D) - T tasks
    cos_sim  = img_norm @ txt_norm.T                       # (N, T)
    
    # Convert to list of lists: [task][segment]
    result = []
    for task_idx in range(cos_sim.shape[1]):  # For each task
        task_scores = cos_sim[:, task_idx].cpu().tolist()  # Scores for all segments
        result.append(task_scores)
    
    return result

def image_callback(msg: CompressedImage):
    global scores_pub, prompt_feature, all_task_scores_pub, all_prompt_features

    # Check if we have any prompt features - be more specific about what we need
    if all_prompt_features is None:
        rospy.logwarn_throttle(5, "[CLIP] waiting for task steps...")
        ready_pub.publish(False)
        return
    
    if CURRENT_STEP_IDX < 0 or CURRENT_STEP_IDX >= len(ALL_STEPS):
        rospy.logwarn_throttle(5, f"[CLIP] invalid step index {CURRENT_STEP_IDX} for {len(ALL_STEPS)} steps...")
        ready_pub.publish(False)
        return
    
    if prompt_feature is None:
        rospy.logwarn_throttle(5, f"[CLIP] prompt feature not set for step {CURRENT_STEP_IDX}...")
        ready_pub.publish(False)
        return

    # If we reach here, we're ready to process
    ready_pub.publish(True)

    arr    = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        rospy.logerr("[CLIP] failed to decode compressed image")
        return

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    width = cv_img.shape[1]
    mid_point = width // 2
    left_half = cv_img[:, :mid_point, :]
    right_half = cv_img[:, mid_point:, :]
    rotated_cv_img = np.concatenate([right_half, left_half], axis=1)
    
    shift_amount = width // 8
    rotated_cv_img = np.roll(rotated_cv_img, shift=-shift_amount, axis=1)    
    
    pil_img = Image.fromarray(rotated_cv_img)
    
    segs      = split_image(pil_img, NUM_SEGMENTS)
    img_feats = encode_image_segments(segs)
    
    # Compute scores for current task (backward compatibility)
    cosines = compute_target_cosines(img_feats)
    
    # Compute scores for all tasks
    all_task_cosines = compute_all_task_cosines(img_feats)
      
    if segs:
        front_segment_pil = segs[1]
        front_segment_cv = np.array(front_segment_pil)
        front_segment_cv = cv2.cvtColor(front_segment_cv, cv2.COLOR_RGB2BGR)
        try:
            img_msg = bridge.cv2_to_imgmsg(front_segment_cv, "bgr8")
            if debug_img_pub:
                debug_img_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish debug image: {e}")
    
    # Publish current task scores (backward compatibility)        
    fa = Float32MultiArray()
    fa.layout.dim = [
        MultiArrayDimension(label="segment", size=NUM_SEGMENTS, stride=NUM_SEGMENTS * 1),
        MultiArrayDimension(label="label",   size=1,             stride=1)
    ]
    fa.data = [float(v) for v in cosines]
    scores_pub.publish(fa)
    
    # Publish all task scores
    if all_task_scores_pub and all_task_cosines:
        all_fa = Float32MultiArray()
        num_tasks = len(all_task_cosines)
        
        # Flatten the data: [task0_seg0, task0_seg1, ..., task1_seg0, task1_seg1, ...]
        flattened_data = []
        for task_scores in all_task_cosines:
            flattened_data.extend(task_scores)
        
        all_fa.layout.dim = [
            MultiArrayDimension(label="task", size=num_tasks, stride=num_tasks * NUM_SEGMENTS),
            MultiArrayDimension(label="segment", size=NUM_SEGMENTS, stride=NUM_SEGMENTS)
        ]
        all_fa.data = [float(v) for v in flattened_data]
        all_task_scores_pub.publish(all_fa)

def main():
    global scores_pub, reset_values_pub, debug_img_pub, ready_pub, all_task_scores_pub
    rospy.init_node("clip_segment_processor", anonymous=False)
    
    quiet = rospy.get_param('~quiet', False)
    logger = Logger(quiet=quiet, prefix='360CLIP')
    
    # Publishers
    scores_pub = rospy.Publisher(SCORES_TOPIC, Float32MultiArray, queue_size=1)
    all_task_scores_pub = rospy.Publisher("/clip/all_task_scores", Float32MultiArray, queue_size=1)
    reset_values_pub = rospy.Publisher("/clip/reset_values", Bool, queue_size=1)
    ready_pub = rospy.Publisher("/clip/ready", Bool, queue_size=1)
    debug_img_pub = rospy.Publisher("/debug/front_segment_image", RosImage, queue_size=1)
    
    # Subscribers for VG task system
    steps_sub = rospy.Subscriber("/steps", String, steps_callback, queue_size=1)
    step_idx_sub = rospy.Subscriber("/current_step_idx", Int16, step_idx_callback, queue_size=1)
    
    # Image subscriber
    rospy.Subscriber(IMAGE_TOPIC, CompressedImage, image_callback,
                     queue_size=1, buff_size=2**24)
    
    logger.loginfo("CLIP Node initialized with VG task integration")
    logger.loginfo("Subscribing to /steps and /current_step_idx from VG")
    rospy.spin()

if __name__ == "__main__":
    main()
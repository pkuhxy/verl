# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import math


def compute_score(
    predict_str: str, 
    ground_truth: str, 
    extra_info: dict = None,
    iou_weight: float = 0.4,
    dist_weight: float = 0.3,
    consist_weight: float = 0.3,
    alpha: float = 0.5
) -> float:
    """
    Compute the reward score for VTG Stage 2.
    
    :param predict_str: Model's prediction output string with CoT format
    :param ground_truth: Ground truth JSON string, e.g., "[0.0, 5.0]"
    :param extra_info: Dictionary containing additional information, including:
                       - 'duration': Total video duration for normalization.
                       If None or 'duration' not present, uses gt_end as normalization factor.
    :param iou_weight: Weight for IoU reward (default 0.4)
    :param dist_weight: Weight for boundary distance reward (default 0.3)
    :param consist_weight: Weight for consistency reward (default 0.3)
    :param alpha: Decay factor for consistency reward (default 0.5)
    :return: Combined reward score
    """
    # Extract video_duration from extra_info if available
    video_duration = None
    if extra_info is not None and isinstance(extra_info, dict):
        video_duration = extra_info.get('duration', None)
    
    # 1. Extract final result from prediction
    final_result = extract_final_result(predict_str)
    

    # print(f"extra_info: {extra_info}")

    # 2. If format is wrong, return 0.0
    if final_result is None:
        return 0.0
    
    pred_start, pred_end = final_result
    
    # 3. Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
        gt_start, gt_end = float(gt_list[0]), float(gt_list[1])
    except:
        return 0.0
    
    # 4. Determine normalization factor D
    if video_duration is not None and video_duration > 0:
        D = video_duration
    else:
        # Use gt_end as fallback normalization factor
        D = max(gt_end, pred_end, 1.0)
    
    # 5. Compute IoU Reward
    r_iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
    
    # 6. Compute Boundary Distance Reward
    r_dist = compute_boundary_distance_reward(pred_start, pred_end, gt_start, gt_end, D)
    
    # 7. Compute Consistency Reward
    state_changes = extract_state_changes(predict_str)
    r_consist = compute_consistency_reward(state_changes, pred_start, pred_end, alpha)
    
    # 8. Combine rewards with weights
    final_score = iou_weight * r_iou + dist_weight * r_dist + consist_weight * r_consist
    
    return final_score


def extract_final_result(text: str):
    """
    Extract the time span from <FINAL_RESULT> section.
    
    :param text: Full prediction text
    :return: (start_time, end_time) or None if not found
    """
    # Pattern to match <FINAL_RESULT> section with [start, end] format
    pattern = r"<FINAL_RESULT>\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]\s*</FINAL_RESULT>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Fallback: try to find any [number, number] pattern in FINAL_RESULT section
    section_pattern = r"<FINAL_RESULT>(.*?)</FINAL_RESULT>"
    section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
    if section_match:
        content = section_match.group(1)
        bracket_pattern = r"\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]"
        bracket_match = re.search(bracket_pattern, content)
        if bracket_match:
            return float(bracket_match.group(1)), float(bracket_match.group(2))
    
    return None


def extract_state_changes(text: str) -> dict:
    """
    Extract STATE_CHANGE timestamps from the CoT reasoning.
    
    :param text: Full prediction text
    :return: Dictionary with 'start' and 'end' times from STATE_CHANGE tags
    """
    result = {'start': None, 'end': None}
    
    # Pattern to match <STATE_CHANGE type="START" time="X">
    start_pattern = r'<STATE_CHANGE\s+type=["\']START["\']\s+time=["\'](\d+(?:\.\d+)?)["\']>'
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    if start_match:
        result['start'] = float(start_match.group(1))
    
    # Pattern to match <STATE_CHANGE type="END" time="X">
    end_pattern = r'<STATE_CHANGE\s+type=["\']END["\']\s+time=["\'](\d+(?:\.\d+)?)["\']>'
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    if end_match:
        result['end'] = float(end_match.group(1))
    
    return result


def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Compute temporal IoU between prediction and ground truth.
    
    :param pred_start: Predicted start time
    :param pred_end: Predicted end time
    :param gt_start: Ground truth start time
    :param gt_end: Ground truth end time
    :return: IoU value in [0, 1]
    """
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 1e-9:
        return 0.0
    
    return intersection / union


def compute_boundary_distance_reward(
    pred_start: float, 
    pred_end: float, 
    gt_start: float, 
    gt_end: float, 
    D: float
) -> float:
    """
    Compute boundary distance reward.
    
    R_dist = 1 - 0.5 * (|s_pred - s_gt| / D + |e_pred - e_gt| / D)
    
    :param pred_start: Predicted start time
    :param pred_end: Predicted end time
    :param gt_start: Ground truth start time
    :param gt_end: Ground truth end time
    :param D: Normalization factor (video duration)
    :return: Boundary distance reward in [0, 1]
    """
    if D <= 1e-9:
        return 0.0
    
    start_error = abs(pred_start - gt_start) / D
    end_error = abs(pred_end - gt_end) / D
    
    r_dist = 1.0 - 0.5 * (start_error + end_error)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, r_dist))


def compute_consistency_reward(
    state_changes: dict, 
    pred_start: float, 
    pred_end: float, 
    alpha: float
) -> float:
    """
    Compute consistency reward between STATE_CHANGE timestamps and FINAL_RESULT.
    
    R_consist = exp(-alpha * |T_state - T_final|)
    
    If both start and end STATE_CHANGE are present, average the rewards.
    If no STATE_CHANGE is found, return 0.0 (penalize missing reasoning).
    
    :param state_changes: Dictionary with 'start' and 'end' times
    :param pred_start: Final predicted start time
    :param pred_end: Final predicted end time
    :param alpha: Decay factor
    :return: Consistency reward in [0, 1]
    """
    rewards = []
    
    if state_changes['start'] is not None:
        start_diff = abs(state_changes['start'] - pred_start)
        r_start = math.exp(-alpha * start_diff)
        rewards.append(r_start)
    
    if state_changes['end'] is not None:
        end_diff = abs(state_changes['end'] - pred_end)
        r_end = math.exp(-alpha * end_diff)
        rewards.append(r_end)
    
    if len(rewards) == 0:
        # No STATE_CHANGE found, penalize missing reasoning
        return 0.0
    
    return sum(rewards) / len(rewards)


# Alias for compatibility
def acc_reward_iou(result, ground_truth: str) -> tuple:
    """
    Compute IoU and check if prediction covers ground truth.
    Kept for compatibility with Tiou.py style.
    
    :param result: Prediction time range (start, end)
    :param ground_truth: Ground truth JSON string
    :return: (iou, is_covering)
    """
    if result is None:
        return 0.0, False
    
    pred_start, pred_end = result
    try:
        gt_list = json.loads(ground_truth)
        gt_start, gt_end = float(gt_list[0]), float(gt_list[1])
    except:
        return 0.0, False
    
    iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
    
    # Check if prediction fully covers ground truth
    is_covering = (pred_start <= gt_start + 1e-9) and (pred_end >= gt_end - 1e-9)
    
    return iou, is_covering

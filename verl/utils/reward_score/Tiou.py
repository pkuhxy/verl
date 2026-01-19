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



def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.0) -> float:
    """
    计算分数。
    
    :param predict_str: 模型的预测输出字符串
    :param ground_truth: 真实值的 JSON 字符串，例如 "[0.0, 5.0]"
    :param format_score: 格式正确的基础得分权重。
                          如果为 0.0，则完全由 IoU 决定分数（推荐）。
                          如果 > 0.0，则只要格式对就有基础分。
    :return: float 分数
    """
    
    # 1. 提取时间片段
    result = extract_time_span(predict_str)
    
    # 2. 如果格式错误 (提取不到)，直接返回 0.0
    if result is None:
        return 0.0
    
    # 3. 格式正确，计算 IoU
    iou = acc_reward_iou(result, ground_truth)
    
    # 4. 计算最终得分
    # 逻辑：总分 = (1 - 权重) * IoU得分 + 权重 * 格式得分(1.0)
    # 如果 format_score 是 0，则分数就是 IoU。
    final_score = (1.0 - format_score) * iou + format_score * 1.0
    
    return final_score

# --- 依赖的辅助函数 (保持不变) ---
def extract_time_span(text):
    pattern = r"The event happens in\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+seconds"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

def acc_reward_iou(result, ground_truth: str) -> float:
    if result is None: return 0.0
    pred_start, pred_end = result
    try:
        gt_list = json.loads(ground_truth)
        gt_start, gt_end = float(gt_list[0]), float(gt_list[1])
    except: return 0.0

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 1e-9: return 0.0
    return intersection / union

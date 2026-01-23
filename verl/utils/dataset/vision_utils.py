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

from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    return fetch_image(image, image_patch_size=image_patch_size)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    min_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    total_pixels: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Args:
        video: Video dict containing video path or frames.
        image_patch_size: Patch size for image processing.
        nframes: Number of frames to sample (mutually exclusive with fps).
        fps: Frames per second for sampling, i.e., number of frames to sample per second (mutually exclusive with nframes).
        min_frames: Minimum frames when using fps sampling.
        max_frames: Maximum frames when using fps sampling.
        min_pixels: Minimum total pixels for the video.
        max_pixels: Maximum total pixels for the video to control vision token count.
            This limits the video resolution/frames to control memory usage.
        total_pixels: Total pixels budget for the video.
        return_video_sample_fps: Whether to return the sample fps.
        return_video_metadata: Whether to return video metadata.
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if min_frames is not None:
                video["min_frames"] = min_frames
            if max_frames is not None:
                video["max_frames"] = max_frames

    # Add min_pixels to control vision token count if specified
    if min_pixels is not None and "min_pixels" not in video:
        video["min_pixels"] = min_pixels

    # Add max_pixels to control vision token count if specified
    if max_pixels is not None and "max_pixels" not in video:
        video["max_pixels"] = max_pixels

    # Add total_pixels to control vision token count if specified
    if total_pixels is not None and "total_pixels" not in video:
        video["total_pixels"] = total_pixels

    return fetch_video(
        video,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}

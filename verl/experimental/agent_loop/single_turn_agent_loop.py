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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # breakpoint()

        # 2. apply chat template and tokenize
        # For VL models with video (e.g., Qwen3VL), use text prompt to let vLLM
        # handle multi-modal tokenization internally
        has_video = videos is not None and len(videos) > 0
        
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
            return_text=has_video  # Return text prompt for video, token ids otherwise
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        response_mask = [1] * len(output.token_ids)

        # For text prompt (video case), we need to tokenize the prompt to get token ids
        # The actual prompt used by vLLM is the text, but we need token ids for output
        if has_video and isinstance(prompt_ids, str):
            # Use processor to tokenize the text prompt for output
            if self.processor is not None:
                # split the videos and according metadatas
                if videos is not None:
                    videos_tensor, video_metadata = zip(*videos, strict=False)
                    videos_tensor, video_metadata = list(videos_tensor), list(video_metadata)
                else:
                    video_metadata = None

                model_inputs = self.processor(
                    text=[prompt_ids],
                    images=images,
                    videos=videos_tensor,
                    video_metadata=video_metadata,
                    return_tensors="pt",
                    do_sample_frames=False,
                )
                prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
            else:
                prompt_ids = self.tokenizer.encode(prompt_ids, add_special_tokens=False)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output

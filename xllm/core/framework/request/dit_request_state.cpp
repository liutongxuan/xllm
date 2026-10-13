/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/request/dit_request_state.h"

#include <utility>

#include "worker.pb.h"

namespace xllm {

bool DiTGenerationParams::to_proto(
    proto::DiTGenerationParams* proto_params) const {
  if (proto_params == nullptr) {
    return false;
  }
  proto_params->Clear();
  proto_params->set_width(width);
  proto_params->set_height(height);
  proto_params->set_num_inference_steps(num_inference_steps);
  proto_params->set_true_cfg_scale(true_cfg_scale);
  proto_params->set_guidance_scale(guidance_scale);
  proto_params->set_num_images_per_prompt(num_images_per_prompt);
  proto_params->set_seed_is_set(seed_is_set);
  if (seed_is_set) {
    proto_params->set_seed(seed);
  }
  proto_params->set_max_sequence_length(max_sequence_length);
  proto_params->set_strength(strength);
  proto_params->set_enable_cfg_renorm(enable_cfg_renorm);
  proto_params->set_cfg_renorm_min(cfg_renorm_min);
  proto_params->set_num_frames(num_frames);
  proto_params->set_video_fps(video_fps);
  proto_params->set_guidance_scale_2(guidance_scale_2);
  proto_params->set_seconds(seconds);
  proto_params->set_boundary_ratio(boundary_ratio);
  proto_params->set_flow_shift(flow_shift);
  proto_params->set_num_videos_per_prompt(num_videos_per_prompt);
  // Text diffusion params
  if (max_new_tokens > 0) {
    proto_params->set_max_new_tokens(max_new_tokens);
  }
  if (diffusion_steps > 0) {
    proto_params->set_diffusion_steps(diffusion_steps);
  }
  proto_params->set_temperature(temperature);
  proto_params->set_top_k(top_k);
  proto_params->set_top_p(top_p);
  proto_params->set_repetition_penalty(repetition_penalty);
  proto_params->set_audio_duration_frames(audio_duration_frames);
  proto_params->set_audio_steps(audio_steps);
  proto_params->set_audio_guidance_method(audio_guidance_method);
  proto_params->set_audio_sampling_rate(audio_sampling_rate);
  return true;
}

bool DiTGenerationParams::from_proto(
    const proto::DiTGenerationParams& proto_params) {
  DiTGenerationParams decoded_params;
  decoded_params.width = proto_params.width();
  decoded_params.height = proto_params.height();
  decoded_params.num_inference_steps = proto_params.num_inference_steps();
  decoded_params.true_cfg_scale = proto_params.true_cfg_scale();
  decoded_params.guidance_scale = proto_params.guidance_scale();
  decoded_params.num_images_per_prompt = proto_params.num_images_per_prompt();
  if (proto_params.seed_is_set()) {
    decoded_params.seed_is_set = true;
    if (proto_params.has_seed()) {
      decoded_params.seed = proto_params.seed();
    }
  } else if (proto_params.has_seed()) {
    // Backward compat: messages sent before seed_is_set existed.
    decoded_params.seed = proto_params.seed();
    decoded_params.seed_is_set = true;
  }
  decoded_params.max_sequence_length = proto_params.max_sequence_length();
  decoded_params.strength = proto_params.strength();
  decoded_params.enable_cfg_renorm = proto_params.enable_cfg_renorm();
  decoded_params.cfg_renorm_min = proto_params.cfg_renorm_min();
  decoded_params.num_frames = proto_params.num_frames();
  decoded_params.video_fps = proto_params.video_fps();
  decoded_params.guidance_scale_2 = proto_params.guidance_scale_2();
  decoded_params.seconds = proto_params.seconds();
  decoded_params.boundary_ratio = proto_params.boundary_ratio();
  decoded_params.flow_shift = proto_params.flow_shift();
  decoded_params.num_videos_per_prompt = proto_params.num_videos_per_prompt();
  // Text diffusion params
  if (proto_params.has_max_new_tokens()) {
    decoded_params.max_new_tokens = proto_params.max_new_tokens();
  }
  if (proto_params.has_diffusion_steps()) {
    decoded_params.diffusion_steps = proto_params.diffusion_steps();
  }
  if (proto_params.has_temperature()) {
    decoded_params.temperature = proto_params.temperature();
  }
  if (proto_params.has_top_k()) {
    decoded_params.top_k = proto_params.top_k();
  }
  if (proto_params.has_top_p()) {
    decoded_params.top_p = proto_params.top_p();
  }
  if (proto_params.has_repetition_penalty()) {
    decoded_params.repetition_penalty = proto_params.repetition_penalty();
  }
  if (proto_params.has_audio_duration_frames()) {
    decoded_params.audio_duration_frames = proto_params.audio_duration_frames();
  }
  if (proto_params.has_audio_steps()) {
    decoded_params.audio_steps = proto_params.audio_steps();
  }
  if (proto_params.has_audio_guidance_method()) {
    decoded_params.audio_guidance_method = proto_params.audio_guidance_method();
  }
  if (proto_params.has_audio_sampling_rate()) {
    decoded_params.audio_sampling_rate = proto_params.audio_sampling_rate();
  }
  *this = std::move(decoded_params);
  return true;
}

}  // namespace xllm

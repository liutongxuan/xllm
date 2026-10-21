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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/request/dit_request_state.h"

namespace xllm {

// Inputs consumed by DiT model pipelines.
class DiTModelParams final {
 public:
  DiTModelParams to(const torch::Device& device,
                    torch::ScalarType dtype = torch::kBFloat16) const {
    DiTModelParams params = *this;
    params.tensor_sources = tensor_sources.to(device, dtype);
    params.media_sources = media_sources.to(device);
    return params;
  }

  int32_t batch_size = 0;

  // Primary input text description for image generation.
  std::vector<std::string> prompts;

  // Secondary prompt for additional details (e.g., color, lighting).
  std::vector<std::string> prompts_2;

  // Negative prompt to exclude low-quality features.
  std::vector<std::string> negative_prompts;

  // Secondary negative prompt to exclude additional unwanted features.
  std::vector<std::string> negative_prompts_2;

  DiTMediaSources media_sources;
  DiTTensorSources tensor_sources;

  // Transcript of the prompt audio, used for duration estimation only.
  std::string audio_prompt_text;

  DiTGenerationParams generation_params;
};

}  // namespace xllm

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <optional>

#include "model_input_param_groups.h"
#include "model_input_params.h"

namespace xllm {
namespace model_input {

// `ModelInput` is the typed view of a model's per-call input, partitioned by
// model family (llm / vlm / dit / rec). It coexists with the legacy
// `xllm::ModelInputParams` while migration is in progress:
//
//   * Producers (workers/tests) typically still hand out legacy params and
//     `make_model_input_from_legacy` builds the typed view on the boundary.
//   * Model bases declare typed `forward(...)` overloads; `CausalLMImpl` /
//     `RecCausalLMImpl` / `CausalVLMImpl` use SFINAE traits in
//     `model_traits.h` to route calls directly to the model when it consumes
//     typed input, otherwise the base default falls back to
//     `apply_model_input_to_legacy` and the legacy `forward(...)`.
//   * To opt a model into the typed path, give it a
//     `forward(..., const ModelInput&)` (and ideally a `&&` overload). The
//     trait will pick it up automatically.
struct ModelInput {
  std::optional<LLMModelInputParams> llm;
  std::optional<VLMModelInputParams> vlm;
  std::optional<DitModelInputParams> dit;
  std::optional<RecModelInputParams> rec;
};

ModelInput make_model_input_from_legacy(const xllm::ModelInputParams& params);
ModelInput make_model_input_from_legacy(xllm::ModelInputParams&& params);
void apply_model_input_to_legacy(const ModelInput& src,
                                 xllm::ModelInputParams* dst);
void apply_model_input_to_legacy(ModelInput&& src, xllm::ModelInputParams* dst);

bool has_llm(const ModelInput& input);
bool has_vlm(const ModelInput& input);
bool has_dit(const ModelInput& input);
bool has_rec(const ModelInput& input);

}  // namespace model_input
}  // namespace xllm

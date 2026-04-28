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

#include "model_input.h"

#include <utility>

namespace xllm {
namespace model_input {

ModelInput make_model_input_from_legacy(const xllm::ModelInputParams& params) {
  ModelInput model_input;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(params);
  model_input.llm = std::move(bundle.llm);
  model_input.vlm = std::move(bundle.vlm);
  model_input.dit = std::move(bundle.dit);
  model_input.rec = std::move(bundle.rec);
  return model_input;
}

ModelInput make_model_input_from_legacy(xllm::ModelInputParams&& params) {
  ModelInput model_input;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(std::move(params));
  model_input.llm = std::move(bundle.llm);
  model_input.vlm = std::move(bundle.vlm);
  model_input.dit = std::move(bundle.dit);
  model_input.rec = std::move(bundle.rec);
  return model_input;
}

void apply_model_input_to_legacy(const ModelInput& src,
                                 xllm::ModelInputParams* params) {
  if (src.llm.has_value()) {
    apply_llm_model_input_params_to_legacy(*src.llm, params);
  }
  if (src.vlm.has_value()) {
    apply_vlm_model_input_params_to_legacy(*src.vlm, params);
  }
  if (src.dit.has_value()) {
    apply_dit_model_input_params_to_legacy(*src.dit, params);
  }
  if (src.rec.has_value()) {
    apply_rec_model_input_params_to_legacy(*src.rec, params);
  }
}

void apply_model_input_to_legacy(ModelInput&& src,
                                 xllm::ModelInputParams* params) {
  if (src.llm.has_value()) {
    apply_llm_model_input_params_to_legacy(std::move(*src.llm), params);
  }
  if (src.vlm.has_value()) {
    apply_vlm_model_input_params_to_legacy(std::move(*src.vlm), params);
  }
  if (src.dit.has_value()) {
    apply_dit_model_input_params_to_legacy(std::move(*src.dit), params);
  }
  if (src.rec.has_value()) {
    apply_rec_model_input_params_to_legacy(std::move(*src.rec), params);
  }
}

bool has_llm(const ModelInput& input) { return input.llm.has_value(); }

bool has_vlm(const ModelInput& input) { return input.vlm.has_value(); }

bool has_dit(const ModelInput& input) { return input.dit.has_value(); }

bool has_rec(const ModelInput& input) { return input.rec.has_value(); }

}  // namespace model_input
}  // namespace xllm

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

#include "core/framework/model/model_input_params.h"
#include "core/framework/model/rec_model_params.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm {

inline const std::vector<torch::Tensor>& qwen_deep_stacks(
    const ModelInputParams& params) {
  static const std::vector<torch::Tensor> empty;
  return params.has_multimodal() ? params.multimodal().deep_stacks : empty;
}

inline const std::vector<torch::Tensor>& qwen_deep_stacks(
    const RecModelParams&) {
  static const std::vector<torch::Tensor> empty;
  return empty;
}

inline bool qwen_is_spec_verify(const ModelInputParams& params) {
  return params.is_spec_verify;
}

inline bool qwen_is_spec_verify(const RecModelParams&) { return false; }

inline void prepare_qwen_rec_layer(const ModelInputParams&,
                                   layer::AttentionMetadata&,
                                   size_t,
                                   size_t) {}

inline void prepare_qwen_rec_layer(const RecModelParams& params,
                                   layer::AttentionMetadata& metadata,
                                   size_t layer_id,
                                   size_t layer_count) {
  const auto* rec = params.llmrec_params();
  if (rec == nullptr) {
    return;
  }
  CHECK_EQ(rec->full_k_caches.size(), layer_count);
  CHECK_EQ(rec->full_v_caches.size(), layer_count);
  CHECK_EQ(rec->unshared_k_caches.size(), layer_count);
  CHECK_EQ(rec->unshared_v_caches.size(), layer_count);
  metadata.full_k_cache = rec->full_k_caches[layer_id];
  metadata.full_v_cache = rec->full_v_caches[layer_id];
  metadata.unshared_k_cache = rec->unshared_k_caches[layer_id];
  metadata.unshared_v_cache = rec->unshared_v_caches[layer_id];
}

}  // namespace xllm

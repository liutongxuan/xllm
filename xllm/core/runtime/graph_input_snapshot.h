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

#include "core/framework/model/model_params_view.h"
#include "core/framework/model/rec_model_params.h"
#include "core/util/rec_model_utils.h"

namespace xllm {

// ModelParamsView retains its cloned owner; Rec snapshots own the native
// aggregate directly. Tensor handles intentionally share storage.
inline ModelParamsView clone_graph_input(const ModelParamsView& params) {
  return params.clone().view();
}

inline RecModelParams clone_graph_input(const RecModelParams& params) {
  return params.clone();
}

inline bool is_rec_multi_round_graph_input(const ModelParamsView&) {
  return false;
}

inline bool is_rec_multi_round_graph_input(const RecModelParams&) {
  return is_rec_multi_round_mode();
}

inline bool has_llmrec_graph_input(const ModelParamsView&) { return false; }

inline bool has_llmrec_graph_input(const RecModelParams& params) {
  return params.has_llmrec_params();
}

}  // namespace xllm

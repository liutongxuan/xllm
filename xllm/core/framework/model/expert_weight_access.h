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

#include <cstdint>
#include <vector>

namespace xllm {

// Weight operations required by expert load balancing, independent of forward.
class ExpertWeightAccess {
 public:
  virtual ~ExpertWeightAccess() = default;

  virtual void prepare_expert_weight(
      int32_t layer_id,
      const std::vector<int32_t>& expert_ids) = 0;
  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}
  virtual void update_expert_weight(int32_t layer_id) = 0;

  // Returns whether the last prepare_expert_weight() call for the given layer
  // succeeded. Default is true so models that do not fail-report keep the old
  // behavior; DSV4 (npu_torch) overrides this to expose real prepare status
  // so EplbExecutor does not advance ready_layer_id on silent failures.
  virtual bool last_prepare_expert_weight_ok(int32_t /*layer_id*/) const {
    return true;
  }
};

}  // namespace xllm

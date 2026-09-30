/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <unordered_map>
#include <vector>

namespace xllm {

class BatchStorage;

// Shared sequence ordering policy, independent of ordinary and Rec input
// builders. Ordering finishes before the per-forward sampling snapshot.
class BatchSequenceOrdering final {
 public:
  static void prepare(BatchStorage& storage);
  static std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index(
      std::vector<uint32_t>& kv_cache_tokens_num);
};

}  // namespace xllm

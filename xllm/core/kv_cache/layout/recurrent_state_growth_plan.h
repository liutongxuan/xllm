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

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace xllm {

enum class RecurrentLiveSlot : int8_t { ABSENT, PRIVATE, SHARED };

// Execution facts after retaining the latest live or restore slot.
struct RecurrentStateProgress {
  RecurrentLiveSlot live_slot = RecurrentLiveSlot::ABSENT;
  size_t cached_tokens = 0;
  size_t published_checkpoint_index = 0;
  bool decode_step = false;
  bool graph_warmup = false;
};

// Logical demand for a writable recurrent slot and an optional checkpoint.
// Hash lookup, publication, slot ownership and allocation remain with the leaf.
class RecurrentStateGrowthPlan final {
 public:
  RecurrentStateGrowthPlan(size_t checkpoint_stride,
                           bool enable_prefix_cache,
                           const RecurrentStateProgress& progress)
      : live_slot_(progress.live_slot),
        retain_checkpoints_(enable_prefix_cache && !progress.decode_step) {
    CHECK_GT(checkpoint_stride, 0u);
    if (!retain_checkpoints_ || progress.graph_warmup ||
        live_slot_ == RecurrentLiveSlot::ABSENT) {
      return;
    }
    const size_t index = progress.cached_tokens / checkpoint_stride;
    if (progress.cached_tokens % checkpoint_stride == 0 &&
        index > progress.published_checkpoint_index) {
      checkpoint_index_ = index;
    }
  }

  std::optional<size_t> checkpoint_index() const { return checkpoint_index_; }

  // A cache hit removes checkpoint demand, but a shared prefill restore source
  // still needs a private output slot before it can be updated.
  size_t additional_slots(bool checkpoint_cached) const {
    if (live_slot_ == RecurrentLiveSlot::ABSENT) {
      return 1;
    }
    if (!retain_checkpoints_) {
      return 0;
    }
    return static_cast<size_t>(
        live_slot_ == RecurrentLiveSlot::SHARED ||
        (checkpoint_index_.has_value() && !checkpoint_cached));
  }

  // Preserving a checkpoint is optional when a private live slot can advance.
  // Initial allocation and a shared prefill restore source cannot be skipped.
  bool can_advance_with(size_t allocated_slots) const {
    return allocated_slots > 0 || live_slot_ == RecurrentLiveSlot::PRIVATE ||
           (live_slot_ == RecurrentLiveSlot::SHARED && !retain_checkpoints_);
  }

 private:
  RecurrentLiveSlot live_slot_;
  bool retain_checkpoints_;
  std::optional<size_t> checkpoint_index_;
};

}  // namespace xllm

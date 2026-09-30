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

#include "core/framework/batch/batch_sampling_plan.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "core/framework/batch/batch_input_data.h"
#include "core/framework/request/sequence.h"

namespace xllm {

BatchSamplingPlan BatchSamplingPlan::create(const BatchInputData& data) {
  return create(data.sequences, data.allowed_max_tokens);
}

BatchSamplingPlan BatchSamplingPlan::create(
    const std::vector<Sequence*>& sequences,
    const std::vector<uint32_t>& token_budgets) {
  CHECK_EQ(sequences.size(), token_budgets.size());
  BatchSamplingPlan plan;
  plan.windows_.reserve(sequences.size());
  plan.rows_.reserve(sequences.size());
  plan.sample_rows_.reserve(sequences.size());
  for (size_t index = 0; index < sequences.size(); ++index) {
    plan.append_sequence(sequences[index], token_budgets[index]);
  }
  return plan;
}

BatchSamplingPlan BatchSamplingPlan::create(Sequence* sequence,
                                            uint32_t token_budget) {
  BatchSamplingPlan plan;
  plan.windows_.reserve(1);
  plan.rows_.reserve(1);
  plan.sample_rows_.reserve(1);
  plan.append_sequence(sequence, token_budget);
  return plan;
}

void BatchSamplingPlan::append_row(BatchSamplingRow row) {
  CHECK_LT(rows_.size(),
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  if (row.sample) {
    sample_rows_.emplace_back(rows_.size());
  }
  rows_.emplace_back(row);
}

void BatchSamplingPlan::append_sequence(Sequence* sequence,
                                        uint32_t token_budget) {
  CHECK(sequence != nullptr);
  CHECK_LE(sequence->num_tokens(), std::numeric_limits<uint32_t>::max());
  const uint32_t token_count = static_cast<uint32_t>(sequence->num_tokens());
  const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
  CHECK_LE(cached_tokens, std::numeric_limits<uint32_t>::max());
  const uint32_t token_begin = static_cast<uint32_t>(cached_tokens);
  const uint32_t query_tokens =
      token_count > token_begin
          ? std::min(token_count - token_begin, token_budget)
          : 0;
  if (token_count > token_begin) {
    CHECK_GT(token_budget, 0);
  }
  const uint32_t token_end = token_begin + query_tokens;
  const size_t row_begin = rows_.size();
  const auto& slots = sequence->sample_slots();
  if (slots.empty()) {
    if (query_tokens > 0 && token_end == token_count) {
      append_row({sequence, token_end - 1});
    }
  } else {
    for (const SampleSlot& slot : slots) {
      CHECK_LE(slot.token_position, std::numeric_limits<uint32_t>::max());
      const uint32_t source =
          slot.token_position == 0
              ? 0
              : static_cast<uint32_t>(slot.token_position - 1);
      if (source < token_begin || source >= token_end) {
        continue;
      }
      // Positions 0 and 1 intentionally select the same source token twice.
      append_row({sequence, source, slot.sample_id, /*from_sample_slot=*/true});
    }
  }
  windows_.emplace_back(BatchSamplingWindow{
      sequence, token_begin, token_end, token_count, row_begin, rows_.size()});
}

}  // namespace xllm

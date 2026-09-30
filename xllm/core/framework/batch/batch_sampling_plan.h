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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {

class Sequence;
struct BatchInputData;

struct BatchSamplingWindow {
  Sequence* sequence;
  uint32_t token_begin;
  uint32_t token_end;
  uint32_t token_count;
  size_t row_begin;
  size_t row_end;
};

struct BatchSamplingRow {
  Sequence* sequence;
  uint32_t source_position = 0;
  size_t sample_id = 0;
  bool from_sample_slot = false;
  bool sample = true;
};

// Immutable host row mapping captured after sequence ordering and before KV
// advancement. It contains no tensors and stays alive through output writeback.
class BatchSamplingPlan final {
 public:
  BatchSamplingPlan(const BatchSamplingPlan&) = delete;
  BatchSamplingPlan& operator=(const BatchSamplingPlan&) = delete;
  BatchSamplingPlan(BatchSamplingPlan&&) noexcept = default;
  BatchSamplingPlan& operator=(BatchSamplingPlan&&) noexcept = default;

  static BatchSamplingPlan create(const BatchInputData& data);
  static BatchSamplingPlan create(const std::vector<Sequence*>& sequences,
                                  const std::vector<uint32_t>& token_budgets);
  static BatchSamplingPlan create(Sequence* sequence, uint32_t token_budget);

  size_t sequence_count() const { return windows_.size(); }
  const BatchSamplingWindow& window(size_t index) const {
    return windows_.at(index);
  }
  const std::vector<BatchSamplingRow>& rows() const { return rows_; }
  const std::vector<size_t>& sample_rows() const { return sample_rows_; }

 private:
  BatchSamplingPlan() = default;
  void append_sequence(Sequence* sequence, uint32_t token_budget);
  void append_row(BatchSamplingRow row);

  std::vector<BatchSamplingWindow> windows_;
  std::vector<BatchSamplingRow> rows_;
  // Output row -> selected row. Selected and sampled offsets remain distinct.
  std::vector<size_t> sample_rows_;
};

}  // namespace xllm

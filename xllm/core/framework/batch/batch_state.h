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

#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_input_data.h"
#include "core/framework/batch/batch_sampling_plan.h"
#include "core/framework/batch/batch_sequence_ordering.h"
#include "core/framework/batch/batch_storage.h"
#include "core/runtime/forward_params.h"
#include "core/util/threadpool.h"

namespace xllm {

struct ModelArgs;

// Ordinary LLM/VLM input coordination. Shared sequence resources are composed
// through BatchStorage; domain-specific builders stay outside that storage.
class BatchState final {
 public:
  void reserve(size_t sequence_count, size_t group_count) {
    storage_.reserve(sequence_count, group_count);
  }
  void add(Sequence* sequence, uint32_t token_budget) {
    storage_.add(sequence, token_budget);
  }
  void add(SequencesGroup* group) { storage_.add(group); }
  void set_batch_id() { storage_.set_batch_id(); }
  uint64_t batch_id() const { return storage_.batch_id(); }
  bool empty() const { return storage_.empty(); }
  BatchSequencePlan& sequence_plan() { return storage_.sequence_plan(); }
  const BatchSequencePlan& sequence_plan() const {
    return storage_.sequence_plan();
  }
  const std::vector<SequencesGroup*>& sequence_groups() const {
    return storage_.sequence_groups();
  }
  void set_swap_block_transfer_infos(std::vector<BlockTransferInfo> infos) {
    storage_.set_swap_block_transfer_infos(std::move(infos));
  }
  void update_forward_type(Sequence* sequence) {
    storage_.update_forward_type(sequence);
  }
  void refresh_forward_type(const std::vector<Sequence*>& sequences) {
    storage_.refresh_forward_type(sequences);
  }
  void refresh_sequences_from_groups() {
    storage_.refresh_sequences_from_groups();
  }
  std::vector<Sequence*> group_sequences() const {
    return storage_.group_sequences();
  }
  size_t num_group_sequences() const { return storage_.num_group_sequences(); }
  Sequence* group_sequence(size_t index) const {
    return storage_.group_sequence(index);
  }
  BatchInputData input_data(const BatchSequencePlan& plan) {
    return storage_.input_data(plan);
  }
  // Prepare the final sequence view before domain output handlers capture
  // their targets. Build methods may then advance KV state.
  BatchInputData prepare_sequence_input_data();
  BatchInputData prepare_distributed_input_data();
  ForwardInput build_sequence_input(
      const BatchInputData& data,
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size,
      const ModelArgs& args,
      int32_t cp_size,
      const BatchSamplingPlan* sampling_plan = nullptr);
  ForwardInput build_distributed_input(
      const BatchInputData& data,
      const ModelArgs& args,
      ThreadPool* thread_pool,
      int32_t cp_size,
      const BatchSamplingPlan* sampling_plan = nullptr);
  static std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index(
      std::vector<uint32_t>& kv_cache_tokens_num) {
    return BatchSequenceOrdering::cal_seq_exchange_index(kv_cache_tokens_num);
  }

 private:
  BatchStorage storage_;
};

}  // namespace xllm

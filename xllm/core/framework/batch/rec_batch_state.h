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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_storage.h"
#include "core/framework/batch/rec_batch_output_handler.h"
#include "core/framework/config/rec_execution_config.h"

namespace xllm {

struct ModelArgs;
class ThreadPool;
class MPMCThreadPool;

// Owns Rec sequence management, forward preparation and output processing.
// Shared sequence storage remains a private implementation detail.
class RecBatchState final {
 public:
  explicit RecBatchState(RecExecutionConfig config);
  explicit RecBatchState(BatchInputType input_type);

  BatchInputType input_type() const { return config_.input_type(); }
  const RecExecutionConfig& execution_config() const { return config_; }
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
  size_t num_scheduled_sequences() const {
    return storage_.sequence_plan().size();
  }
  size_t num_groups() const { return storage_.sequence_groups().size(); }
  void set_swap_block_transfer_infos(std::vector<BlockTransferInfo> infos) {
    storage_.set_swap_block_transfer_infos(std::move(infos));
  }
  const std::vector<SequencesGroup*>& sequence_groups() const {
    return storage_.sequence_groups();
  }
  const BatchSequencePlan& sequence_plan() const {
    return storage_.sequence_plan();
  }
  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return storage_.sequence_plan().budgets();
  }

  bool uses_group_input() const;
  size_t size() const;
  Sequence* sequence(size_t index) const;
  std::vector<Sequence*> get_sequences() const;
  void refresh_forward_type();
  void refresh_sequences_from_groups();

  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_batch_size,
                                     const ModelArgs& args,
                                     int32_t cp_size);
  ForwardInput prepare_forward_input(const ModelArgs& args,
                                     ThreadPool* thread_pool,
                                     int32_t cp_size);
  ForwardInput prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                         uint32_t min_decoding_batch_size,
                                         const ModelArgs& args,
                                         MPMCThreadPool* thread_pool);

  void process_sample_output(const RawForwardOutput& output,
                             bool replace_fake_token);
  void process_sample_output(const SampleOutput& output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size);
  void process_beam_search_output(const RawForwardOutput& output,
                                  bool replace_fake_token);
  void process_beam_sequence_group(const ForwardOutput& output);
  void finish();

 private:
  BatchStorage storage_;
  RecExecutionConfig config_;
  RecBatchOutputHandler output_handler_;
};

}  // namespace xllm

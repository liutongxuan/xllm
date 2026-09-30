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

#include "core/framework/batch/rec_batch_state.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/batch/batch_sequence_ordering.h"
#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/batch/rec_forward_input_builder.h"

namespace xllm {

RecBatchState::RecBatchState(RecExecutionConfig config)
    : config_(std::move(config)), output_handler_(config_) {
  CHECK(config_.valid()) << "Unsupported batch input type";
}

RecBatchState::RecBatchState(BatchInputType input_type)
    : RecBatchState(RecExecutionConfig(input_type)) {}

bool RecBatchState::uses_group_input() const {
  return config_.uses_group_input();
}

size_t RecBatchState::size() const {
  return uses_group_input() ? storage_.num_group_sequences()
                            : storage_.sequence_plan().size();
}

Sequence* RecBatchState::sequence(size_t index) const {
  return uses_group_input() ? storage_.group_sequence(index)
                            : storage_.sequence_plan()[index].sequence;
}

std::vector<Sequence*> RecBatchState::get_sequences() const {
  if (!uses_group_input() && !storage_.sequence_plan().empty()) {
    return storage_.sequence_plan().sequences();
  }
  return storage_.group_sequences();
}

void RecBatchState::refresh_forward_type() {
  storage_.refresh_forward_type(get_sequences());
}

void RecBatchState::refresh_sequences_from_groups() {
  if (!uses_group_input()) {
    storage_.refresh_sequences_from_groups();
  }
}

ForwardInput RecBatchState::prepare_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size) {
  if (config_.input_type() == BatchInputType::SEQUENCE) {
    CHECK(storage_.sequence_groups().empty() ||
          !storage_.sequence_plan().empty())
        << "Sequence input requires scheduled sequences";
    const auto data = storage_.input_data(storage_.sequence_plan());
    output_handler_.prepare(data);
    ForwardInputBuilder builder(data,
                                &args,
                                cp_size,
                                /*thread_pool=*/nullptr,
                                &output_handler_.sampling_plan());
    auto input = builder.build_forward_input(num_decoding_tokens,
                                             min_decoding_batch_size);
    storage_.set_linear_restore_src_blocks(
        builder.take_linear_restore_src_blocks());
    return input;
  }
  return prepare_rec_forward_input(num_decoding_tokens,
                                   min_decoding_batch_size,
                                   args,
                                   /*thread_pool=*/nullptr);
}

ForwardInput RecBatchState::prepare_forward_input(const ModelArgs& args,
                                                  ThreadPool* thread_pool,
                                                  int32_t cp_size) {
  CHECK(config_.input_type() == BatchInputType::SEQUENCE)
      << "Distributed input transport requires a sequence batch";
  CHECK(storage_.sequence_groups().empty() || !storage_.sequence_plan().empty())
      << "Sequence input requires scheduled sequences";
  BatchSequenceOrdering::prepare(storage_);
  const auto data = storage_.input_data(storage_.sequence_plan());
  output_handler_.prepare(data);
  ForwardInputBuilder builder(
      data, &args, cp_size, thread_pool, &output_handler_.sampling_plan());
  auto input = builder.build_forward_input(/*num_decoding_tokens=*/0,
                                           /*min_decoding_batch_size=*/0);
  storage_.set_linear_restore_src_blocks(
      builder.take_linear_restore_src_blocks());
  if (storage_.has_partial_finished_beam_group()) {
    input.sampling_params.acc_logprob = torch::Tensor();
  }
  return input;
}

ForwardInput RecBatchState::prepare_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    MPMCThreadPool* thread_pool) {
  CHECK(config_.input_type() != BatchInputType::SEQUENCE)
      << "Rec input requires an explicit Rec batch input type";
  output_handler_.clear();
  if (storage_.empty()) {
    return {};
  }
  BatchSequencePlan group_plan;
  const BatchSequencePlan* plan = &storage_.sequence_plan();
  if (uses_group_input()) {
    CHECK(!storage_.sequence_groups().empty())
        << "OneRec input requires request groups";
    group_plan.reserve(size());
    for (auto* sequence : get_sequences()) {
      group_plan.add(sequence, std::numeric_limits<uint32_t>::max());
    }
    plan = &group_plan;
  } else {
    CHECK(storage_.sequence_groups().empty() || !plan->empty())
        << "Sequence input requires scheduled sequences";
  }
  auto data = storage_.input_data(*plan);
  output_handler_.prepare(data);
  auto builder =
      RecForwardInputBuilder::create(config_, data, &args, thread_pool);
  return builder->build_rec_forward_input(num_decoding_tokens,
                                          min_decoding_batch_size);
}

void RecBatchState::process_sample_output(const RawForwardOutput& output,
                                          bool replace_fake_token) {
  const auto sequences = get_sequences();
  output_handler_.process_sample_output(
      {sequences, storage_.sequence_groups()}, output, replace_fake_token);
}

void RecBatchState::process_sample_output(
    const SampleOutput& output,
    bool replace_fake_token,
    bool force_requested_beam_result_size) {
  const auto sequences = get_sequences();
  output_handler_.process_sample_output({sequences, storage_.sequence_groups()},
                                        output,
                                        replace_fake_token,
                                        force_requested_beam_result_size);
}

void RecBatchState::process_beam_search_output(const RawForwardOutput& output,
                                               bool replace_fake_token) {
  const auto sequences = get_sequences();
  output_handler_.process_beam_search_output(
      {sequences, storage_.sequence_groups()}, output, replace_fake_token);
}

void RecBatchState::process_beam_sequence_group(const ForwardOutput& output) {
  const auto sequences = get_sequences();
  output_handler_.process_beam_sequence_group(
      {sequences, storage_.sequence_groups()}, output);
}

void RecBatchState::finish() {
  for (auto* group : storage_.sequence_groups()) {
    group->finish();
  }
  for (auto* sequence : get_sequences()) {
    sequence->finish();
  }
}

}  // namespace xllm

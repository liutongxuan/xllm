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

#include "core/framework/batch/batch_state.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/model/model_args.h"

namespace xllm {

BatchInputData BatchState::prepare_sequence_input_data() {
  CHECK(storage_.sequence_groups().empty() || !storage_.sequence_plan().empty())
      << "Sequence input requires scheduled sequences; group-only input "
         "requires a domain-specific input builder";
  return input_data(storage_.sequence_plan());
}

BatchInputData BatchState::prepare_distributed_input_data() {
  CHECK(storage_.sequence_groups().empty() || !storage_.sequence_plan().empty())
      << "Sequence input requires scheduled sequences";
  BatchSequenceOrdering::prepare(storage_);
  return input_data(storage_.sequence_plan());
}

ForwardInput BatchState::build_sequence_input(
    const BatchInputData& data,
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size,
    const BatchSamplingPlan* sampling_plan) {
  ForwardInputBuilder builder(data, &args, cp_size, nullptr, sampling_plan);
  auto input =
      builder.build_forward_input(num_decoding_tokens, min_decoding_batch_size);
  storage_.set_linear_restore_src_blocks(
      builder.take_linear_restore_src_blocks());
  return input;
}

ForwardInput BatchState::build_distributed_input(
    const BatchInputData& data,
    const ModelArgs& args,
    ThreadPool* thread_pool,
    int32_t cp_size,
    const BatchSamplingPlan* sampling_plan) {
  ForwardInputBuilder builder(data, &args, cp_size, thread_pool, sampling_plan);
  auto input = builder.build_forward_input(/*num_decoding_tokens=*/0,
                                           /*min_decoding_batch_size=*/0);
  storage_.set_linear_restore_src_blocks(
      builder.take_linear_restore_src_blocks());
  if (storage_.has_partial_finished_beam_group()) {
    input.sampling_params.acc_logprob = torch::Tensor();
  }
  return input;
}

}  // namespace xllm

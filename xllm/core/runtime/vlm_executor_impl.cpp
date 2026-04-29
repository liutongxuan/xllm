/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/request/mm_data_visitor.h"
#include "platform/device.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      args_(args),
      device_(device),
      options_(options) {
  if (FLAGS_enable_graph) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Device::type_str());
  }
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

MMDict VlmExecutorImpl::encode(const model_input::ModelInput& input) {
  return model_->encode(input);
}

MMDict VlmExecutorImpl::encode(model_input::ModelInput&& input) {
  return model_->encode(std::move(input));
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const model_input::ModelInput& input) {
  return run_with_typed_input(tokens, positions, kv_caches, input);
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 model_input::ModelInput&& input) {
  return run_with_typed_input(tokens, positions, kv_caches, std::move(input));
}

ModelOutput VlmExecutorImpl::run_with_typed_input(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    model_input::ModelInput input) {
  torch::NoGradGuard no_grad;
  CHECK(input.llm.has_value())
      << "VlmExecutorImpl::run requires llm partition in ModelInput";
  CHECK(input.vlm.has_value())
      << "VlmExecutorImpl::run requires vlm partition in ModelInput";
  MMBatchData& mm_data = input.vlm->mm_data;
  EncoderInputGatherVisitor input_gather;
  mm_data.foreach (input_gather);
  CHECK(input_gather.finish(mm_data));
  mm_data.to(device_);
  MMDict embedding = encode(input);
  EncoderOutputScatterVisitor scatter(embedding);
  mm_data.foreach (scatter);
  CHECK(scatter.finish());

  EncoderEmbeddingGatherVisitor gather(device_);
  mm_data.foreach (gather);
  CHECK(gather.finish(mm_data));

  input.llm->input_embedding = model_->get_input_embeddings(tokens, input);

  if (llm_executor_) {
    return llm_executor_->run(tokens, positions, kv_caches, std::move(input));
  }

  return model_->forward(tokens, positions, kv_caches, std::move(input));
}

}  // namespace xllm

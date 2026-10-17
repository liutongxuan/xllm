/* Copyright 2025-2026 The xLLM Authors.

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

#include "executor.h"

#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/runtime/py_executor_impl.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options)
    : beam_search_enabled_(
          BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
  const auto& model_config = ModelConfig::get_instance();
  std::string backend;
  if (ModelConfig::is_python_model_impl(model_config.model_impl())) {
    backend = "python";
  } else if (options.npu_kernel_backend() == "XLITE") {
    backend = "xlite";
  } else if (options.backend() != "vlm" && options.enable_graph()) {
    backend = Platform::type_str();
  } else {
    backend = options.backend();
  }
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}

bool Executor::supports_prepared_attention_metadata() const {
  return impl_->supports_prepared_attention_metadata();
}

void Executor::prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                          ModelInputParams& params) {
  impl_->prepare_attention_metadata(kv_caches, params);
}

void Executor::warmup_prepared_graph(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
  impl_->warmup_prepared_graph(tokens, positions, kv_caches, params);
}

ModelOutput Executor::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

SampleOutput Executor::sample(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) {
  return sampler_.forward(logits, params, filter_mask);
}

BeamSearchOutput Executor::beam_search(const SamplingParameters& params,
                                       const SampleOutput& sample_output) {
  if (!params.use_beam_search || !params.acc_logprob.defined() ||
      params.acc_logprob.numel() == 0) {
    return {};
  }
  CHECK(beam_search_enabled_) << "beam search kernel is not enabled";
  return beam_searcher_.forward(
      params.acc_logprob, sample_output.top_tokens, sample_output.top_logprobs);
}

void Executor::configure_rec_sampling(RecPipelineType pipeline_type) {
  CHECK(rec_sampler_ == nullptr) << "Rec sampling is already configured";
  rec_sampler_ = std::make_unique<RecSampler>(pipeline_type);
}

SampleOutput Executor::sample_rec(torch::Tensor& logits,
                                  const SamplingParameters& params,
                                  const torch::Tensor& filter_mask,
                                  const RecSamplingContext* context) {
  CHECK(rec_sampler_ != nullptr) << "Rec sampling is not configured";
  return rec_sampler_->forward(logits, params, filter_mask, context);
}

void Executor::initialize_rec_constraints(const RecConstraintTables& tables,
                                          const torch::Device& device) {
  CHECK(rec_sampler_ != nullptr) << "Rec sampling is not configured";
  rec_sampler_->initialize_device_constraints(tables, device);
}

bool Executor::can_use_rec_device_constraints(const SamplingParameters& params,
                                              int32_t current_step,
                                              int32_t beam_width) const {
  CHECK(rec_sampler_ != nullptr) << "Rec sampling is not configured";
  return rec_sampler_->can_use_device_constraints(
      params, current_step, beam_width);
}

void Executor::prepare_graph_input(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
  impl_->prepare_graph_input(tokens, positions, kv_caches, params);
}

bool Executor::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  return impl_->prepare_static_mtp_graph_tasks(signal, signal_stream);
}

PyExecutorImpl* Executor::python_impl() {
  return dynamic_cast<PyExecutorImpl*>(impl_.get());
}

}  // namespace xllm

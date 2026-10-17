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
#include "core/framework/parallel_state/parallel_args.h"
#include "core/runtime/py_executor_impl.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   const ParallelArgs* parallel_args)
    : model_(*model),
      beam_search_enabled_(
          BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
  if (parallel_args != nullptr) {
    const auto& mapping = parallel_args->mapping_data();
    if (mapping.contains("attnDp") && mapping["attnDp"].contains("rank")) {
      attention_dp_rank_ = mapping["attnDp"]["rank"].get<int64_t>();
      CHECK_GE(attention_dp_rank_, 0) << "invalid attnDp rank";
    }
  }
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

LogitsOutput Executor::compute_logits(const torch::Tensor& hidden_states,
                                      const torch::Tensor& selected_token_idxes,
                                      bool return_selected_hidden,
                                      const ParallelInput* parallel_input) {
  LogitsOutput output;
  output.selected_token_idxes =
      parallel_input != nullptr
          ? remap_selected_token_idxes(
                selected_token_idxes, *parallel_input, hidden_states)
          : selected_token_idxes;
  if (output.selected_token_idxes.defined() &&
      output.selected_token_idxes.device() != hidden_states.device()) {
    output.selected_token_idxes =
        output.selected_token_idxes
            .to(hidden_states.device(), /*non_blocking=*/false)
            .contiguous();
  }

  if (!return_selected_hidden) {
    output.logits = model_.logits(hidden_states, output.selected_token_idxes);
    return output;
  }

  output.logits = model_.logits(
      hidden_states, output.selected_token_idxes, output.selected_hidden);
  if (!output.selected_hidden.defined()) {
    if (output.selected_token_idxes.defined()) {
      // Equal row counts need not imply an identity selection. Gather in the
      // projection's resolved index order, retaining any converted indices.
      output.gather_idxes = output.selected_token_idxes.to(torch::kLong);
      output.selected_hidden =
          hidden_states.index_select(/*dim=*/0, output.gather_idxes);
    } else {
      output.selected_hidden = hidden_states;
    }
  }
  return output;
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

torch::Tensor Executor::remap_selected_token_idxes(
    const torch::Tensor& selected_token_idxes,
    const ParallelInput& parallel_input,
    const torch::Tensor& hidden_states) const {
  const int64_t hidden_num_rows = hidden_states.size(0);
  if (!selected_token_idxes.defined() || selected_token_idxes.numel() == 0 ||
      attention_dp_rank_ < 0 ||
      parallel_input.dp_global_token_nums.size() <= 1 || hidden_num_rows <= 0) {
    return selected_token_idxes;
  }

  CHECK_LT(attention_dp_rank_,
           static_cast<int64_t>(parallel_input.dp_global_token_nums.size()))
      << "attnDp rank exceeds dp_global_token_nums";
  const int64_t local_selected_rows = selected_token_idxes.numel();
  const int64_t local_token_count =
      parallel_input.dp_global_token_nums.at(attention_dp_rank_);
  if (hidden_num_rows == local_token_count ||
      hidden_num_rows == local_selected_rows) {
    return selected_token_idxes;
  }

  int64_t dp_offset = 0;
  for (int64_t i = 0; i < attention_dp_rank_; ++i) {
    dp_offset += parallel_input.dp_global_token_nums[i];
  }
  torch::Tensor selected_cpu =
      selected_token_idxes.to(torch::dtype(torch::kLong).device(torch::kCPU));
  torch::Tensor logical_selected_cpu = selected_cpu + dp_offset;
  const auto& padding_idx =
      parallel_input.dp_ep_padding_data.lm_head_skip_padding_token_indices();
  if (padding_idx.defined() && padding_idx.numel() > 0 &&
      hidden_num_rows > padding_idx.numel()) {
    torch::Tensor padding_cpu =
        padding_idx.to(torch::dtype(torch::kLong).device(torch::kCPU));
    const int64_t max_logical_selected =
        logical_selected_cpu.max().item<int64_t>();
    if (max_logical_selected < padding_cpu.numel()) {
      return padding_cpu.index_select(/*dim=*/0, logical_selected_cpu)
          .to(torch::dtype(selected_token_idxes.scalar_type())
                  .device(hidden_states.device()),
              /*non_blocking=*/false)
          .contiguous();
    }
  }

  if (dp_offset == 0) {
    return selected_token_idxes;
  }
  const int64_t max_selected_idx = selected_cpu.max().item<int64_t>();
  if (max_selected_idx + dp_offset >= hidden_num_rows) {
    return selected_token_idxes;
  }
  torch::Tensor remapped =
      selected_token_idxes.to(hidden_states.device(), /*non_blocking=*/false)
          .contiguous();
  return (remapped + dp_offset).to(remapped.scalar_type()).contiguous();
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

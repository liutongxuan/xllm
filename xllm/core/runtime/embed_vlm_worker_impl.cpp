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

#include "embed_vlm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "options.h"
#include "util/timer.h"

namespace xllm {
namespace {
std::vector<int32_t> get_q_seq_lens_vec(const ForwardInput& input) {
  model_input::ModelInput typed_input = input.get_typed_input();
  CHECK(typed_input.llm.has_value())
      << "EmbedVLMWorker requires the llm partition in ModelInput";
  return typed_input.llm->q_seq_lens_vec;
}
}  // namespace

EmbedVLMWorkerImpl::EmbedVLMWorkerImpl(const ParallelArgs& parallel_args,
                                       const torch::Device& device,
                                       const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool EmbedVLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  context.set_image_embedding_mode(false);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  return true;
}

std::optional<ForwardOutput> EmbedVLMWorkerImpl::step(
    const ForwardInput& input) {
  torch::DeviceGuard device_guard(device_);
  auto ret = device_.synchronize_default_stream();

  Timer timer;
  ForwardInput input_on_device = input.to(device_, dtype_);

  // TODO to adapt multi stream parallel later, just use [0] temporarily
  // all tensors should be on the same device as model
  auto flatten_tokens = input_on_device.token_ids;
  auto flatten_positions = input_on_device.positions;
  model_input::ModelInput typed_input = input_on_device.get_typed_input();
  auto sampling_params = input_on_device.sampling_params;

  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      flatten_tokens, flatten_positions, kv_caches_, std::move(typed_input));
  auto hidden_states = model_output.hidden_states;
  ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output
  ForwardOutput output;
  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined() &&
      input.sampling_params.is_embeddings) {
    auto embeddings =
        model_->pooler(hidden_states, sampling_params.selected_token_idxes);
    sample_output.embeddings = embeddings;
    // split full embeddings and add them to mm_embeddings
    // so that the user could receive embeddings of images and texts
    if (FLAGS_enable_return_mm_full_embeddings) {
      std::vector<int32_t> q_seq_len_vec = get_q_seq_lens_vec(input_on_device);
      sample_output.mm_embeddings.reserve(q_seq_len_vec.size());
      int32_t token_start_idx = 0;
      for (const int seq_len : q_seq_len_vec) {
        auto image_embed =
            embeddings.slice(0, token_start_idx, token_start_idx + seq_len);
        sample_output.mm_embeddings.emplace_back(image_embed);
        token_start_idx += seq_len;
      }
    }

    output.sample_output = sample_output;
    output.embedding = embeddings;
  }
  ret = device_.synchronize_default_stream();
  return output;
}

}  // namespace xllm

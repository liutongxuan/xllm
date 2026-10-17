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

#pragma once

#include <torch/types.h>

#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/rec_sampler.h"
#include "framework/sampling/sampler.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class PyExecutorImpl;
struct RecConstraintTables;

class Executor final {
 public:
  Executor(CausalLM* model,
           const ModelArgs& args,
           const torch::Device& device,
           const runtime::Options& options);

  virtual ~Executor() = default;

  bool supports_prepared_attention_metadata() const;
  // Called on Prepare after binding Slot-owned tensor and native metadata.
  // No model computation; the supported backend must not mutate active state.
  void prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                  ModelInputParams& params);

  // Regular warmup Tasks capture final Slot views before serving uses them.
  void warmup_prepared_graph(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& params);

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params);

  // Sampling preserves in-place logits transforms and raw logprob semantics.
  SampleOutput sample(torch::Tensor& logits,
                      const SamplingParameters& params,
                      const torch::Tensor& filter_mask = torch::Tensor());
  BeamSearchOutput beam_search(const SamplingParameters& params,
                               const SampleOutput& sample_output);

  void configure_rec_sampling(RecPipelineType pipeline_type);
  SampleOutput sample_rec(torch::Tensor& logits,
                          const SamplingParameters& params,
                          const torch::Tensor& filter_mask = torch::Tensor(),
                          const RecSamplingContext* context = nullptr);
  void initialize_rec_constraints(const RecConstraintTables& tables,
                                  const torch::Device& device);
  bool can_use_rec_device_constraints(const SamplingParameters& params,
                                      int32_t current_step,
                                      int32_t beam_width) const;

  void prepare_graph_input(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelInputParams& params);

  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const Stream& signal_stream);

  // Returns the Python executor when this instance is backed by
  // ``model_impl=python``. Native executors return nullptr. The accessor is
  // used by the MTP pair bridge to share one Python composite graph between
  // target and draft workers.
  PyExecutorImpl* python_impl();

 private:
  Sampler sampler_;
  BeamSearcher beam_searcher_;
  std::unique_ptr<RecSampler> rec_sampler_;
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace xllm

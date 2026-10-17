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

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/rec_sampler.h"
#include "framework/sampling/sampler.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"
#include "util/rec_model_utils.h"

namespace xllm {

struct RecConstraintTables;

class RecExecutor final {
 public:
  RecExecutor(CausalLM* model,
              const ModelArgs& args,
              const torch::Device& device,
              const runtime::Options& options,
              RecPipelineType pipeline_type);

  ~RecExecutor() = default;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params);

  SampleOutput sample(torch::Tensor& logits,
                      const SamplingParameters& params,
                      const torch::Tensor& filter_mask = torch::Tensor());
  BeamSearchOutput beam_search(const SamplingParameters& params,
                               const SampleOutput& sample_output);

  SampleOutput sample_rec(torch::Tensor& logits,
                          const SamplingParameters& params,
                          const torch::Tensor& filter_mask = torch::Tensor(),
                          const RecSamplingContext* context = nullptr);
  void initialize_rec_constraints(const RecConstraintTables& tables,
                                  const torch::Device& device);
  bool can_use_rec_device_constraints(const SamplingParameters& params,
                                      int32_t current_step,
                                      int32_t beam_width) const;

 private:
  Sampler sampler_;
  BeamSearcher beam_searcher_;
  std::unique_ptr<RecSampler> rec_sampler_;
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace xllm

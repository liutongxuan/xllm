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

#include <memory>

#include "core/framework/sampling/sampling_params.h"
#include "util/rec_model_utils.h"

namespace xllm {

struct RecSamplingContext {
  torch::Tensor sequence_group;
  int32_t current_step = -1;
  int32_t beam_width = 1;
  bool use_device_constraints = false;
};
class Sampler;
struct RecConstraintTables;

class RecSampler final {
 public:
  explicit RecSampler(RecPipelineType pipeline_type);
  ~RecSampler();

  void initialize_device_constraints(const RecConstraintTables& tables,
                                     const torch::Device& device);

  bool can_use_device_constraints(const SamplingParameters& params,
                                  int32_t current_step,
                                  int32_t beam_width) const;

  // logits: [batch_size, vocab_size]
  SampleOutput forward(torch::Tensor& logits,
                       const SamplingParameters& params,
                       const torch::Tensor& filter_mask = torch::Tensor(),
                       const RecSamplingContext* context = nullptr) const;

 private:
  struct RecConstraintDeviceTensors {
    torch::Tensor first_token_ids;
    torch::Tensor prefix1_offsets;
    torch::Tensor prefix1_values;
    torch::Tensor prefix1_pair_keys;
    torch::Tensor prefix2_value_offsets;
    torch::Tensor prefix2_values;
    int64_t max_prefix1_degree = 0;
    int64_t max_prefix2_degree = 0;
    bool initialized = false;
  };

  SampleOutput sample_with_device_constraints(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& sequence_group,
      int32_t current_step) const;

  class SamplingStrategy {
   public:
    virtual ~SamplingStrategy() = default;
    virtual SampleOutput forward(torch::Tensor& logits,
                                 const SamplingParameters& params,
                                 const torch::Tensor& filter_mask,
                                 const RecSamplingContext* context) const = 0;
  };

  class DefaultSamplingStrategy final : public SamplingStrategy {
   public:
    explicit DefaultSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  class OneRecConstrainedSamplingStrategy final : public SamplingStrategy {
   public:
    explicit OneRecConstrainedSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  class MultiRoundFastPathSamplingStrategy final : public SamplingStrategy {
   public:
    explicit MultiRoundFastPathSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  static std::unique_ptr<SamplingStrategy> create_sampling_strategy(
      RecPipelineType type,
      const Sampler& sampler);

  RecPipelineType pipeline_type_;
  RecConstraintDeviceTensors constraint_device_tensors_;
  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<SamplingStrategy> strategy_;
};

}  // namespace xllm

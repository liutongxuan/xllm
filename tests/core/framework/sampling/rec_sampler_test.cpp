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

#include "core/framework/sampling/rec_sampler.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>

namespace xllm {
namespace {

SamplingParameters make_rec_greedy_params() {
  SamplingParameters params;
  params.selected_token_idxes = torch::tensor({0}, torch::kLong);
  params.sample_idxes = torch::tensor({0}, torch::kLong);
  params.do_sample = torch::tensor({false}, torch::kBool);
  params.all_greedy_sample = true;
  params.all_random_sample = false;
  return params;
}

TEST(RecSamplerTest, OneRecAndDefaultPipelinesHonorExplicitConstraintMask) {
  constexpr std::array<RecPipelineType, 4> kPipelineTypes = {
      RecPipelineType::kLlmRecDefault,
      RecPipelineType::kLlmRecWithMmData,
      RecPipelineType::kOneRecDefault,
      RecPipelineType::kOneRecXAttentionPipeline};
  const auto params = make_rec_greedy_params();
  const torch::Tensor filter_mask =
      torch::tensor({{-INFINITY, -INFINITY, 0.0F}});

  for (const RecPipelineType pipeline_type : kPipelineTypes) {
    RecSampler sampler(pipeline_type);
    torch::Tensor logits = torch::tensor({{9.0F, 5.0F, 1.0F}});
    const auto result = sampler.forward(logits, params, filter_mask);
    EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 2)
        << "pipeline_type=" << static_cast<int32_t>(pipeline_type);
  }
}

TEST(RecSamplerTest, UninitializedDeviceConstraintsUseNormalSampling) {
  RecSampler sampler(RecPipelineType::kOneRecXAttentionPipeline);
  auto params = make_rec_greedy_params();
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{-INFINITY, 0.0F, 10.0F}});
  torch::Tensor logits = torch::tensor({{4.0F, 2.0F, 1.0F}});
  const torch::Tensor expected_logprobs = torch::log_softmax(logits, -1);
  RecSamplingContext context;
  context.current_step = 0;
  context.beam_width = 2;
  context.use_device_constraints = true;

  EXPECT_FALSE(sampler.can_use_device_constraints(
      params, context.current_step, context.beam_width));
  const auto result =
      sampler.forward(logits, params, torch::Tensor(), &context);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 2);
  EXPECT_NEAR(result.logprobs[0].item<float>(),
              expected_logprobs[0][2].item<float>(),
              1e-6);
  EXPECT_EQ(result.top_tokens[0][0].item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(logits[0][0]).item<bool>());
}

TEST(RecSamplerTest, ConstraintMaskTakesPrecedenceOverParameterMask) {
  RecSampler sampler(RecPipelineType::kOneRecDefault);
  auto params = make_rec_greedy_params();
  params.filter_mask = torch::tensor({{-INFINITY, 0.0F, -INFINITY}});
  const torch::Tensor filter_mask =
      torch::tensor({{-INFINITY, -INFINITY, 0.0F}});
  torch::Tensor logits = torch::tensor({{9.0F, 5.0F, 1.0F}});

  const auto result = sampler.forward(logits, params, filter_mask);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 2);
}

}  // namespace
}  // namespace xllm

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/runtime/executor.h"
#include "core/runtime/executor_impl_factory.h"

namespace xllm {
namespace {

class PostprocessModel final : public CausalLM {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& /*params*/) override {
    return ModelOutput(tokens.to(options_).view({-1, 1}));
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_token_idxes) override {
    return selected_token_idxes.defined()
               ? hidden_states.index_select(
                     /*dim=*/0, selected_token_idxes.to(torch::kLong))
               : hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}

 private:
  torch::TensorOptions options_ =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
};

class PostprocessExecutorImpl final : public ExecutorImpl {
 public:
  explicit PostprocessExecutorImpl(CausalLM* model) : model_(model) {}

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override {
    return model_->forward(tokens, positions, kv_caches, params);
  }

 private:
  CausalLM* model_;
};

SamplingParameters greedy_params(int64_t rows) {
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(rows, torch::kInt32);
  params.sample_idxes = torch::arange(rows, torch::kInt32);
  params.do_sample = torch::zeros({rows}, torch::kBool);
  return params;
}

class ExecutorPostprocessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const bool registered =
        ExecutorImplFactory::get_instance().register_creator(
            "executor_postprocess_test",
            [](CausalLM* model,
               const ModelArgs& /*args*/,
               const torch::Device& /*device*/,
               const runtime::Options& /*options*/) {
              return std::make_unique<PostprocessExecutorImpl>(model);
            });
    ASSERT_TRUE(registered);
    auto& config = ModelConfig::get_instance();
    previous_model_impl_ = config.model_impl();
    previous_reranker_ = config.enable_qwen3_reranker();
    config.model_impl("auto").enable_qwen3_reranker(false);
    options_.backend("executor_postprocess_test").enable_graph(false);
    executor_ = std::make_unique<Executor>(
        &model_, ModelArgs{}, torch::Device(torch::kCPU), options_);
  }

  void TearDown() override {
    executor_.reset();
    ModelConfig::get_instance()
        .model_impl(previous_model_impl_)
        .enable_qwen3_reranker(previous_reranker_);
  }

  PostprocessModel model_;
  runtime::Options options_;
  std::unique_ptr<Executor> executor_;
  std::string previous_model_impl_;
  bool previous_reranker_ = false;
};

TEST_F(ExecutorPostprocessTest,
       ForwardLeavesLogitsAndSamplingToExplicitStages) {
  const torch::Tensor tokens = torch::tensor({1, 2, 3}, torch::kInt32);
  const torch::Tensor positions = torch::arange(3, torch::kInt32);
  std::vector<KVCache> kv_caches;
  ModelInputParams params;
  const auto& generator =
      torch::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  const torch::Tensor before = generator.get_state().clone();

  const auto output = executor_->forward(tokens, positions, kv_caches, params);

  EXPECT_TRUE(torch::equal(output.hidden_states,
                           torch::tensor({{1.0F}, {2.0F}, {3.0F}})));
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
}

TEST_F(ExecutorPostprocessTest, SamplingMutatesLogitsAndReportsRawLogprobs) {
  SamplingParameters params = greedy_params(1);
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{0.0F, 0.0F, 10.0F}});
  torch::Tensor logits = torch::tensor({{4.0F, 2.0F, 1.0F}});
  const torch::Tensor alias = logits;
  const torch::Tensor raw_logprobs = torch::log_softmax(logits, -1);

  const auto output = executor_->sample(logits, params);

  EXPECT_TRUE(torch::equal(alias, torch::tensor({{4.0F, 2.0F, 11.0F}})));
  EXPECT_EQ(output.next_tokens[0].item<int64_t>(), 2);
  EXPECT_NEAR(
      output.logprobs[0].item<float>(), raw_logprobs[0][2].item<float>(), 1e-6);
  EXPECT_EQ(output.top_tokens[0][0].item<int64_t>(), 0);
  EXPECT_NEAR(output.top_logprobs[0][0].item<float>(),
              raw_logprobs[0][0].item<float>(),
              1e-6);
}

TEST_F(ExecutorPostprocessTest, SamplingResultsRemainIndependentAcrossCalls) {
  const SamplingParameters params = greedy_params(1);
  torch::Tensor first_logits = torch::tensor({{10.0F, 0.0F, 1.0F}});
  torch::Tensor second_logits = torch::tensor({{0.0F, 10.0F, 1.0F}});

  const auto first = executor_->sample(first_logits, params);
  const auto second = executor_->sample(second_logits, params);

  EXPECT_EQ(first.next_tokens[0].item<int64_t>(), 0);
  EXPECT_EQ(second.next_tokens[0].item<int64_t>(), 1);
  EXPECT_NE(first.next_tokens.data_ptr(), second.next_tokens.data_ptr());
}

}  // namespace
}  // namespace xllm

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

#include "core/runtime/rec_graph_input.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

TEST(RecGraphInputTest, ReusesBindingsAfterInPlaceRoundUpdate) {
  RecModelParams params;
  auto& strategy = params.mutable_llmrec_params();
  strategy.batch_size = 1;
  strategy.beam_width = 2;
  strategy.total_round = 3;
  strategy.current_round_tensor = torch::zeros({1}, torch::kInt32);
  strategy.unshared_k_caches = {torch::ones({2, 4})};
  strategy.unshared_v_caches = {torch::zeros({2, 4})};
  RecModelParams captured = params.clone();

  strategy.current_round_tensor.fill_(1);
  strategy.unshared_k_caches.front().fill_(7);

  EXPECT_TRUE(rec_graph_bindings_match(captured, params));
  EXPECT_EQ(captured.llmrec_params()->current_round_tensor.item<int32_t>(), 1);
}

TEST(RecGraphInputTest, RejectsEqualShapeKvWithNewStorage) {
  RecModelParams params;
  auto& strategy = params.mutable_onerec_xattention_params();
  strategy.unshared_k_caches = {torch::ones({2, 4})};
  strategy.unshared_v_caches = {torch::zeros({2, 4})};
  RecModelParams captured = params.clone();
  ASSERT_TRUE(rec_graph_bindings_match(captured, params));

  strategy.unshared_k_caches.front() =
      strategy.unshared_k_caches.front().clone();

  EXPECT_FALSE(rec_graph_bindings_match(captured, params));
  EXPECT_TRUE(torch::equal(
      captured.onerec_xattention_params()->unshared_k_caches.front(),
      strategy.unshared_k_caches.front()));
}

TEST(RecGraphInputTest, SnapshotRetainsStrategyAfterSourceDestruction) {
  RecModelParams captured = [] {
    RecModelParams source;
    auto& strategy = source.mutable_onerec_xattention_params();
    strategy.current_round_tensor = torch::tensor({2}, torch::kInt32);
    strategy.unshared_k_caches = {torch::full({2, 4}, 5.0)};
    return source.clone();
  }();

  const auto* strategy = captured.onerec_xattention_params();
  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->current_round_tensor.item<int32_t>(), 2);
  ASSERT_EQ(strategy->unshared_k_caches.size(), 1);
  EXPECT_TRUE(torch::equal(strategy->unshared_k_caches.front(),
                           torch::full({2, 4}, 5.0)));
  RecModelParams next = captured.clone();
  EXPECT_TRUE(rec_graph_bindings_match(captured, next));
}

TEST(RecGraphInputTest, RejectsChangedStaticStrategyOrVariant) {
  RecModelParams params;
  params.mutable_llmrec_params().beam_width = 2;
  RecModelParams captured = params.clone();

  params.mutable_llmrec_params().beam_width = 3;
  EXPECT_FALSE(rec_graph_bindings_match(captured, params));

  params.mutable_onerec_xattention_params();
  EXPECT_FALSE(rec_graph_bindings_match(captured, params));
}

}  // namespace
}  // namespace xllm

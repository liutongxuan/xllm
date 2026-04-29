/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "model_input.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "causal_lm.h"
#include "causal_vlm.h"
#include "dit_model.h"
#include "model_traits.h"
#include "rec_causal_lm.h"

namespace xllm {
namespace model_input {
namespace {

struct TypedForwardModel {
  bool typed_const_called = false;
  bool typed_rvalue_called = false;
  bool legacy_called = false;

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInput&) {
    typed_const_called = true;
    return ModelOutput();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      ModelInput&&) {
    typed_rvalue_called = true;
    return ModelOutput();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) {
    legacy_called = true;
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) {}
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}
  void update_expert_weight(int32_t) {}
};

struct LegacyOnlyForwardModel {
  bool legacy_called = false;

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) {
    legacy_called = true;
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) {}
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}
  void update_expert_weight(int32_t) {}
};

struct TypedForwardHolder {
  TypedForwardModel* impl = nullptr;
  TypedForwardModel* operator->() const { return impl; }
};

struct LegacyOnlyForwardHolder {
  LegacyOnlyForwardModel* impl = nullptr;
  LegacyOnlyForwardModel* operator->() const { return impl; }
};

static_assert(detail::has_typed_forward<TypedForwardHolder>::value,
              "typed forward holder must satisfy has_typed_forward");
static_assert(detail::has_typed_forward_rvalue<TypedForwardHolder>::value,
              "typed forward holder must satisfy has_typed_forward_rvalue");
static_assert(!detail::has_typed_forward<LegacyOnlyForwardHolder>::value,
              "legacy-only holder must not satisfy has_typed_forward");
static_assert(!detail::has_typed_forward_rvalue<LegacyOnlyForwardHolder>::value,
              "legacy-only holder must not satisfy has_typed_forward_rvalue");

class FakeCausalLM final : public CausalLM {
 public:
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInput&) override {
    return ModelOutput();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      ModelInput&&) override {
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

class FakeCausalVLM final : public CausalVLM {
 public:
  MMDict encode(const ModelInputParams&) override { return MMDict(); }
  torch::Tensor get_input_embeddings(const torch::Tensor&,
                                     const ModelInputParams&) override {
    return torch::Tensor();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInput&) override {
    return ModelOutput();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      ModelInput&&) override {
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

class FakeRecCausalLM final : public RecCausalLM {
 public:
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInput&) override {
    return ModelOutput();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      ModelInput&&) override {
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

class FakeDiTModel final : public DiTModel {
 public:
  DiTForwardOutput forward(const DiTForwardInput&) override {
    return DiTForwardOutput();
  }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  const torch::TensorOptions& options() const override { return options_; }
  void load_model(std::unique_ptr<DiTModelLoader>) override {}

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

TEST(ModelInputTest, CausalLmCreatesLlmInputDirectly) {
  FakeCausalLM model;
  ModelInputParams params;
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputTest, CausalLmCreatesLlmInputDirectlyFromMovedParams) {
  FakeCausalLM model;
  ModelInputParams params;
  const ModelInput input = model.create_model_input(std::move(params));
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputTest, RecCausalLmCreatesRecInputDirectlyWhenPresent) {
  FakeRecCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_rec(input));
}

TEST(ModelInputTest,
     RecCausalLmCreatesRecInputDirectlyFromMovedParamsWhenPresent) {
  FakeRecCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = model.create_model_input(std::move(params));
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_rec(input));
}

TEST(ModelInputTest, CausalVlmCreatesVlmInputDirectly) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputTest, CausalVlmCreatesVlmInputDirectlyFromMovedParams) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  const ModelInput input = model.create_model_input(std::move(params));
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputTest, DitModelCreatesDitInputDirectly) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");
  const ModelInput input = model.create_model_input(params);
  EXPECT_FALSE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_TRUE(has_dit(input));
}

TEST(ModelInputTest, DitModelCreatesDitInputDirectlyFromMovedParams) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");
  const ModelInput input = model.create_model_input(std::move(params));
  EXPECT_FALSE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_TRUE(has_dit(input));
}

TEST(ModelInputTest, MakeModelInputFromLegacyMoveKeepsPartitions) {
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = make_model_input_from_legacy(std::move(params));
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_TRUE(has_rec(input));
}

TEST(ModelInputTest, ApplyToLegacyMoveKeepsPartitions) {
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  ModelInput input = make_model_input_from_legacy(std::move(params));
  ModelInputParams restored;
  apply_model_input_to_legacy(std::move(input), &restored);

  EXPECT_TRUE(restored.deep_stacks.size() == 1);
  EXPECT_FALSE(std::holds_alternative<std::monostate>(restored.rec_params));
}

TEST(ModelInputTest, CausalLmImplDispatchesTypedForwardWhenSupported) {
  TypedForwardModel impl;
  CausalLMImpl<TypedForwardHolder> wrapper(
      TypedForwardHolder{&impl}, torch::TensorOptions().device(torch::kCPU));

  std::vector<KVCache> kv_caches;
  torch::Tensor tokens;
  torch::Tensor positions;
  ModelInput typed_input;
  typed_input.llm = LLMModelInputParams{};

  wrapper.forward(tokens, positions, kv_caches, typed_input);
  EXPECT_TRUE(impl.typed_const_called);
  EXPECT_FALSE(impl.legacy_called);

  impl.typed_const_called = false;
  ModelInput rvalue_input;
  rvalue_input.llm = LLMModelInputParams{};
  wrapper.forward(tokens, positions, kv_caches, std::move(rvalue_input));
  EXPECT_TRUE(impl.typed_rvalue_called);
  EXPECT_FALSE(impl.typed_const_called);
  EXPECT_FALSE(impl.legacy_called);
}

TEST(ModelInputTest, CausalLmImplFallsBackToLegacyWhenNoTypedForward) {
  LegacyOnlyForwardModel impl;
  CausalLMImpl<LegacyOnlyForwardHolder> wrapper(
      LegacyOnlyForwardHolder{&impl},
      torch::TensorOptions().device(torch::kCPU));

  std::vector<KVCache> kv_caches;
  torch::Tensor tokens;
  torch::Tensor positions;
  ModelInput typed_input;
  typed_input.llm = LLMModelInputParams{};

  wrapper.forward(tokens, positions, kv_caches, typed_input);
  EXPECT_TRUE(impl.legacy_called);
}

}  // namespace
}  // namespace model_input
}  // namespace xllm

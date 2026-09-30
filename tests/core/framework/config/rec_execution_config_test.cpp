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

#include "core/framework/config/rec_execution_config.h"

#include <gtest/gtest.h>

#include "core/framework/config/rec_execution_config_binding.h"

namespace xllm {
namespace {

TEST(RecExecutionConfigTest, ResolvesFourExecutionModes) {
  struct ExecutionCase {
    const char* model_type;
    int32_t rounds;
    RecPipelineType pipeline;
    BatchInputType input;
    bool requires_kv;
  };
  const ExecutionCase cases[] = {
      {"qwen3",
       0,
       RecPipelineType::kLlmRecDefault,
       BatchInputType::SEQUENCE,
       true},
      {"qwen2",
       4,
       RecPipelineType::kLlmRecMultiRoundPipeline,
       BatchInputType::REC_MULTI_ROUND,
       false},
      {"onerec",
       0,
       RecPipelineType::kOneRecDefault,
       BatchInputType::ONEREC,
       false},
      {"onerec",
       3,
       RecPipelineType::kOneRecXAttentionPipeline,
       BatchInputType::ONEREC_XATTENTION,
       true},
  };
  for (const auto& execution : cases) {
    const auto config = RecExecutionConfig::resolve(
        execution.model_type, execution.rounds, /*enable_prefill_only=*/false);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->pipeline_type(), execution.pipeline);
    EXPECT_EQ(config->input_type(), execution.input);
    EXPECT_EQ(config->decode_rounds(), execution.rounds);
    EXPECT_EQ(config->requires_kv_cache(), execution.requires_kv);
    EXPECT_EQ(config->uses_group_input(),
              config->rec_type() == RecType::kOneRec);
  }
}

TEST(RecExecutionConfigTest,
     LegacyPrefillContractOnlyAppliesToSingleRoundOneRec) {
  const auto legacy = RecExecutionConfig::resolve(
      "onerec", /*decode_rounds=*/0, /*enable_prefill_only=*/true);
  const auto xattention = RecExecutionConfig::resolve(
      "onerec", /*decode_rounds=*/2, /*enable_prefill_only=*/true);
  const auto llmrec = RecExecutionConfig::resolve(
      "qwen3", /*decode_rounds=*/0, /*enable_prefill_only=*/true);
  ASSERT_TRUE(legacy.has_value());
  ASSERT_TRUE(xattention.has_value());
  ASSERT_TRUE(llmrec.has_value());
  EXPECT_TRUE(legacy->use_legacy_onerec_prefill_only_contract());
  EXPECT_FALSE(xattention->use_legacy_onerec_prefill_only_contract());
  EXPECT_FALSE(llmrec->use_legacy_onerec_prefill_only_contract());
  EXPECT_EQ(legacy->decode_rounds(), 0);
}

TEST(RecExecutionConfigTest, ValidatesTopologyBeforeExecution) {
  const auto llmrec = RecExecutionConfig::resolve(
      "qwen3", /*decode_rounds=*/0, /*enable_prefill_only=*/false);
  ASSERT_TRUE(llmrec.has_value());
  EXPECT_FALSE(
      llmrec->validate_topology(/*dp_size=*/2, /*nnodes=*/4).has_value());
  EXPECT_TRUE(
      llmrec->validate_topology(/*dp_size=*/0, /*nnodes=*/1).has_value());
  EXPECT_TRUE(
      llmrec->validate_topology(/*dp_size=*/1, /*nnodes=*/0).has_value());
  const RecExecutionConfig local_modes[] = {
      RecExecutionConfig(BatchInputType::ONEREC),
      RecExecutionConfig(BatchInputType::ONEREC_XATTENTION),
      RecExecutionConfig(BatchInputType::REC_MULTI_ROUND),
  };
  for (const auto& config : local_modes) {
    EXPECT_FALSE(
        config.validate_topology(/*dp_size=*/1, /*nnodes=*/1).has_value());
    EXPECT_TRUE(
        config.validate_topology(/*dp_size=*/2, /*nnodes=*/1).has_value());
    EXPECT_TRUE(
        config.validate_topology(/*dp_size=*/1, /*nnodes=*/2).has_value());
  }
}

TEST(RecExecutionConfigTest, RejectsUnsupportedModelsRoundsAndInputContracts) {
  EXPECT_FALSE(RecExecutionConfig::resolve("unknown",
                                           /*decode_rounds=*/0,
                                           /*enable_prefill_only=*/false)
                   .has_value());
  EXPECT_FALSE(RecExecutionConfig::resolve(RecModelKind::kNone,
                                           /*decode_rounds=*/0,
                                           /*enable_prefill_only=*/false)
                   .has_value());
  EXPECT_FALSE(RecExecutionConfig::resolve("onerec",
                                           /*decode_rounds=*/-1,
                                           /*enable_prefill_only=*/false)
                   .has_value());
  const RecExecutionConfig invalid(static_cast<BatchInputType>(-1));
  EXPECT_FALSE(invalid.valid());
  EXPECT_EQ(invalid.rec_type(), RecType::kNone);
  EXPECT_TRUE(
      invalid.validate_topology(/*dp_size=*/1, /*nnodes=*/1).has_value());
}

TEST(RecExecutionConfigTest, InputOnlyDefaultsAreDeterministic) {
  const RecExecutionConfig llmrec(BatchInputType::SEQUENCE);
  const RecExecutionConfig onerec(BatchInputType::ONEREC);
  const RecExecutionConfig multi_round(BatchInputType::REC_MULTI_ROUND);
  const RecExecutionConfig xattention(BatchInputType::ONEREC_XATTENTION);
  EXPECT_EQ(llmrec.decode_rounds(), 0);
  EXPECT_EQ(onerec.decode_rounds(), 0);
  EXPECT_EQ(multi_round.decode_rounds(), 1);
  EXPECT_EQ(xattention.decode_rounds(), 1);
  EXPECT_FALSE(onerec.use_legacy_onerec_prefill_only_contract());
}

TEST(RecExecutionConfigTest,
     LegacyBindingRejectsConflictsWithoutChangingSnapshot) {
  RecExecutionConfigBinding binding;
  const auto original = RecExecutionConfig::resolve(
      "onerec", /*decode_rounds=*/0, /*enable_prefill_only=*/true);
  const auto incompatible = RecExecutionConfig::resolve(
      "onerec", /*decode_rounds=*/3, /*enable_prefill_only=*/false);
  ASSERT_TRUE(original.has_value());
  ASSERT_TRUE(incompatible.has_value());
  EXPECT_EQ(binding.bind(original.value()),
            RecExecutionBindingResult::INITIALIZED);
  EXPECT_EQ(binding.bind(original.value()),
            RecExecutionBindingResult::UNCHANGED);
  EXPECT_EQ(binding.bind(incompatible.value()),
            RecExecutionBindingResult::CONFLICT);
  EXPECT_EQ(binding.config(), original);
  EXPECT_EQ(binding.bind(RecExecutionConfig(static_cast<BatchInputType>(-1))),
            RecExecutionBindingResult::CONFLICT);
  EXPECT_EQ(binding.config(), original);
}

}  // namespace
}  // namespace xllm

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

#include "models/model_registry.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "core/common/extension_registry.h"
#include "core/common/options.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

CausalLMFactory unused_lm_factory() {
  return [](const ModelContext& /*context*/) -> std::unique_ptr<CausalLM> {
    return nullptr;
  };
}

CausalVLMFactory unused_vlm_factory() {
  return [](const ModelContext& /*context*/) -> std::unique_ptr<CausalVLM> {
    return nullptr;
  };
}

TEST(ModelRegistryTest, DeepseekV4DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v4_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v4_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, RegisteredAtbModelRejectsTorchBackend) {
  ASSERT_TRUE(ModelRegistry::get_causallm_factory("llama"));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_FALSE(resolve_model_registration(
      "llama", "TORCH", &effective_backend, &resolved_name, &error_message));
  EXPECT_EQ(error_message,
            "Model type llama only supports --npu_kernel_backend=ATB.");
}

TEST(ModelRegistryTest, ExtensionModelRetainsDefaultAtbRouting) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("extension_model",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "extension_model");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("extension_model",
                                         "ATB",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "extension_model");
  EXPECT_TRUE(error_message.empty());
}

TEST(ModelRegistryTest, GlmMoeDsaMtpSupportsAtbAndTorchBackends) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "ATB",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");
}

TEST(ModelRegistryTest, NpuResolutionUsesEachInstanceExtensionSelection) {
  std::string directory_pattern =
      (std::filesystem::temp_directory_path() / "xllm_extension_options_XXXXXX")
          .string();
  char* const directory = ::mkdtemp(directory_pattern.data());
  ASSERT_NE(directory, nullptr);
  const std::filesystem::path model_path(directory);
  ScopeGuard remove_directory(
      [&model_path] { std::filesystem::remove_all(model_path); });
  {
    std::ofstream config(model_path / "config.json");
    config << R"({"model_type":"qwen2"})";
    ASSERT_TRUE(config.good());
  }

  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  KernelConfig& kernel_config = KernelConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  const std::string previous_extension = execution_config.extension_impl();
  const std::string previous_backend = kernel_config.npu_kernel_backend();
  const std::string previous_model_impl = model_config.model_impl();
  ScopeGuard restore_config([&] {
    execution_config.extension_impl(previous_extension);
    kernel_config.npu_kernel_backend(previous_backend);
    model_config.model_impl(previous_model_impl);
  });
  execution_config.extension_impl("unrelated_process_extension");
  model_config.model_impl("");

  const std::string family = "npu_public_options_custom";
  ASSERT_TRUE(ModelRegistry::register_causallm_factory(
      "qwen2", family, unused_lm_factory()));

  Options selected_options;
  selected_options.model_path(model_path.string())
      .backend("llm")
      .extension_impl(family);
  Options builtin_options;
  builtin_options.model_path(model_path.string()).backend("llm");

  resolve_npu_kernel_backend(&selected_options,
                             /*model_component=*/"causal_lm");
  resolve_npu_kernel_backend(&builtin_options, /*model_component=*/"causal_lm");

  EXPECT_EQ(selected_options.npu_kernel_backend(), "TORCH");
  EXPECT_EQ(builtin_options.npu_kernel_backend(), "ATB");
  EXPECT_EQ(selected_options.extension_impl(), family);
  EXPECT_TRUE(builtin_options.extension_impl().empty());
  EXPECT_EQ(execution_config.extension_impl(), "unrelated_process_extension");
}

TEST(ModelRegistryTest, SelectedLlmExtensionResolvesBeforeBuiltinBackendRules) {
  const std::string family = "npu_backend_lm_custom";
  ASSERT_TRUE(ModelRegistry::register_causallm_factory(
      "qwen2", family, unused_lm_factory()));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("qwen2",
                                         "AUTO",
                                         family,
                                         "causal_lm",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "qwen2");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("qwen2",
                                         "TORCH",
                                         family,
                                         "causal_lm",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "qwen2");

  EXPECT_TRUE(resolve_model_registration("qwen2",
                                         "ATB",
                                         family,
                                         "causal_lm",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "qwen2");
}

TEST(ModelRegistryTest, SelectedVlmExtensionResolvesBeforeBuiltinBackendRules) {
  const std::string family = "npu_backend_vlm_custom";
  ASSERT_TRUE(ModelRegistry::register_causalvlm_factory(
      "qwen2_5_vl", family, unused_vlm_factory()));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("qwen2_5_vl",
                                         "AUTO",
                                         family,
                                         "causal_vlm",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "qwen2_5_vl");

  EXPECT_TRUE(resolve_model_registration("qwen2_5_vl",
                                         "TORCH",
                                         family,
                                         "causal_vlm",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "qwen2_5_vl");
}

TEST(ModelRegistryTest, ExtensionCanReplaceBuiltinTorchOnlyModelWithAtb) {
  const std::string family = "npu_backend_explicit_custom";
  ASSERT_TRUE(ModelRegistry::register_causallm_factory(
      "qwen3_next", family, unused_lm_factory()));
  std::string effective_backend;
  std::string resolved_name;

  EXPECT_TRUE(resolve_model_registration("qwen3_next",
                                         "ATB",
                                         family,
                                         "causal_lm",
                                         &effective_backend,
                                         &resolved_name));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "qwen3_next");
}

TEST(ModelRegistryTest, ExecutorOnlyFamilyRetainsBuiltinBackendRules) {
  const std::string family = "npu_backend_executor_custom";
  ExtensionRegistry::get_instance().register_implementation(family, "executor");
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("qwen2",
                                         "AUTO",
                                         family,
                                         "causal_lm",
                                         &effective_backend,
                                         &resolved_name));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "qwen2_atb");

  EXPECT_FALSE(resolve_model_registration("qwen2",
                                          "TORCH",
                                          family,
                                          "causal_lm",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type qwen2 only supports --npu_kernel_backend=ATB.");
}

TEST(ModelRegistryTest, ExtensionBackendSelectionIsSpecificToModelComponent) {
  const std::string family = "npu_backend_other_component_custom";
  ASSERT_TRUE(ModelRegistry::register_causalvlm_factory(
      "qwen2", family, unused_vlm_factory()));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_FALSE(resolve_model_registration("qwen2",
                                          "TORCH",
                                          family,
                                          "causal_lm",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type qwen2 only supports --npu_kernel_backend=ATB.");
}

TEST(ModelRegistryTest, SelectedModelFamilyRequiresRequestedType) {
  const std::string family = "npu_backend_missing_model_custom";
  ASSERT_TRUE(ModelRegistry::register_causallm_factory(
      "npu_backend_other_model", family, unused_lm_factory()));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_FALSE(resolve_model_registration("qwen2",
                                          "AUTO",
                                          family,
                                          "causal_lm",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Missing causal_lm extension for model type: qwen2 with "
            "extension_impl=" +
                family);
}

TEST(ModelRegistryTest, SelectedExtensionStillValidatesBackendAvailability) {
  const std::string family = "npu_backend_validation_custom";
  ASSERT_TRUE(ModelRegistry::register_causallm_factory(
      "qwen2", family, unused_lm_factory()));
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_FALSE(resolve_model_registration("qwen2",
                                          "INVALID",
                                          family,
                                          "causal_lm",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Unsupported --npu_kernel_backend=INVALID. Supported values: "
            "AUTO, ATB, TORCH, XLITE.");

  const bool xlite_available = resolve_model_registration("qwen2",
                                                          "XLITE",
                                                          family,
                                                          "causal_lm",
                                                          &effective_backend,
                                                          &resolved_name,
                                                          &error_message);
#if defined(USE_XLITE)
  EXPECT_TRUE(xlite_available);
  EXPECT_EQ(effective_backend, "XLITE");
  EXPECT_EQ(resolved_name, "qwen2");
#else
  EXPECT_FALSE(xlite_available);
  EXPECT_EQ(error_message,
            "XLITE backend is not compiled; please build with USE_XLITE=ON.");
#endif
}

TEST(ModelRegistryTest, UnknownExtensionFamilyCannotBypassBackendRules) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_FALSE(resolve_model_registration("qwen2",
                                          "TORCH",
                                          "npu_backend_unknown_custom",
                                          "causal_lm",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message, "Unknown extension_impl=npu_backend_unknown_custom");
}

#if defined(USE_NPU) || defined(USE_MLU)
TEST(ModelRegistryTest, ResolvesDescriptorForCanonicalAlias) {
  ResolvedModelDescriptor descriptor;
  std::string error_message;

  ASSERT_TRUE(ModelRegistry::resolve_model_descriptor(
      "glm5_3_flash", "", &descriptor, &error_message))
      << error_message;
  EXPECT_EQ(descriptor.requested_name, "glm5_3_flash");
  EXPECT_EQ(descriptor.canonical_name, "glm5_next");
  EXPECT_EQ(descriptor.registration_name, "glm5_next");
  EXPECT_EQ(descriptor.family, ModelFamily::LLM);
  EXPECT_EQ(descriptor.family_name, "llm");
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.model_args_loader));
}
#endif

TEST(ModelRegistryTest, ExposesPartialRegistrationWithoutInferringFamily) {
  const std::string name = "descriptor_partial_registration_test";
  ModelRegistry::register_model_args_loader(
      name,
      [](const JsonReader& /*json*/, ModelArgs* /*args*/) { return true; });

  ResolvedModelDescriptor descriptor;
  std::string error_message;
  ASSERT_TRUE(ModelRegistry::resolve_model_descriptor(
      name, "", &descriptor, &error_message))
      << error_message;
  EXPECT_EQ(descriptor.registration_name, name);
  EXPECT_EQ(descriptor.family, ModelFamily::UNKNOWN);
  EXPECT_TRUE(descriptor.family_name.empty());
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.model_args_loader));
  EXPECT_TRUE(descriptor.meta.causal_lm_factories.empty());
}

TEST(ModelRegistryTest, SnapshotsAllRegisteredModelMetadata) {
  const std::string name = "descriptor_complete_registration_test";
  ModelRegistry::register_causallm_factory(name,
                                           [](const ModelContext& /*context*/) {
                                             return std::unique_ptr<CausalLM>();
                                           });
  ModelRegistry::register_rec_model_factory(
      name, [](const ModelContext& /*context*/) {
        return std::unique_ptr<RecModel>();
      });
  ModelRegistry::register_model_args_loader(
      name,
      [](const JsonReader& /*json*/, ModelArgs* /*args*/) { return true; });
  ModelRegistry::register_quant_args_loader(
      name,
      [](const JsonReader& /*json*/, QuantArgs* /*args*/) { return true; });
  ModelRegistry::register_tokenizer_args_loader(
      name,
      [](const JsonReader& /*json*/, TokenizerArgs* /*args*/) { return true; });
  ModelRegistry::register_multimodal_processor_factory(
      name,
      [](const ModelArgs& /*model_args*/,
         std::shared_ptr<Tokenizer> /*tokenizer*/,
         const TokenizerArgs& /*tokenizer_args*/) {
        return std::unique_ptr<MultimodalProcessorBase>();
      });
  ModelRegistry::register_cp_sharding_mode(name, CpShardingMode::MODEL);
  ModelRegistry::register_dcp_capability(name);
  ModelRegistry::register_mtp_shared_weight_owner(name);

  ResolvedModelDescriptor descriptor;
  ASSERT_TRUE(
      ModelRegistry::resolve_model_descriptor(name, "", &descriptor, nullptr));
  EXPECT_EQ(descriptor.family, ModelFamily::LLM);
  EXPECT_EQ(descriptor.family_name, "llm");
  EXPECT_TRUE(descriptor.meta.causal_lm_factories.contains("native"));
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.rec_model_factory));
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.model_args_loader));
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.quant_args_loader));
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.tokenizer_args_loader));
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.multimodal_processor_factory));
  EXPECT_EQ(descriptor.meta.cp_sharding_mode, CpShardingMode::MODEL);
  EXPECT_TRUE(descriptor.meta.supports_dcp);
  EXPECT_TRUE(descriptor.meta.owns_mtp_shared_weights);
}

TEST(ModelRegistryTest, RejectsUnknownDescriptorWithoutMutatingRegistry) {
  const std::string name = "descriptor_unknown_registration_test";
  ResolvedModelDescriptor descriptor;
  std::string error_message;

  EXPECT_FALSE(ModelRegistry::resolve_model_descriptor(
      name, "", &descriptor, &error_message));
  EXPECT_NE(error_message.find(name), std::string::npos);
  EXPECT_EQ(ModelRegistry::get_model_family(name), ModelFamily::UNKNOWN);
  EXPECT_TRUE(ModelRegistry::get_model_backend(name).empty());
}

TEST(ModelRegistryTest, ResolvesDiTDescriptorWithoutNpuBackendResolution) {
  const std::string name = "descriptor_dit_pipeline_test";
  ModelRegistry::register_dit_model_factory(
      name, [](const DiTModelContext& /*context*/) {
        return std::unique_ptr<DiTModel>();
      });

  ResolvedModelDescriptor descriptor;
  std::string error_message;
  ASSERT_TRUE(ModelRegistry::resolve_model_descriptor(
      name, "INVALID", &descriptor, &error_message))
      << error_message;
  EXPECT_EQ(descriptor.registration_name, name);
  EXPECT_EQ(descriptor.family, ModelFamily::DIT);
  EXPECT_EQ(descriptor.family_name, "dit");
  EXPECT_TRUE(descriptor.npu_kernel_backend.empty());
  EXPECT_TRUE(static_cast<bool>(descriptor.meta.dit_model_factory));
}

#if defined(USE_NPU)
TEST(ModelRegistryTest, ResolvesDescriptorForNpuBackendSpecificKey) {
  ResolvedModelDescriptor descriptor;
  std::string error_message;

  ASSERT_TRUE(ModelRegistry::resolve_model_descriptor(
      "qwen3", "AUTO", &descriptor, &error_message))
      << error_message;
  EXPECT_EQ(descriptor.registration_name, "qwen3_atb");
  EXPECT_EQ(descriptor.family, ModelFamily::LLM);
  EXPECT_EQ(descriptor.family_name, "llm");
  EXPECT_EQ(descriptor.npu_kernel_backend, "ATB");
  EXPECT_TRUE(descriptor.meta.causal_lm_factories.contains("native"));

  ASSERT_TRUE(ModelRegistry::resolve_model_descriptor(
      "qwen3", "TORCH", &descriptor, &error_message))
      << error_message;
  EXPECT_EQ(descriptor.registration_name, "qwen3");
  EXPECT_EQ(descriptor.npu_kernel_backend, "TORCH");
}
#endif

TEST(ModelRegistryTest, ExposesRegisteredModelFamily) {
  ModelRegistry::register_model_backend("model_family_llm_test", "llm");
  ModelRegistry::register_model_backend("model_family_vlm_test", "vlm");
  ModelRegistry::register_model_backend("model_family_rec_test", "rec");
  ModelRegistry::register_model_backend("model_family_dit_test", "dit");

  EXPECT_EQ(ModelRegistry::get_model_family("model_family_llm_test"),
            ModelFamily::LLM);
  EXPECT_EQ(ModelRegistry::get_model_family("model_family_vlm_test"),
            ModelFamily::VLM);
  EXPECT_EQ(ModelRegistry::get_model_family("model_family_rec_test"),
            ModelFamily::REC);
  EXPECT_EQ(ModelRegistry::get_model_family("model_family_dit_test"),
            ModelFamily::DIT);
  EXPECT_EQ(ModelRegistry::get_model_family("unknown_model_family"),
            ModelFamily::UNKNOWN);
}

TEST(ModelRegistryTest, PreservesFamilyForConflictingBackendRegistration) {
  const std::string name = "model_family_conflict_test";
  ModelRegistry::register_model_backend(name, "llm");
  ModelRegistry::register_model_backend(name, "vlm");

  EXPECT_EQ(ModelRegistry::get_model_backend(name), "llm");
  EXPECT_EQ(ModelRegistry::get_model_family(name), ModelFamily::LLM);
}

}  // namespace
}  // namespace xllm

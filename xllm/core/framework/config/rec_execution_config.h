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

#include <cstdint>
#include <optional>
#include <string_view>

#include "core/framework/batch/batch_input_type.h"
#include "core/framework/request/rec_type.h"

namespace xllm {

enum class RecModelKind : int8_t {
  kNone = 0,
  kOneRec = 1,
  kLlmRec = 2,
};

enum class RecPipelineType : uint8_t {
  kLlmRecDefault = 0,
  kLlmRecWithMmData = 1,
  kOneRecDefault = 2,
  kLlmRecMultiRoundPipeline = 3,
  kOneRecXAttentionPipeline = 4,
};

inline constexpr bool is_onerec_model_type(std::string_view model_type) {
  return model_type == "onerec";
}

inline constexpr bool is_llmrec_model_type(std::string_view model_type) {
  return model_type == "qwen2" || model_type == "qwen3" ||
         model_type == "qwen3_moe";
}

inline constexpr RecModelKind get_rec_model_kind(std::string_view model_type) {
  if (is_onerec_model_type(model_type)) {
    return RecModelKind::kOneRec;
  }
  return is_llmrec_model_type(model_type) ? RecModelKind::kLlmRec
                                          : RecModelKind::kNone;
}

// A resolved execution contract. It contains no process-global configuration.
class RecExecutionConfig final {
 public:
  static constexpr std::optional<RecExecutionConfig> resolve(
      RecModelKind model_kind,
      int32_t decode_rounds,
      bool enable_prefill_only) {
    if ((model_kind != RecModelKind::kOneRec &&
         model_kind != RecModelKind::kLlmRec) ||
        decode_rounds < 0) {
      return std::nullopt;
    }
    return RecExecutionConfig(model_kind,
                              decode_rounds,
                              model_kind == RecModelKind::kOneRec &&
                                  decode_rounds == 0 && enable_prefill_only);
  }

  static constexpr std::optional<RecExecutionConfig> resolve(
      std::string_view model_type,
      int32_t decode_rounds,
      bool enable_prefill_only) {
    return resolve(
        get_rec_model_kind(model_type), decode_rounds, enable_prefill_only);
  }

  // Deterministic defaults for callers that specify only an input contract.
  // A production deployment supplies its resolved configuration explicitly.
  explicit constexpr RecExecutionConfig(BatchInputType input_type)
      : model_kind_(
            input_type == BatchInputType::SEQUENCE ||
                    input_type == BatchInputType::REC_MULTI_ROUND
                ? RecModelKind::kLlmRec
                : (input_type == BatchInputType::ONEREC ||
                           input_type == BatchInputType::ONEREC_XATTENTION
                       ? RecModelKind::kOneRec
                       : RecModelKind::kNone)),
        decode_rounds_(input_type == BatchInputType::REC_MULTI_ROUND ||
                               input_type == BatchInputType::ONEREC_XATTENTION
                           ? 1
                           : 0) {}

  constexpr RecModelKind model_kind() const { return model_kind_; }
  constexpr bool valid() const { return model_kind_ != RecModelKind::kNone; }
  constexpr RecType rec_type() const {
    if (!valid()) {
      return RecType::kNone;
    }
    return model_kind_ == RecModelKind::kOneRec ? RecType::kOneRec
                                                : RecType::kLlmRec;
  }
  constexpr int32_t decode_rounds() const { return decode_rounds_; }
  constexpr bool is_multi_round() const { return decode_rounds_ > 0; }
  constexpr bool use_legacy_onerec_prefill_only_contract() const {
    return legacy_prefill_only_;
  }
  constexpr RecPipelineType pipeline_type() const {
    if (!valid()) {
      return static_cast<RecPipelineType>(255);
    }
    if (model_kind_ == RecModelKind::kOneRec) {
      return is_multi_round() ? RecPipelineType::kOneRecXAttentionPipeline
                              : RecPipelineType::kOneRecDefault;
    }
    return is_multi_round() ? RecPipelineType::kLlmRecMultiRoundPipeline
                            : RecPipelineType::kLlmRecDefault;
  }
  constexpr BatchInputType input_type() const {
    if (!valid()) {
      return static_cast<BatchInputType>(-1);
    }
    if (model_kind_ == RecModelKind::kOneRec) {
      return is_multi_round() ? BatchInputType::ONEREC_XATTENTION
                              : BatchInputType::ONEREC;
    }
    return is_multi_round() ? BatchInputType::REC_MULTI_ROUND
                            : BatchInputType::SEQUENCE;
  }
  constexpr bool uses_group_input() const {
    return model_kind_ == RecModelKind::kOneRec;
  }
  constexpr bool requires_kv_cache() const {
    return input_type() == BatchInputType::SEQUENCE ||
           input_type() == BatchInputType::ONEREC_XATTENTION;
  }
  constexpr bool supports_data_parallel() const {
    return model_kind_ == RecModelKind::kLlmRec && !is_multi_round();
  }
  constexpr bool supports_multi_node() const {
    return supports_data_parallel();
  }
  constexpr std::optional<std::string_view> validate_topology(
      int32_t dp_size,
      int32_t nnodes) const {
    if (!valid()) {
      return "Unsupported Rec execution input type";
    }
    if (dp_size <= 0 || nnodes <= 0) {
      return "REC topology requires positive dp_size and nnodes";
    }
    if (dp_size != 1 && !supports_data_parallel()) {
      return "Only single-round LlmRec supports data parallelism";
    }
    if (nnodes != 1 && !supports_multi_node()) {
      return "Only single-round LlmRec supports multi-node execution";
    }
    return std::nullopt;
  }

  constexpr bool operator==(const RecExecutionConfig&) const = default;

 private:
  constexpr RecExecutionConfig(RecModelKind model_kind,
                               int32_t decode_rounds,
                               bool legacy_prefill_only)
      : model_kind_(model_kind),
        decode_rounds_(decode_rounds),
        legacy_prefill_only_(legacy_prefill_only) {}

  RecModelKind model_kind_;
  int32_t decode_rounds_;
  bool legacy_prefill_only_ = false;
};

}  // namespace xllm

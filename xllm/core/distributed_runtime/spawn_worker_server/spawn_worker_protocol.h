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

#include <array>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "core/framework/config/rec_execution_config.h"

namespace xllm::spawn_worker_protocol {

inline constexpr int32_t kArgumentCount = 41;
inline constexpr int32_t kMinimumArgumentCount = 34;
inline constexpr int32_t kIndexerCacheDtypeArgumentIndex = 34;
inline constexpr int32_t kEnableMtpDraftBodyTp1ArgumentIndex = 35;
inline constexpr int32_t kTextEncoderTpSizeArgumentIndex = 36;
inline constexpr int32_t kDraftSamplingModeArgumentIndex = 37;
inline constexpr int32_t kRecModelKindArgumentIndex = 38;
inline constexpr int32_t kRecDecodeRoundsArgumentIndex = 39;
inline constexpr int32_t kRecLegacyPrefillOnlyArgumentIndex = 40;
inline constexpr char kDefaultIndexerCacheDtype[] = "auto";
inline constexpr char kDefaultDraftSamplingMode[] = "greedy";

inline std::optional<std::string> parse_indexer_cache_dtype(
    int32_t argc,
    char* const argv[]) {
  if (argc < kMinimumArgumentCount || argv == nullptr) {
    return std::nullopt;
  }

  if (argc == kMinimumArgumentCount) {
    return std::string(kDefaultIndexerCacheDtype);
  }

  if (argv[kIndexerCacheDtypeArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  return std::string(argv[kIndexerCacheDtypeArgumentIndex]);
}

inline std::string parse_draft_sampling_mode(int32_t argc, char* const argv[]) {
  if (argv == nullptr || argc <= kDraftSamplingModeArgumentIndex ||
      argv[kDraftSamplingModeArgumentIndex] == nullptr) {
    return std::string(kDefaultDraftSamplingMode);
  }
  return std::string(argv[kDraftSamplingModeArgumentIndex]);
}

inline std::array<std::string, 3> encode_rec_execution_config(
    const std::optional<RecExecutionConfig>& config) {
  if (!config.has_value()) {
    return {"0", "0", "0"};
  }
  return {std::to_string(static_cast<int32_t>(config->model_kind())),
          std::to_string(config->decode_rounds()),
          config->use_legacy_onerec_prefill_only_contract() ? "1" : "0"};
}

inline std::optional<RecExecutionConfig> parse_rec_execution_config(
    int32_t argc,
    char* const argv[]) {
  if (argc <= kRecLegacyPrefillOnlyArgumentIndex || argv == nullptr) {
    return std::nullopt;
  }
  std::array<int32_t, 3> values;
  for (size_t i = 0; i < values.size(); ++i) {
    const char* argument = argv[kRecModelKindArgumentIndex + i];
    if (argument == nullptr) {
      return std::nullopt;
    }
    const std::string_view text(argument);
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), values[i]);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
      return std::nullopt;
    }
  }
  if (values[0] < static_cast<int32_t>(RecModelKind::kOneRec) ||
      values[0] > static_cast<int32_t>(RecModelKind::kLlmRec) ||
      (values[2] != 0 && values[2] != 1)) {
    return std::nullopt;
  }
  const auto config = RecExecutionConfig::resolve(
      static_cast<RecModelKind>(values[0]), values[1], values[2] != 0);
  if (!config.has_value() ||
      config->use_legacy_onerec_prefill_only_contract() != (values[2] != 0)) {
    return std::nullopt;
  }
  return config;
}

}  // namespace xllm::spawn_worker_protocol

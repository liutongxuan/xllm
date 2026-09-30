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
#include <utility>

#include "core/framework/config/rec_execution_config.h"

namespace xllm {

enum class RecExecutionBindingResult : uint8_t {
  INITIALIZED,
  UNCHANGED,
  CONFLICT,
};

// Startup binding for legacy process-wide Rec implementations. The caller
// serializes access; conflicts preserve the first execution contract.
class RecExecutionConfigBinding final {
 public:
  RecExecutionBindingResult bind(RecExecutionConfig config) {
    if (!config.valid()) {
      return RecExecutionBindingResult::CONFLICT;
    }
    if (config_.has_value()) {
      return config_.value() == config ? RecExecutionBindingResult::UNCHANGED
                                       : RecExecutionBindingResult::CONFLICT;
    }
    config_ = std::move(config);
    return RecExecutionBindingResult::INITIALIZED;
  }

  const std::optional<RecExecutionConfig>& config() const { return config_; }

 private:
  std::optional<RecExecutionConfig> config_;
};

}  // namespace xllm

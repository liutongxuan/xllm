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
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"

namespace xllm {

struct TokenizerArgs;

enum class ChatGenerationMode : uint8_t {
  CHAT = 0,
  REASONING = 1,
  UNKNOWN = 2,
};

struct RenderedChatPrompt final {
  std::string prompt;
  ChatGenerationMode generation_mode = ChatGenerationMode::UNKNOWN;
};

// Request preparation consumes rendered prompts without knowing template
// syntax or model-specific rendering implementations. Implementations must
// support concurrent calls from the master's request workers.
class ChatPromptRenderer {
 public:
  virtual ~ChatPromptRenderer() = default;

  virtual std::optional<RenderedChatPrompt> render(
      const ChatMessages& messages,
      const std::vector<JsonTool>& tools,
      const nlohmann::ordered_json& kwargs) const = 0;
};

// Masters invoke this after loading tokenizer metadata, only on serving ranks.
// The factory owns its captured configuration; an LLM fork copies that policy
// and constructs a new renderer using the fork's own tokenizer/model metadata.
using ChatPromptRendererFactory =
    std::function<std::unique_ptr<ChatPromptRenderer>(const TokenizerArgs&,
                                                      const std::string&)>;

}  // namespace xllm

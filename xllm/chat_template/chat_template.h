/* Copyright 2025-2026 The xLLM Authors.

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
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "chat_template/chat_template_input.h"

namespace xllm::chat_template {

enum class GenerationMode : uint8_t {
  CHAT = 0,
  REASONING = 1,
  UNKNOWN = 2,
};

struct RenderResult {
  std::string prompt;
  GenerationMode generation_mode = GenerationMode::UNKNOWN;
};

}  // namespace xllm::chat_template

namespace xllm {

// Immutable after construction; supports concurrent synchronous rendering.
class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  std::optional<chat_template::RenderResult> render(
      const std::vector<chat_template::MessageView>& messages,
      const std::vector<chat_template::ToolView>& tools = {},
      const nlohmann::ordered_json& kwargs = {}) const;

  // Compatibility classification for the existing think-marker protocol.
  // Unrecognized suffixes stay unknown; this does not infer request intent.
  static chat_template::GenerationMode generation_mode_from_prompt(
      std::string_view prompt);

 protected:
  virtual std::optional<std::string> render_prompt(
      const std::vector<chat_template::MessageView>& messages,
      const std::vector<chat_template::ToolView>& tools,
      const nlohmann::ordered_json& kwargs) const = 0;
};

}  // namespace xllm

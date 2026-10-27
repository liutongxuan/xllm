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

#include "chat_template/chat_template.h"

#include <utility>

namespace xllm {

std::optional<RenderedChatPrompt> ChatTemplate::render(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  std::optional<std::string> prompt =
      apply(messages, json_tools, chat_template_kwargs);
  if (!prompt.has_value()) {
    return std::nullopt;
  }
  const ChatGenerationMode generation_mode =
      generation_mode_from_prompt(prompt.value());
  return RenderedChatPrompt{std::move(prompt.value()), generation_mode};
}

ChatGenerationMode ChatTemplate::generation_mode_from_prompt(
    std::string_view prompt) {
  constexpr std::string_view kJsonWhitespace = " \t\n\r";
  while (!prompt.empty() &&
         kJsonWhitespace.find(prompt.back()) != std::string_view::npos) {
    prompt.remove_suffix(1);
  }
  constexpr std::string_view kThinkingMarker = "<think>";
  constexpr std::string_view kChatMarker = "</think>";
  if (prompt.ends_with(kThinkingMarker)) {
    return ChatGenerationMode::REASONING;
  }
  if (prompt.ends_with(kChatMarker)) {
    return ChatGenerationMode::CHAT;
  }
  return ChatGenerationMode::UNKNOWN;
}

}  // namespace xllm

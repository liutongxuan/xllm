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

#include <utility>

#include "chat_template/chat_template_renderer.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace {

inline std::optional<chat_template::RenderResult> render_template(
    const ChatTemplate& chat_template,
    const ChatMessages& messages,
    const std::vector<JsonTool>& tools = {},
    const nlohmann::ordered_json& kwargs = {}) {
  return chat_template.render(
      make_template_messages(messages), make_template_tools(tools), kwargs);
}

inline std::optional<std::string> apply_template(
    const ChatTemplate& chat_template,
    const ChatMessages& messages,
    const std::vector<JsonTool>& tools = {},
    const nlohmann::ordered_json& kwargs = {}) {
  auto result = render_template(chat_template, messages, tools, kwargs);
  if (!result.has_value()) {
    return std::nullopt;
  }
  return std::move(result->prompt);
}

inline std::optional<std::string> apply_template(
    const ChatTemplate& chat_template,
    const ChatMessages& messages,
    const nlohmann::ordered_json& kwargs) {
  return apply_template(chat_template, messages, {}, kwargs);
}

template <typename Template>
std::optional<std::string> apply_template(
    const Template& chat_template,
    nlohmann::ordered_json messages,
    nlohmann::ordered_json tools = nlohmann::ordered_json::array(),
    const nlohmann::ordered_json& kwargs = {}) {
  return chat_template.apply(std::move(messages), std::move(tools), kwargs);
}

}  // namespace
}  // namespace xllm

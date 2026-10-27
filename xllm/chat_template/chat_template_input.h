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

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace xllm::chat_template {

struct TemplateConfig {
  std::string source;
  std::string bos_token;
  std::string eos_token;
};

// Views borrow request data for one synchronous render. Templates never retain
// them. Media payloads are deliberately absent: rendering only needs their type
// and position in the conversation.
struct ContentPartView {
  std::string_view type;
  std::string_view text;
};

struct ToolCallView {
  struct Function {
    std::string_view name;
    std::string_view arguments;
  };
  std::string_view id;
  std::string_view type;
  Function function;
};

struct MessageView {
  using Content = std::variant<std::string_view, std::vector<ContentPartView>>;
  std::string_view role;
  Content content;
  std::optional<std::string_view> tool_call_id;
  std::optional<std::string_view> reasoning_content;
  std::optional<std::vector<ToolCallView>> tool_calls;
};

struct ToolView {
  struct Function {
    std::string_view name;
    std::string_view description;
    const nlohmann::ordered_json& parameters;
    std::optional<bool> defer_loading;
  };
  std::string_view type;
  Function function;
};

}  // namespace xllm::chat_template

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

#include <memory>
#include <vector>

#include "chat_template/chat_template.h"
#include "core/framework/request/chat_prompt_renderer.h"

namespace xllm {

// Returned views borrow messages/tools and must not outlive a synchronous
// render.
std::vector<chat_template::MessageView> make_template_messages(
    const ChatMessages& messages);
std::vector<chat_template::ToolView> make_template_tools(
    const std::vector<JsonTool>& tools);
chat_template::TemplateConfig make_template_config(const TokenizerArgs& args);

// Adapts the independently usable template module to core request preparation.
class ChatTemplateRenderer final : public ChatPromptRenderer {
 public:
  explicit ChatTemplateRenderer(std::unique_ptr<ChatTemplate> chat_template);

  std::optional<RenderedChatPrompt> render(
      const ChatMessages& messages,
      const std::vector<JsonTool>& tools,
      const nlohmann::ordered_json& kwargs) const override;

 private:
  std::unique_ptr<ChatTemplate> chat_template_;
};

ChatPromptRendererFactory create_chat_prompt_renderer_factory(
    bool use_cpp_template);

}  // namespace xllm

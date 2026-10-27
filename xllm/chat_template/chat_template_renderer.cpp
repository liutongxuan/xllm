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

#include "chat_template/chat_template_renderer.h"

#include <glog/logging.h>

#include <utility>
#include <variant>

#include "chat_template/chat_template_factory.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {

std::vector<chat_template::MessageView> make_template_messages(
    const ChatMessages& messages) {
  std::vector<chat_template::MessageView> views;
  views.reserve(messages.size());
  for (const auto& message : messages) {
    chat_template::MessageView view;
    view.role = message.role;
    if (const auto* text = std::get_if<std::string>(&message.content)) {
      view.content = std::string_view(*text);
    } else {
      const auto& content = std::get<MMContentVec>(message.content);
      std::vector<chat_template::ContentPartView> parts;
      parts.reserve(content.size());
      for (const auto& part : content) {
        parts.emplace_back(
            chat_template::ContentPartView{part.type, part.text});
      }
      view.content = std::move(parts);
    }
    if (message.tool_call_id.has_value()) {
      view.tool_call_id = *message.tool_call_id;
    }
    if (message.reasoning_content.has_value()) {
      view.reasoning_content = *message.reasoning_content;
    }
    if (message.tool_calls.has_value()) {
      view.tool_calls.emplace();
      view.tool_calls->reserve(message.tool_calls->size());
      for (const auto& call : *message.tool_calls) {
        view.tool_calls->emplace_back(chat_template::ToolCallView{
            call.id, call.type, {call.function.name, call.function.arguments}});
      }
    }
    views.emplace_back(std::move(view));
  }
  return views;
}

std::vector<chat_template::ToolView> make_template_tools(
    const std::vector<JsonTool>& tools) {
  std::vector<chat_template::ToolView> views;
  views.reserve(tools.size());
  for (const auto& tool : tools) {
    views.emplace_back(chat_template::ToolView{tool.type,
                                               {tool.function.name,
                                                tool.function.description,
                                                tool.function.parameters,
                                                tool.function.defer_loading}});
  }
  return views;
}

chat_template::TemplateConfig make_template_config(const TokenizerArgs& args) {
  return {args.chat_template(), args.bos_token(), args.eos_token()};
}

ChatTemplateRenderer::ChatTemplateRenderer(
    std::unique_ptr<ChatTemplate> chat_template)
    : chat_template_(std::move(chat_template)) {
  CHECK(chat_template_ != nullptr);
}

std::optional<RenderedChatPrompt> ChatTemplateRenderer::render(
    const ChatMessages& messages,
    const std::vector<JsonTool>& tools,
    const nlohmann::ordered_json& kwargs) const {
  auto result = chat_template_->render(
      make_template_messages(messages), make_template_tools(tools), kwargs);
  if (!result.has_value()) {
    return std::nullopt;
  }
  ChatGenerationMode mode = ChatGenerationMode::UNKNOWN;
  switch (result->generation_mode) {
    case chat_template::GenerationMode::CHAT:
      mode = ChatGenerationMode::CHAT;
      break;
    case chat_template::GenerationMode::REASONING:
      mode = ChatGenerationMode::REASONING;
      break;
    case chat_template::GenerationMode::UNKNOWN:
      break;
  }
  return RenderedChatPrompt{std::move(result->prompt), mode};
}

ChatPromptRendererFactory create_chat_prompt_renderer_factory(
    bool use_cpp_template) {
  return [use_cpp_template](const TokenizerArgs& args,
                            const std::string& model_type) {
    return std::make_unique<ChatTemplateRenderer>(ChatTemplateFactory::create(
        make_template_config(args), model_type, use_cpp_template));
  };
}

}  // namespace xllm

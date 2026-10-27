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

#include "application/chat_template_renderer.h"

#include <gtest/gtest.h>

#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace {

TEST(ChatTemplateRenderer, RendererAcceptsRequestContract) {
  TokenizerArgs args;
  args.chat_template("{{ messages[0].content }} <think>\n");
  auto renderer = create_chat_prompt_renderer_factory(
      /*use_cpp_template=*/false)(args, /*model_type=*/"qwen3");
  const ChatMessages messages = {Message("user", std::string("hello"))};
  const auto rendered = renderer->render(
      messages, /*tools=*/{}, nlohmann::ordered_json::object());
  ASSERT_TRUE(rendered.has_value());
  EXPECT_EQ(rendered->prompt, "hello <think>");
  EXPECT_EQ(rendered->generation_mode, ChatGenerationMode::REASONING);
}

TEST(ChatTemplateRenderer, CopiedFactoryUsesFreshModelAndTokenizerMetadata) {
  const ChatPromptRendererFactory factory = [] {
    const bool use_cpp_template = true;
    return create_chat_prompt_renderer_factory(use_cpp_template);
  }();
  const ChatPromptRendererFactory fork_factory = factory;
  TokenizerArgs args;
  auto first = factory(args, /*model_type=*/"deepseek_v32");
  auto fork = fork_factory(args, /*model_type=*/"deepseek_v4");
  const ChatMessages messages = {Message("user", "hello")};
  const auto first_render = first->render(messages, {}, {});
  const auto fork_render = fork->render(messages, {}, {});
  ASSERT_TRUE(first_render.has_value());
  ASSERT_TRUE(fork_render.has_value());
  EXPECT_EQ(first_render->generation_mode, ChatGenerationMode::CHAT);
  EXPECT_EQ(fork_render->generation_mode, ChatGenerationMode::REASONING);

  const auto jinja_factory =
      create_chat_prompt_renderer_factory(/*use_cpp_template=*/false);
  args.chat_template("first model");
  auto first_model = jinja_factory(args, /*model_type=*/"qwen3");
  args.chat_template("fork model");
  auto fork_model = jinja_factory(args, /*model_type=*/"deepseek_v4");
  const auto first_prompt = first_model->render(
      /*messages=*/{}, /*tools=*/{}, nlohmann::ordered_json::object());
  const auto fork_prompt = fork_model->render(
      /*messages=*/{}, /*tools=*/{}, nlohmann::ordered_json::object());
  ASSERT_TRUE(first_prompt.has_value());
  ASSERT_TRUE(fork_prompt.has_value());
  EXPECT_EQ(first_prompt->prompt, "first model");
  EXPECT_EQ(fork_prompt->prompt, "fork model");
}

TEST(ChatTemplateRenderer, RendererPreservesRenderFailure) {
  TokenizerArgs args;
  args.chat_template("{{ raise_exception('invalid message') }}");
  auto renderer = create_chat_prompt_renderer_factory(
      /*use_cpp_template=*/false)(args, /*model_type=*/"qwen3");
  EXPECT_FALSE(renderer
                   ->render(/*messages=*/{},
                            /*tools=*/{},
                            nlohmann::ordered_json::object())
                   .has_value());
}

TEST(ChatTemplateRenderer, ProjectsBorrowedTextAndSchemaWithoutPayloadCopies) {
  ChatMessages messages = {Message("assistant", std::string(4096, 'x'))};
  messages[0].tool_call_id = "call-id";
  messages[0].reasoning_content = "reason";
  messages[0].tool_calls =
      Message::ToolCallVec{{"call-id", "function", {"search", "{\"q\":1}"}}};
  const auto views = make_template_messages(messages);
  ASSERT_EQ(views.size(), 1);
  EXPECT_EQ(std::get<std::string_view>(views[0].content).data(),
            std::get<std::string>(messages[0].content).data());
  ASSERT_TRUE(views[0].reasoning_content.has_value());
  EXPECT_EQ(views[0].reasoning_content->data(),
            messages[0].reasoning_content->data());
  ASSERT_TRUE(views[0].tool_calls.has_value());
  EXPECT_EQ(views[0].tool_calls->at(0).function.arguments.data(),
            messages[0].tool_calls->at(0).function.arguments.data());
  EXPECT_EQ(views[0].tool_call_id, "call-id");

  std::vector<JsonTool> tools(1);
  tools[0].function.parameters = {{"z", 1}, {"a", 2}};
  tools[0].function.defer_loading = false;
  const auto tool_views = make_template_tools(tools);
  ASSERT_EQ(tool_views.size(), 1);
  EXPECT_EQ(&tool_views[0].function.parameters, &tools[0].function.parameters);
  ASSERT_TRUE(tool_views[0].function.defer_loading.has_value());
  EXPECT_FALSE(*tool_views[0].function.defer_loading);
}

TEST(ChatTemplateRenderer, PreservesContentOrderAndAbsentVersusEmptyFields) {
  ChatMessages messages = {
      Message("user",
              MMContentVec{MMContent("text", "first"),
                           MMContent("image_url", ImageURL{"unused", {}}),
                           MMContent("text", "last")}),
      Message("assistant", "")};
  messages[1].tool_call_id = "";
  messages[1].reasoning_content = "";
  messages[1].tool_calls.emplace();
  const auto views = make_template_messages(messages);
  ASSERT_EQ(views.size(), 2);
  EXPECT_FALSE(views[0].tool_call_id.has_value());
  EXPECT_FALSE(views[0].reasoning_content.has_value());
  EXPECT_FALSE(views[0].tool_calls.has_value());
  const auto& parts =
      std::get<std::vector<chat_template::ContentPartView>>(views[0].content);
  ASSERT_EQ(parts.size(), 3);
  EXPECT_EQ(parts[0].text, "first");
  EXPECT_EQ(parts[1].type, "image_url");
  EXPECT_EQ(parts[2].text, "last");
  ASSERT_TRUE(views[1].tool_call_id.has_value());
  ASSERT_TRUE(views[1].reasoning_content.has_value());
  ASSERT_TRUE(views[1].tool_calls.has_value());
  EXPECT_TRUE(views[1].tool_calls->empty());
}

}  // namespace
}  // namespace xllm

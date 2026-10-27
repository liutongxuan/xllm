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

#include "chat_template/chat_template.h"

#include <gtest/gtest.h>

#include "chat_template/chat_template_factory.h"
#include "chat_template/deepseek_v32_cpp_template.h"
#include "chat_template/deepseek_v4_cpp_template.h"
#include "chat_template/jinja_chat_template.h"

namespace xllm {
namespace {

TEST(ChatTemplateFactory, DeepseekV32UsesJinjaWhenNativeRenderingDisabled) {
  TokenizerArgs args;
  auto impl = ChatTemplateFactory::create(
      args, /*model_type=*/"deepseek_v32", /*use_cpp_template=*/false);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesJinjaWhenNativeRenderingDisabled) {
  TokenizerArgs args;
  auto impl = ChatTemplateFactory::create(
      args, /*model_type=*/"deepseek_v4", /*use_cpp_template=*/false);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV32UsesNativeRendererWhenEnabled) {
  TokenizerArgs args;
  auto impl = ChatTemplateFactory::create(
      args, /*model_type=*/"deepseek_v32", /*use_cpp_template=*/true);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesNativeRendererWhenEnabled) {
  TokenizerArgs args;
  auto impl = ChatTemplateFactory::create(
      args, /*model_type=*/"deepseek_v4", /*use_cpp_template=*/true);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, UnsupportedNativeModelFailsExplicitly) {
  TokenizerArgs args;
  EXPECT_DEATH(ChatTemplateFactory::create(
                   args, /*model_type=*/"qwen3", /*use_cpp_template=*/true),
               "only support");
}

TEST(ChatTemplateFactory, RendererAcceptsRequestContract) {
  TokenizerArgs args;
  args.chat_template("{{ messages[0].content }} <think>\n");
  std::unique_ptr<ChatPromptRenderer> renderer = ChatTemplateFactory::create(
      args, /*model_type=*/"qwen3", /*use_cpp_template=*/false);
  const ChatMessages messages = {Message("user", std::string("hello"))};
  const auto rendered = renderer->render(
      messages, /*tools=*/{}, nlohmann::ordered_json::object());
  ASSERT_TRUE(rendered.has_value());
  EXPECT_EQ(rendered->prompt, "hello <think>");
  EXPECT_EQ(rendered->generation_mode, ChatGenerationMode::REASONING);
}

TEST(ChatTemplateFactory, CopiedFactoryUsesFreshModelAndTokenizerMetadata) {
  const ChatPromptRendererFactory factory = [] {
    const bool use_cpp_template = true;
    return ChatTemplateFactory::renderer_factory(use_cpp_template);
  }();
  const ChatPromptRendererFactory fork_factory = factory;
  TokenizerArgs args;
  auto first = factory(args, /*model_type=*/"deepseek_v32");
  auto fork = fork_factory(args, /*model_type=*/"deepseek_v4");
  EXPECT_NE(dynamic_cast<DeepseekV32CppTemplate*>(first.get()), nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV4CppTemplate*>(fork.get()), nullptr);

  const auto jinja_factory =
      ChatTemplateFactory::renderer_factory(/*use_cpp_template=*/false);
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

TEST(ChatTemplateFactory, RendererPreservesRenderFailure) {
  TokenizerArgs args;
  args.chat_template("{{ raise_exception('invalid message') }}");
  std::unique_ptr<ChatPromptRenderer> renderer = ChatTemplateFactory::create(
      args, /*model_type=*/"qwen3", /*use_cpp_template=*/false);
  EXPECT_FALSE(renderer
                   ->render(/*messages=*/{},
                            /*tools=*/{},
                            nlohmann::ordered_json::object())
                   .has_value());
}

}  // namespace
}  // namespace xllm

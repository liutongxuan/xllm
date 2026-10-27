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
  chat_template::TemplateConfig config;
  auto impl = ChatTemplateFactory::create(
      config, /*model_type=*/"deepseek_v32", /*use_cpp_template=*/false);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesJinjaWhenNativeRenderingDisabled) {
  chat_template::TemplateConfig config;
  auto impl = ChatTemplateFactory::create(
      config, /*model_type=*/"deepseek_v4", /*use_cpp_template=*/false);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV32UsesNativeRendererWhenEnabled) {
  chat_template::TemplateConfig config;
  auto impl = ChatTemplateFactory::create(
      config, /*model_type=*/"deepseek_v32", /*use_cpp_template=*/true);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesNativeRendererWhenEnabled) {
  chat_template::TemplateConfig config;
  auto impl = ChatTemplateFactory::create(
      config, /*model_type=*/"deepseek_v4", /*use_cpp_template=*/true);
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, UnsupportedNativeModelFailsExplicitly) {
  chat_template::TemplateConfig config;
  EXPECT_DEATH(ChatTemplateFactory::create(
                   config, /*model_type=*/"qwen3", /*use_cpp_template=*/true),
               "only support");
}

TEST(ChatTemplate, OwnsConfigurationAndRenderedPrompt) {
  chat_template::TemplateConfig config{
      "{{ bos_token }}{{ messages[0].content }}{{ eos_token }} <think>",
      "<bos>",
      "<eos>"};
  auto renderer = ChatTemplateFactory::create(
      config, /*model_type=*/"qwen3", /*use_cpp_template=*/false);
  config = {};
  std::optional<chat_template::RenderResult> result;
  {
    const std::string content("hello\0world", 11);
    chat_template::MessageView message;
    message.role = "user";
    message.content = std::string_view(content);
    result = renderer->render({message});
  }
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(
      result->prompt,
      std::string("<bos>") + std::string("hello\0world", 11) + "<eos> <think>");
  EXPECT_EQ(result->generation_mode, chat_template::GenerationMode::REASONING);
}

TEST(ChatTemplate, PreservesSchemaOrderAndOptionalToolMetadata) {
  const nlohmann::ordered_json parameters = {{"z", {{"type", "string"}}},
                                             {"a", {{"type", "integer"}}}};
  const chat_template::ToolView tool{
      "function", {"search", "description", parameters, false}};
  chat_template::TemplateConfig config;
  config.source = "{% for tool in tools %}{{ tool | tojson }}{% endfor %}";
  auto renderer = ChatTemplateFactory::create(
      config, /*model_type=*/"qwen3", /*use_cpp_template=*/false);
  const auto result = renderer->render({}, {tool});
  ASSERT_TRUE(result.has_value());
  const auto json = nlohmann::ordered_json::parse(result->prompt);
  EXPECT_EQ(json["function"]["parameters"].dump(), parameters.dump());
  EXPECT_EQ(json["function"]["defer_loading"], false);
  EXPECT_EQ(result->generation_mode, chat_template::GenerationMode::UNKNOWN);
}

TEST(ChatTemplate, NativeTemplatesKeepTypedContentSeparateFromPlainText) {
  chat_template::MessageView typed_message;
  typed_message.role = "user";
  typed_message.content = std::vector<chat_template::ContentPartView>{
      {"text", "not a native plain-text message"}, {"image_url", ""}};
  chat_template::MessageView empty_message;
  empty_message.role = "user";
  empty_message.content = std::string_view("");
  for (const std::string model : {"deepseek_v32", "deepseek_v4"}) {
    auto renderer =
        ChatTemplateFactory::create({}, model, /*use_cpp_template=*/true);
    const auto typed = renderer->render({typed_message});
    const auto empty = renderer->render({empty_message});
    ASSERT_TRUE(typed.has_value());
    ASSERT_TRUE(empty.has_value());
    EXPECT_EQ(typed->prompt, empty->prompt);
  }
}

TEST(ChatTemplate, RenderErrorsRemainExplicit) {
  chat_template::TemplateConfig config;
  config.source = "{{ raise_exception('invalid message') }}";
  auto renderer = ChatTemplateFactory::create(
      config, /*model_type=*/"qwen3", /*use_cpp_template=*/false);
  EXPECT_FALSE(renderer->render({}).has_value());
}

}  // namespace
}  // namespace xllm

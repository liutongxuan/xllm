include(cc_library)
include(cc_binary)

cc_library(
  NAME
    chat_template_renderer
  HDRS
    chat_template_renderer.h
  SRCS
    chat_template_renderer.cpp
  DEPS
    :chat_prompt_renderer
    :chat_template_factory
    glog::glog
)

# Shared composition for the server and offline C/C++/Python entry points.
cc_library(
  NAME
    master_factory
  HDRS
    master_factory.h
  SRCS
    master_factory.cpp
  DEPS
    :master
    :chat_template_renderer
    :config
    glog::glog
)

# Request-path hop 2: the chat-template render LLMRequestFactory performs on
# Qwen2.5 Jinja and DeepSeek V3.2/V4 native templates. Compare Release builds.
cc_binary(
  NAME
    chat_template_benchmark
  SRCS
    chat_template_benchmark.cpp
  DEPS
    :chat_template_renderer
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    nlohmann_json::nlohmann_json
    benchmark::benchmark
    benchmark::benchmark_main
)

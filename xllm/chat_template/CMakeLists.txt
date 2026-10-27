include(cc_library)

# Base class (independent of concrete implementations)
cc_library (
  NAME
    chat_template
  HDRS
    chat_template.h
    chat_template_input.h
  SRCS
    chat_template.cpp
  DEPS
    nlohmann_json::nlohmann_json
)

cc_library (
  NAME
    jinja_chat_template
  HDRS
    jinja_chat_template.h
  SRCS
    jinja_chat_template.cpp
  DEPS
    :chat_template
    :minja
    absl::strings
    nlohmann_json::nlohmann_json
    glog::glog
)

cc_library (
  NAME
    deepseek_v32_cpp_template
  HDRS
    deepseek_v32_cpp_template.h
  SRCS
    deepseek_v32_cpp_template.cpp
  DEPS
    :chat_template
    nlohmann_json::nlohmann_json
    glog::glog
    absl::strings
)

cc_library (
  NAME
    deepseek_v4_cpp_template
  HDRS
    deepseek_v4_cpp_template.h
  SRCS
    deepseek_v4_cpp_template.cpp
  DEPS
    :chat_template
    nlohmann_json::nlohmann_json
    glog::glog
    absl::strings
)

# Factory (depends on all implementations)
cc_library (
  NAME
    chat_template_factory
  HDRS
    chat_template_factory.h
  SRCS
    chat_template_factory.cpp
  DEPS
    :chat_template
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    nlohmann_json::nlohmann_json
    glog::glog
)

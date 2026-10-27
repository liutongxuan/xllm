include(cc_library)

cc_library(
  NAME
    xllm_server
  HDRS
    xllm_server.h
    xllm_server_registry.h
  SRCS
    xllm_server.cpp
    xllm_server_registry.cpp
  DEPS
    :api_service
    :request
    absl::strings
    glog::glog
    proto::xllm_proto
)

# Application assembly supplies concrete renderers to the core masters.
cc_library(
  NAME
    master_factory
  HDRS
    master_factory.h
  SRCS
    master_factory.cpp
  DEPS
    :master
    :chat_template_factory
    :config
    glog::glog
)

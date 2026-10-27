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

#include "server/master_factory.h"

#include <glog/logging.h>

#include <memory>
#include <string>

#include "chat_template/chat_template_factory.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/model_config.h"

namespace xllm {

std::unique_ptr<LLMMaster> create_llm_master(const Options& options) {
  return std::make_unique<LLMMaster>(
      options,
      ChatTemplateFactory::renderer_factory(
          ModelConfig::get_instance().use_cpp_chat_template()));
}

std::unique_ptr<VLMMaster> create_vlm_master(const Options& options) {
  return std::make_unique<VLMMaster>(
      options,
      ChatTemplateFactory::renderer_factory(/*use_cpp_template=*/false));
}

std::unique_ptr<RecMaster> create_rec_master(const Options& options) {
  return std::make_unique<RecMaster>(
      options,
      ChatTemplateFactory::renderer_factory(/*use_cpp_template=*/false));
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return create_llm_master(options);
  }
  if (backend == "vlm") {
    return create_vlm_master(options);
  }
  if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  }
  if (backend == "rec") {
    LOG(INFO) << "creating rec master";
    return create_rec_master(options);
  }

  LOG(FATAL) << "Failed to create master, backend is " << backend;
  return nullptr;
}

}  // namespace xllm

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
#include <mutex>
#include <utility>
#include <vector>

#include "core/framework/request/request.h"

namespace xllm {

// Response callbacks submit cancellations; the scheduling thread applies them.
class CancelRequestQueue final {
 public:
  void submit(std::shared_ptr<Request> request) {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.emplace_back(std::move(request));
  }

  std::vector<std::shared_ptr<Request>> take_all() {
    std::vector<std::shared_ptr<Request>> requests;
    std::lock_guard<std::mutex> lock(mutex_);
    requests.swap(requests_);
    return requests;
  }

 private:
  std::mutex mutex_;
  std::vector<std::shared_ptr<Request>> requests_;
};

}  // namespace xllm

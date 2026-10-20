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

#include "core/distributed_runtime/collective_service.h"

#include <brpc/closure_guard.h>

namespace xllm {

// Communicators rendezvous through their own stores. Creating unused HCCL
// root infos here leaves topology-server threads waiting during process exit.
CollectiveService::CollectiveService(int32_t total_num)
    : total_num_(total_num) {}

void CollectiveService::Sync(::google::protobuf::RpcController* /*controller*/,
                             const proto::AddressInfo* request,
                             proto::CommUniqueIdList* /*response*/,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  std::string address = request->address();
  int32_t global_rank = request->global_rank();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    addrs_map_[global_rank] = address;
  }
  registration_cv_.notify_all();
}

std::unordered_map<int32_t, std::string> CollectiveService::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  registration_cv_.wait(lock, [this]() {
    return addrs_map_.size() >= static_cast<size_t>(total_num_);
  });
  return addrs_map_;
}

std::optional<std::unordered_map<int32_t, std::string>>
CollectiveService::wait_until(std::chrono::steady_clock::time_point deadline) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!registration_cv_.wait_until(lock, deadline, [this]() {
        return addrs_map_.size() >= static_cast<size_t>(total_num_);
      })) {
    return std::nullopt;
  }
  return addrs_map_;
}

}  // namespace xllm

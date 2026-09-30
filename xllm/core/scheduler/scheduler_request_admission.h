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

#include <folly/MPMCQueue.h>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

#include "core/framework/block/kv_cache_manager.h"
#include "core/framework/request/request.h"

namespace xllm {

// Bounds admission across ready requests and in-flight storage prefetches.
// The queue and cache manager outlive this component; the scheduling owner
// must call shutdown() before destroying state used by enqueue_ready.
class SchedulerRequestAdmission final {
 public:
  using ReadyCallback = std::function<void(std::shared_ptr<Request>)>;

  SchedulerRequestAdmission(
      KVCacheManager* kv_cache_manager,
      folly::MPMCQueue<std::shared_ptr<Request>>& request_queue,
      ReadyCallback enqueue_ready);
  ~SchedulerRequestAdmission();

  bool add_request(std::shared_ptr<Request> request);
  void drain();
  void shutdown();

  size_t num_prefetching_requests() const {
    return prefetching_requests_.load(std::memory_order_relaxed);
  }

 private:
  void drain_admissions();
  void drain_completed_prefetches();

  KVCacheManager* kv_cache_manager_;
  folly::MPMCQueue<std::shared_ptr<Request>>& request_queue_;
  ReadyCallback enqueue_ready_;
  std::atomic<size_t> prefetching_requests_{0};
  std::mutex mutex_;
  std::deque<std::shared_ptr<Request>> admissions_;
  std::deque<std::shared_ptr<Request>> completed_prefetches_;
};

}  // namespace xllm

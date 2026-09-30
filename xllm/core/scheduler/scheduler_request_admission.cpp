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

#include "core/scheduler/scheduler_request_admission.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

namespace xllm {

SchedulerRequestAdmission::SchedulerRequestAdmission(
    KVCacheManager* kv_cache_manager,
    folly::MPMCQueue<std::shared_ptr<Request>>& request_queue,
    ReadyCallback enqueue_ready)
    : kv_cache_manager_(kv_cache_manager),
      request_queue_(request_queue),
      enqueue_ready_(std::move(enqueue_ready)) {
  CHECK(kv_cache_manager_ != nullptr);
  CHECK(enqueue_ready_);
}

SchedulerRequestAdmission::~SchedulerRequestAdmission() {
  CHECK_EQ(prefetching_requests_.load(std::memory_order_acquire), 0u)
      << "Request admission destroyed with pending prefetch callbacks";
}

bool SchedulerRequestAdmission::add_request(std::shared_ptr<Request> request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t pending_before_reservation = num_prefetching_requests();
  const size_t queued_requests =
      static_cast<size_t>(std::max<ssize_t>(request_queue_.size(), 0));
  if (queued_requests + pending_before_reservation >=
      request_queue_.capacity()) {
    return false;
  }

  if (!kv_cache_manager_->has_storage_prefetch()) {
    return request_queue_.write(std::move(request));
  }

  prefetching_requests_.fetch_add(1, std::memory_order_relaxed);
  VLOG(1) << "[Mooncake][AdmissionPending] request=" << request->request_id();
  admissions_.emplace_back(std::move(request));
  return true;
}

void SchedulerRequestAdmission::drain_admissions() {
  std::deque<std::shared_ptr<Request>> requests;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    requests.swap(admissions_);
  }

  for (std::shared_ptr<Request>& request : requests) {
    if (request->finished() || request->cancelled()) {
      const size_t previous =
          prefetching_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
      continue;
    }

    kv_cache_manager_->prefetch_from_storage(
        std::move(request), [this](std::shared_ptr<Request> completed) {
          std::lock_guard<std::mutex> lock(mutex_);
          completed_prefetches_.emplace_back(std::move(completed));
        });
  }
}

void SchedulerRequestAdmission::drain_completed_prefetches() {
  std::deque<std::shared_ptr<Request>> completed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    completed.swap(completed_prefetches_);
  }

  for (std::shared_ptr<Request>& request : completed) {
    const bool cancelled = request->finished() || request->cancelled();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!cancelled) {
        enqueue_ready_(request);
      }
      const size_t previous =
          prefetching_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
    }
    VLOG(1) << (cancelled ? "[Mooncake][AdmissionCancelled] request="
                          : "[Mooncake][AdmissionReady] request=")
            << request->request_id();
  }
}

void SchedulerRequestAdmission::drain() {
  kv_cache_manager_->drain_prefetch_completions();
  drain_admissions();
  kv_cache_manager_->drain_prefetch_completions();
  drain_completed_prefetches();
}

void SchedulerRequestAdmission::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const std::shared_ptr<Request>& request : admissions_) {
      request->set_cancel();
    }
    const size_t unissued = admissions_.size();
    admissions_.clear();
    const size_t previous =
        prefetching_requests_.fetch_sub(unissued, std::memory_order_acq_rel);
    CHECK_GE(previous, unissued);
  }
  kv_cache_manager_->drain_prefetch_completions();
  drain_completed_prefetches();
  CHECK_EQ(prefetching_requests_.load(std::memory_order_acquire), 0u)
      << "Request admission shut down with pending prefetch callbacks";
}

}  // namespace xllm

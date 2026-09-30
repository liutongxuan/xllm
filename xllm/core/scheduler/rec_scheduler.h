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

#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <atomic>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <semaphore>
#include <string>
#include <utility>
#include <vector>

#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/distributed_runtime/engine_resources.h"
#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/batch/rec_batch_group.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/scheduler/async_response_processor.h"
#include "core/scheduler/cancel_request_queue.h"
#include "core/scheduler/request_priority_queue.h"
#include "core/scheduler/scheduler.h"
#include "core/scheduler/scheduler_request_admission.h"
#include "core/util/threadpool.h"

namespace xllm {

// Return value structure for schedule_request
struct RecScheduleResult {
  RecBatchGroup batches;
  std::vector<std::shared_ptr<Request>> requests;
  std::vector<Sequence*> sequences;
};

class RecScheduler : public Scheduler {
 public:
  class Options final {
   public:
    PROPERTY(int32_t, max_tokens_per_batch) = 20000;
    PROPERTY(int32_t, max_seqs_per_batch) = 256;
    PROPERTY(int32_t, request_queue_size) = 100000;
    PROPERTY(int32_t, dp_size) = 1;
    PROPERTY(std::optional<std::string>, instance_name);
    PROPERTY(std::optional<InstanceRole>,
             instance_role) = InstanceRole::DEFAULT;
    PROPERTY(bool, enable_schedule_overlap) = false;
    PROPERTY(bool, enable_service_routing) = false;
    PROPERTY(bool, disable_log_stats) = false;
    PROPERTY(std::string, priority_strategy) = "fcfs";
    PROPERTY(int32_t, rec_worker_max_concurrency) = 1;
  };
  template <typename TargetEngine>
    requires requires(TargetEngine& engine, RecBatchGroup& batches) {
      EngineResources::bind(engine);
      { engine.step(batches) } -> std::same_as<ForwardOutput>;
    }
  RecScheduler(TargetEngine* engine, Options options)
      : RecScheduler(
            [engine] {
              CHECK(engine != nullptr);
              return EngineResources::bind(*engine);
            }(),
            [engine](RecBatchGroup& batches) { return engine->step(batches); },
            std::move(options)) {}
  ~RecScheduler() override;

  bool add_request(std::shared_ptr<Request>& request) override;

  void incr_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }
  void decr_pending_requests() override {
    const size_t previous =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(previous, 0u) << "pending requests underflow";
  }
  size_t num_pending_requests() override {
    return pending_requests_.load(std::memory_order_relaxed);
  }
  bool has_pending_prefetch() const override {
    return request_admission_->num_prefetching_requests() > 0;
  }
  uint32_t get_waiting_requests_num() const override;
  void get_latency_metrics(std::vector<int64_t>& /*ttft*/,
                           std::vector<int64_t>& /*tbt*/) override {}
  const InstanceInfo& get_instance_info() override { return instance_info_; }

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

  void generate() override;

 protected:
  RecBatchGroup prepare_rec_batch();
  std::vector<std::shared_ptr<Request>> running_requests_;

 private:
  using RecStep = std::function<ForwardOutput(RecBatchGroup&)>;

  RecScheduler(EngineResources resources, RecStep rec_step, Options options);

  void apply_cancel_requests();

  const Options options_;
  EngineResources resources_;
  RecStep rec_step_;
  KVCacheManager* kv_cache_manager_;
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue_;
  std::unique_ptr<SchedulerRequestAdmission> request_admission_;
  std::unique_ptr<RequestPriorityQueue> prefill_queue_;
  std::vector<Sequence*> running_sequences_;
  std::vector<size_t> running_sequences_budgets_;
  std::shared_ptr<CancelRequestQueue> cancel_request_queue_;
  std::unique_ptr<AsyncResponseProcessor> response_processor_;
  std::atomic<size_t> pending_requests_{0};
  // Read by the service-routing thread; updated by the scheduling owner.
  std::atomic<size_t> waiting_requests_{0};
  bool enable_prefix_cache_ = false;
  InstanceInfo instance_info_;

  // Scheduler pipeline for different rec types
  class SchedulerPipeline {
   public:
    virtual ~SchedulerPipeline() = default;
    virtual BatchInputType input_type() const = 0;
    virtual bool requires_kv_cache() const = 0;
    // Allocate KV cache for sequence, implemented by each pipeline
    virtual bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                                   Sequence* sequence) = 0;
  };

  class LlmRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::SEQUENCE;
    }
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class OneRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC;
    }
    bool requires_kv_cache() const override { return false; }
    bool allocate_kv_cache(KVCacheManager* /*kv_cache_manager*/,
                           Sequence* /*sequence*/) override {
      return true;
    }
  };

  class OneRecXAttentionSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC_XATTENTION;
    }
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class RecMultiRoundSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::REC_MULTI_ROUND;
    }
    bool requires_kv_cache() const override { return false; }
    bool allocate_kv_cache(KVCacheManager* /*kv_cache_manager*/,
                           Sequence* /*sequence*/) override {
      return true;  // RecMultiRound mode does not need KV cache allocation
    }
  };

  // Factory method to create scheduler pipeline
  static std::unique_ptr<SchedulerPipeline> create_scheduler_pipeline(
      RecType rec_type,
      bool is_rec_multi_round);

  RecScheduleResult schedule_request(const absl::Duration& timeout);

  void execute_batch(RecScheduleResult result);

  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  // Lazy-initialized pipeline
  std::unique_ptr<SchedulerPipeline> scheduler_pipeline_;
  std::unique_ptr<RecBatchFactory> rec_batch_factory_;

  // Holds a request consumed by the blocking wait in schedule_request() while
  // the queue was empty. prepare_rec_batch() drains it first, through the same
  // path as request_queue_, so the blocking wait does not lose requests.
  std::shared_ptr<Request> prefetched_request_;

  // Scheduler thread pool for parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  // Semaphore to control concurrent execution of step()
  std::counting_semaphore<10000> step_semaphore_;
};

}  // namespace xllm

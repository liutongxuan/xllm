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

#include "core/scheduler/rec_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>

#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/framework/batch/rec_batch.h"
#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/runtime/xservice_client.h"
#include "core/util/rec_model_utils.h"
#include "core/util/timer.h"

namespace xllm {

namespace {

size_t checked_queue_capacity(int32_t capacity) {
  CHECK_GT(capacity, 0);
  return static_cast<size_t>(capacity);
}

std::ptrdiff_t checked_concurrency(int32_t concurrency) {
  CHECK_GT(concurrency, 0);
  CHECK_LE(concurrency, 10000);
  return static_cast<std::ptrdiff_t>(concurrency);
}

}  // namespace

RecScheduler::RecScheduler(EngineResources resources,
                           RecStep rec_step,
                           Options options)
    : options_(std::move(options)),
      resources_(std::move(resources)),
      rec_step_(std::move(rec_step)),
      kv_cache_manager_(resources_.block_manager_pool()),
      request_queue_(checked_queue_capacity(options_.request_queue_size())),
      step_semaphore_(
          checked_concurrency(options_.rec_worker_max_concurrency())) {
  CHECK(kv_cache_manager_ != nullptr);
  CHECK(resources_.tokenizer() != nullptr);
  CHECK_GT(options_.dp_size(), 0);
  CHECK_GT(options_.max_tokens_per_batch(), 0);
  CHECK_GT(options_.max_seqs_per_batch(), 0);
  request_admission_ = std::make_unique<SchedulerRequestAdmission>(
      kv_cache_manager_,
      request_queue_,
      [this](std::shared_ptr<Request> request) {
        CHECK(request_queue_.write(std::move(request)))
            << "Reserved Rec request queue slot disappeared before prefetch "
               "completed";
      });
  enable_prefix_cache_ = KVCacheConfig::get_instance().enable_prefix_cache();
  cancel_request_queue_ = std::make_shared<CancelRequestQueue>();
  response_processor_ = std::make_unique<AsyncResponseProcessor>(
      resources_.tokenizer(),
      options_.instance_role(),
      options_.enable_service_routing(),
      options_.disable_log_stats(),
      [cancel_request_queue =
           cancel_request_queue_](std::shared_ptr<Request> request) {
        cancel_request_queue->submit(std::move(request));
      });
  if (options_.priority_strategy() == "fcfs" ||
      options_.priority_strategy() == "multi_slo_and_prio") {
    prefill_queue_ = std::make_unique<DequeQueue>();
  } else {
    prefill_queue_ = std::make_unique<HeapQueue>(
        create_comparator(options_.priority_strategy(), /*is_decode=*/false));
  }
  if (options_.enable_service_routing()) {
    XServiceClient* xservice_client = XServiceClient::get_instance();
    CHECK(xservice_client->initialize_done())
        << "XServiceClient not initialized";
    xservice_client->set_scheduler(this);
  }
  instance_info_.name = options_.instance_name().value_or("");
  instance_info_.type =
      options_.instance_role().value_or(InstanceRole::DEFAULT).to_string();
  instance_info_.dp_size = options_.dp_size();
  instance_info_.kv_split_size =
      ParallelConfig::get_instance().kv_split_size_effective();
  if (options_.rec_worker_max_concurrency() > 1) {
    step_threadpool_ = std::make_unique<ThreadPool>(
        /*num_threads=*/static_cast<size_t>(
            options_.rec_worker_max_concurrency()),
        /*cpu_binding=*/false,
        /*pool_name=*/"RecScheduler.step");
  }
}

RecScheduler::~RecScheduler() {
  // Tasks release the semaphore and use scheduler resources after execution.
  step_threadpool_.reset();
  response_processor_->wait_completion();
  request_admission_->shutdown();
}

bool RecScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  if (request->state().rec_type == RecType::kNone) {
    return false;
  }
  return request_admission_->add_request(request);
}

uint32_t RecScheduler::get_waiting_requests_num() const {
  const size_t queued_requests =
      static_cast<size_t>(std::max<ssize_t>(request_queue_.size(), 0));
  return static_cast<uint32_t>(
      waiting_requests_.load(std::memory_order_relaxed) + queued_requests +
      request_admission_->num_prefetching_requests());
}

void RecScheduler::apply_cancel_requests() {
  for (const std::shared_ptr<Request>& request :
       cancel_request_queue_->take_all()) {
    request->set_cancel();
  }
}

void RecScheduler::handle_prefill_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // Admit one fixed execution window, reserving every scheduled sequence's
  // complete KV capacity before handing ownership to the Rec engine.
  const bool requires_kv_cache =
      scheduler_pipeline_ && scheduler_pipeline_->requires_kv_cache();
  while (!prefill_queue_->empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 &&
         kv_cache_manager_->kv_cache_utilization() <
             ::xllm::SchedulerConfig::get_instance()
                 .prefill_scheduling_memory_usage_threshold()) {
    std::shared_ptr<Request> request(prefill_queue_->top());
    if (request->finished() || request->cancelled()) {
      if (requires_kv_cache) {
        kv_cache_manager_->deallocate(request.get());
      }
      //  release the ownership of the request
      finished_requests.emplace_back(std::move(request));
      // remove the request from the priority queue
      prefill_queue_->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1)
          << "Waiting request should have only one sequence.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      if (!requires_kv_cache && prefill_sequence->dp_rank() < 0) {
        prefill_sequence->set_dp_rank(0);
      }

      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        break;
      }

      if (requires_kv_cache) {
        if (!scheduler_pipeline_->allocate_kv_cache(kv_cache_manager_,
                                                    prefill_sequence.get())) {
          can_schedule = false;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        if (requires_kv_cache) {
          kv_cache_manager_->deallocate(seq);
        }
      }
      break;
    }

    if (prefill_sequences.empty()) {
      prefill_queue_->pop_top();
      finished_requests.emplace_back(std::move(request));
      continue;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    prefill_queue_->pop_top();
    running_requests_.emplace_back(std::move(request));
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
  }

  if (running_sequences_.empty() && !prefill_queue_->empty() &&
      remaining_seq_budget > 0) {
    LOG(ERROR)
        << "Request prompt is too long, no enough budget/memory to schedule "
           "a single sequence.";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(prefill_queue_->top());
    prefill_queue_->pop_top();
    if (requires_kv_cache) {
      kv_cache_manager_->deallocate(request.get());
    }
    response_processor_->process_failed_request(
        std::move(request),
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough budget to schedule single sequence."});
  }
}

RecBatchGroup RecScheduler::prepare_rec_batch() {
  Timer timer;
  apply_cancel_requests();
  request_admission_->drain();
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();
  // propagate new requests to prefill_queue_
  // Include those requests that are preempted by others.
  auto propagate_request = [this](std::shared_ptr<Request>& request) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    // Restored/cache-hit requests still need a Rec execution window. Their
    // compute budget and existing KV state are handled during admission.
    prefill_queue_->push(std::move(request));
  };

  // Drain the request prefetched by the blocking wait in schedule_request().
  if (prefetched_request_) {
    propagate_request(prefetched_request_);
    prefetched_request_.reset();
  }

  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
  while (request_queue_.read(request)) {
    propagate_request(request);
  }

  // Select the Rec pipeline once from the first admitted request.
  if (!scheduler_pipeline_ && !prefill_queue_->empty()) {
    const RecType rec_type = prefill_queue_->top()->state().rec_type;
    const bool is_rec_multi_round =
        (rec_type == RecType::kLlmRec) && is_rec_multi_round_mode();
    scheduler_pipeline_ =
        create_scheduler_pipeline(rec_type, is_rec_multi_round);
    rec_batch_factory_ = std::make_unique<RecBatchFactory>(
        options_.dp_size(), scheduler_pipeline_->input_type());
  }

  std::vector<std::shared_ptr<Request>> finished_requests;
  finished_requests.reserve(
      std::min(prefill_queue_->size(),
               static_cast<size_t>(options_.max_seqs_per_batch())));

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);

  handle_prefill_requests(
      remaining_token_budget, remaining_seq_budget, finished_requests);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  RecBatchGroup batches;
  if (rec_batch_factory_) {
    batches = rec_batch_factory_->create_batches(
        running_requests_,
        running_sequences_,
        running_sequences_budgets_,
        kv_cache_manager_->get_swap_block_transfer_infos());
  } else {
    // No pipeline has been selected before the first request arrives.
    CHECK(running_requests_.empty());
    CHECK(running_sequences_.empty());
    batches = RecBatchGroup(static_cast<size_t>(options_.dp_size()),
                            BatchInputType::SEQUENCE);
  }

  // update metrics before returning
  if (std::any_of(batches.begin(), batches.end(), [](const RecBatch& batch) {
        return !batch.empty();
      })) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
    kv_cache_manager_->transfer_blocks(batches);
  } else {
    kv_cache_manager_->transfer_blocks();
  }

  waiting_requests_.store(prefill_queue_->size(), std::memory_order_relaxed);
  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests, get_waiting_requests_num());

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            kv_cache_manager_->num_blocks_in_prefix_cache().size());
  GAUGE_SET(num_free_blocks, kv_cache_manager_->num_free_blocks().size());
  GAUGE_SET(num_used_blocks, kv_cache_manager_->num_used_blocks().size());

  return batches;
}

RecScheduleResult RecScheduler::schedule_request(
    const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  RecScheduleResult result;
  while (true) {
    result.batches = prepare_rec_batch();
    bool all_empty = std::all_of(
        result.batches.begin(),
        result.batches.end(),
        [](const RecBatch& one_batch) { return one_batch.empty(); });
    if (!all_empty) {
      // Move running_requests_ and running_sequences_ into result
      result.requests = std::move(running_requests_);
      result.sequences = std::move(running_sequences_);
      return result;
    }
    const auto now = absl::Now();
    if (now >= deadline) {
      break;
    }
    // Event-driven wait instead of fixed-interval busy polling: block on the
    // request queue until a new request arrives or the deadline is reached.
    // This wakes up immediately on arrival, avoiding the extra latency and CPU
    // spinning of a fixed sleep under high concurrency. The prefetched request
    // is consumed by the next prepare_rec_batch() call.
    std::shared_ptr<Request> request;
    const auto remaining = absl::ToChronoNanoseconds(deadline - now);
    const auto wait_duration =
        request_admission_->num_prefetching_requests() > 0
            ? std::min(remaining, std::chrono::nanoseconds(50'000'000))
            : remaining;
    const auto wait_deadline = std::chrono::steady_clock::now() + wait_duration;
    if (request_queue_.tryReadUntil(wait_deadline, request)) {
      prefetched_request_ = std::move(request);
      waiting_requests_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  // return empty result
  return result;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void RecScheduler::step(const absl::Duration& timeout) {
  if (!options_.enable_schedule_overlap()) {
    // get a new batch of requests
    RecScheduleResult result = schedule_request(timeout);
    bool all_empty = std::all_of(
        result.batches.begin(),
        result.batches.end(),
        [](const RecBatch& one_batch) { return one_batch.empty(); });
    if (all_empty) {
      return;
    }

    // Submit task to thread pool for asynchronous execution
    // After rec_step_() completes, process finished/cancelled requests
    auto function = [this, result = std::move(result)]() mutable {
      execute_batch(std::move(result));

      if (options_.rec_worker_max_concurrency() > 1) {
        step_semaphore_.release();
      }
    };

    if (options_.rec_worker_max_concurrency() > 1) {
      step_semaphore_.acquire();
      step_threadpool_->schedule(std::move(function));
    } else {
      function();
    }

    // Return immediately to allow the next step() call to execute in parallel
  } else {
    LOG(ERROR) << "RecScheduler::step() not supported with "
                  "enable_schedule_overlap";
  }
}

void RecScheduler::generate() {
  bool batch_empty = false;
  while (num_pending_requests() > 0 || !batch_empty ||
         get_waiting_requests_num() > 0) {
    RecScheduleResult result = schedule_request(absl::Milliseconds(50));
    batch_empty =
        std::all_of(result.batches.begin(),
                    result.batches.end(),
                    [](const RecBatch& batch) { return batch.empty(); });
    if (batch_empty) {
      continue;
    }
    execute_batch(std::move(result));
  }
  response_processor_->wait_completion();
}

void RecScheduler::execute_batch(RecScheduleResult result) {
  rec_step_(result.batches);

  std::vector<std::shared_ptr<Request>> finished_requests;
  finished_requests.reserve(result.requests.size());
  for (const std::shared_ptr<Request>& request : result.requests) {
    if (request == nullptr) {
      continue;
    }
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      finished_requests.emplace_back(request);
    }
  }
  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }
}

// Pipeline implementations
bool RecScheduler::LlmRecSchedulerPipeline::allocate_kv_cache(
    KVCacheManager* kv_cache_manager,
    Sequence* sequence) {
  const size_t num_tokens = sequence->num_tokens();
  const size_t max_generated_tokens =
      sequence->stopping_checker()->get_max_generated_tokens();
  // Overflow check to prevent undersized KV cache allocation
  if (std::numeric_limits<size_t>::max() - num_tokens < max_generated_tokens) {
    LOG(ERROR) << "Integer overflow detected in KV cache allocation";
    return false;
  }
  return kv_cache_manager->allocate(sequence,
                                    num_tokens + max_generated_tokens);
}

bool RecScheduler::OneRecXAttentionSchedulerPipeline::allocate_kv_cache(
    KVCacheManager* kv_cache_manager,
    Sequence* sequence) {
  const size_t num_tokens = sequence->num_tokens();
  size_t max_generated_tokens =
      ::xllm::RecConfig::get_instance().max_decode_rounds() > 0
          ? static_cast<size_t>(
                ::xllm::RecConfig::get_instance().max_decode_rounds())
          : kRecDecodeSteps;
  if (const auto* stopping_checker = sequence->stopping_checker()) {
    max_generated_tokens = std::max(
        max_generated_tokens, stopping_checker->get_max_generated_tokens());
  }
  if (std::numeric_limits<size_t>::max() - num_tokens < max_generated_tokens) {
    LOG(ERROR) << "Integer overflow detected in OneRec xattention KV cache "
                  "allocation";
    return false;
  }
  return kv_cache_manager->allocate(sequence,
                                    num_tokens + max_generated_tokens);
}

std::unique_ptr<RecScheduler::SchedulerPipeline>
RecScheduler::create_scheduler_pipeline(RecType rec_type,
                                        bool is_rec_multi_round) {
  if (is_rec_multi_round) {
    return std::make_unique<RecMultiRoundSchedulerPipeline>();
  }
  if (rec_type == RecType::kOneRec && is_onerec_xattention_mode()) {
    return std::make_unique<OneRecXAttentionSchedulerPipeline>();
  }
  if (rec_type == RecType::kLlmRec) {
    return std::make_unique<LlmRecSchedulerPipeline>();
  }
  return std::make_unique<OneRecSchedulerPipeline>();
}

}  // namespace xllm

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

#include "core/scheduler/request_output_dispatcher.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/metrics.h"
#include "core/framework/config/service_config.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/blocking_counter.h"

namespace xllm {

namespace {

struct StreamOutputWork {
  bool has_open_sequences = false;
  std::vector<size_t> indexes;
  std::vector<size_t> num_tokens;
};

StreamOutputWork prepare_stream_output(Request& request) {
  StreamOutputWork work;
  for (size_t i = 0; i < request.sequences().size(); ++i) {
    auto& sequence = request.sequences()[i];
    if (sequence->is_closed()) {
      continue;
    }
    work.has_open_sequences = true;

    if (sequence->has_new_tokens_generated() || sequence->finished()) {
      work.indexes.push_back(i);
      work.num_tokens.push_back(sequence->num_tokens());
    }

    if (sequence->finished()) {
      sequence->close();
    }
  }
  return work;
}

}  // namespace

RequestOutputDispatcher::RequestOutputDispatcher(
    const Tokenizer* tokenizer,
    bool enable_service_routing,
    bool disable_log_stats,
    std::function<void(std::shared_ptr<Request>)> cancel_request)
    : tokenizer_(tokenizer->clone()),
      response_threadpool_(
          /*num_threads=*/::xllm::ServiceConfig::get_instance()
              .num_response_handling_threads(),
          [response_tokenizer = tokenizer_.get()]() {
            response_tokenizer->warmup();
          },
          /*cpu_binding=*/false,
          /*pool_name=*/"RequestOutputDispatcher.response"),
      rpc_threadpool_(/*num_threads=*/1,
                      /*cpu_binding=*/false,
                      /*pool_name=*/"RequestOutputDispatcher.rpc"),
      generate_output_threadpool_(
          /*num_threads=*/16,
          [output_tokenizer = tokenizer_.get()]() {
            output_tokenizer->warmup();
          },
          /*cpu_binding=*/false,
          /*pool_name=*/"RequestOutputDispatcher.generate_output"),
      enable_batch_response_(enable_service_routing),
      disable_log_stats_(disable_log_stats),
      cancel_request_(std::move(cancel_request)) {}

RequestOutputDispatcher::~RequestOutputDispatcher() { wait_completion(); }

void RequestOutputDispatcher::schedule_response(
    const std::shared_ptr<Request>& request,
    ThreadPool::Runnable runnable) {
  const int32_t response_thread_id = request->state().response_thread_id;
  if (response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(std::move(runnable));
  } else {
    response_threadpool_.schedule_with_tid(
        std::move(runnable), static_cast<size_t>(response_thread_id));
  }
}

void RequestOutputDispatcher::process_failed_request(
    std::shared_ptr<Request> request,
    Status status) {
  // schedule the response handling
  const bool disable_log_stats = disable_log_stats_;
  auto runnable = [disable_log_stats, request = request, status = status]() {
    if (!disable_log_stats) {
      request->log_error_statistic(status);
    }
    RequestOutput output;
    output.request_id = request->request_id();
    output.service_request_id = request->service_request_id();
    output.target_xservice_addr = request->source_xservice_addr();
    output.status = status;
    request->state().output_func(output);
  };
  schedule_response(request, std::move(runnable));
}

void RequestOutputDispatcher::process_completed_request(
    std::shared_ptr<Request> request) {
  // schedule the response handling

  // NOTE: Maybe refactor later.
  // For prefill instance in 'ENABLE_SERVICE_DISPATCH_REQUEST' scenario,
  // Currently, the xLLM service uses the BRPC HTTP interface to forward
  // requests. When the prefill process returns the first token to the xLLM
  // service, the result is only returned after the OutputFunc callback function
  // is destructed. Therefore, to ensure the TTFT, the Output callback function
  // needs to be recycled once its execution is complete, allowing the first
  // token to be returned immediately. Additionally, the processing here runs in
  // parallel with the subsequent step where the prefill sends the first token
  // to the decode instance.
  //  When this completes, it destructs the request, which in turn causes the
  //  Output object
  // to be destructed. To prevent the scenario where the request is recycled
  // before the response is fully returned to the xLLM service, the Output
  // object must be detached to avoid premature destruction.
  auto runnable = [this, request = request]() mutable {
    AUTO_COUNTER(responsing_latency_seconds_non_stream);

    double end_2_end_latency_seconds = request->end_to_end_latency_seconds();
    // update the metrics for the request
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
    RequestOutput req_output =
        request->generate_output(*tokenizer_, &generate_output_threadpool_);
    if (!disable_log_stats_) {
      if (req_output.status.has_value() && !req_output.status->ok()) {
        request->log_error_statistic(req_output.status.value());
      } else {
        request->log_statistic(end_2_end_latency_seconds);
      }
    }
    request->state().output_func(req_output);
  };
  schedule_response(request, std::move(runnable));
}

void RequestOutputDispatcher::batch_process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (size_t i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    auto runnable = [counter,
                     this,
                     request = request,
                     request_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_non_stream);
      double end_2_end_latency_seconds = request->end_to_end_latency_seconds();
      // update the metrics for the request
      HISTOGRAM_OBSERVE(
          end_2_end_latency_milliseconds,
          static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
      *request_output = std::move(request->generate_output(*tokenizer_));
      if (!disable_log_stats_) {
        if (request_output->status.has_value() &&
            !request_output->status->ok()) {
          request->log_error_statistic(request_output->status.value());
        } else if (request->finished() || request->cancelled()) {
          request->log_statistic(end_2_end_latency_seconds);
        }
      }
      if (request->sequences()[0]->num_generated_tokens() == 1) {
        // currently only support one sequence when enable_service_routing
        request_output->finished_on_prefill_instance = true;
      }
      counter->decrement_count();
    };
    schedule_response(request, std::move(runnable));
  }

  rpc_threadpool_.schedule(
      [counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs)]() mutable {
        counter->wait();
        auto& resp_callback = requests[0]->state().outputs_func;
        resp_callback(request_outputs);
      });
}

// process non-stream requests
void RequestOutputDispatcher::process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  if (!enable_batch_response_) {
    for (size_t i = 0; i < requests.size(); ++i) {
      process_completed_request(std::move(requests[i]));
    }
  } else {
    // send request response to xllm service in batch through rpc
    batch_process_completed_requests(requests);
  }
}

void RequestOutputDispatcher::process_stream_request(
    std::shared_ptr<Request> request) {
  CHECK(request->state().stream) << "request is not a streaming request";

  StreamOutputWork work = prepare_stream_output(*request);
  if (work.indexes.empty() && !work.has_open_sequences) {
    return;
  }

  // output the delta text til the end of the sequence to the client

  auto runnable = [cancel_request = cancel_request_,
                   request,
                   this,
                   indexes = std::move(work.indexes),
                   num_tokens = std::move(work.num_tokens)]() {
    AUTO_COUNTER(responsing_latency_seconds_stream);

    RequestOutput req_output;
    req_output.request_id = request->request_id();
    for (size_t i = 0; i < indexes.size(); ++i) {
      const size_t index = indexes[i];
      const size_t size = num_tokens[i];
      auto& seq = request->sequences()[index];
      auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
      if (seq_output.has_value()) {
        req_output.outputs.push_back(std::move(seq_output.value()));
      }
    }
    if (!request->state().output_func(req_output)) {
      cancel_request(request);
    }
  };
  schedule_response(request, std::move(runnable));
}

void RequestOutputDispatcher::batch_process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  std::vector<StreamOutputWork> output_work;
  output_work.reserve(requests.size());
  for (const auto& request : requests) {
    CHECK(request->state().stream) << "request is not a streaming request";
    output_work.push_back(prepare_stream_output(*request));
  }

  const size_t requests_size = requests.size();
  auto counter = new BlockingCounter(static_cast<int>(requests_size));
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (size_t i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    StreamOutputWork& work = output_work[i];
    if (work.indexes.empty()) {
      request_outputs[i].request_id = request->request_id();
      request_outputs[i].service_request_id = request->service_request_id();
      request_outputs[i].target_xservice_addr = request->source_xservice_addr();
      counter->decrement_count();
      continue;
    }

    // output the delta text til the end of the sequence to the client
    auto runnable = [this,
                     counter,
                     request,
                     indexes = std::move(work.indexes),
                     num_tokens = std::move(work.num_tokens),
                     req_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_stream);
      const absl::Time response_start_time = absl::Now();

      // RequestOutput req_output;
      req_output->request_id = request->request_id();
      req_output->service_request_id = request->service_request_id();
      req_output->target_xservice_addr = request->source_xservice_addr();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];

        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output->outputs.push_back(std::move(seq_output.value()));
        }
        if (seq->num_generated_tokens() == 1) {
          // currently only support one sequence when enable_service_routing
          // IMPROVE LATER: support enable_schedule_overlap in Default mode
          // for stream request
          req_output->finished_on_prefill_instance = true;
        }
      }
      if (req_output->finished_on_prefill_instance) {
        VLOG(1) << "Prefill response generation request_id="
                << request->request_id() << ", response_thread_id="
                << request->state().response_thread_id << ", total_ms="
                << absl::ToDoubleMilliseconds(absl::Now() -
                                              response_start_time);
      }
      counter->decrement_count();
    };
    schedule_response(request, std::move(runnable));
  }

  rpc_threadpool_.schedule([cancel_request = cancel_request_,
                            counter = std::unique_ptr<BlockingCounter>(counter),
                            requests = std::move(requests),
                            request_outputs =
                                std::move(request_outputs)]() mutable {
    auto& resp_callback = requests[0]->state().outputs_func;
    const absl::Time wait_start_time = absl::Now();
    counter->wait();
    const double wait_ms =
        absl::ToDoubleMilliseconds(absl::Now() - wait_start_time);
    const absl::Time rpc_start_time = absl::Now();
    std::vector<bool> status_set = resp_callback(request_outputs);
    CHECK_EQ(status_set.size(), requests.size());
    if (!request_outputs.empty() &&
        request_outputs[0].finished_on_prefill_instance) {
      VLOG(1) << "Prefill response RPC request_id=" << requests[0]->request_id()
              << ", response_wait_ms=" << wait_ms << ", rpc_ms="
              << absl::ToDoubleMilliseconds(absl::Now() - rpc_start_time);
    }
    for (size_t i = 0; i < requests.size(); ++i) {
      if (!status_set[i]) {
        cancel_request(requests[i]);
      }
    }
  });
}

// process stream requests
void RequestOutputDispatcher::process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  if (!enable_batch_response_) {
    for (auto& req : requests) {
      process_stream_request(req);
    }
  } else {
    // send request response to xllm service in batch through rpc
    batch_process_stream_requests(requests);
  }
}

// for batch generate, wait all response done.
void RequestOutputDispatcher::wait_completion() {
  auto wait_threadpool = [](ThreadPool& threadpool) {
    size_t thread_num = threadpool.size();
    // Add a task to each thread, and when all tasks are completed, it indicates
    // that all previously scheduled tasks in the thread pool have finished.
    BlockingCounter counter(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto runnable = [&counter]() mutable { counter.decrement_count(); };
      threadpool.schedule_with_tid(std::move(runnable), i);
    }
    counter.wait();
  };
  wait_threadpool(response_threadpool_);
  wait_threadpool(rpc_threadpool_);
}

}  // namespace xllm

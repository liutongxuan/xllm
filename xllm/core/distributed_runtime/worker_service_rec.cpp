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

#include <brpc/closure_guard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/distributed_runtime/worker_service.h"
#include "core/framework/speculative/metrics.h"
#include "core/util/tensor_helper.h"
#include "runtime/rec_forward_params.h"
#include "util/timer.h"

namespace xllm {
namespace {

std::vector<SpeculativeTokenStats> record_rec_speculative_metrics(
    const ForwardOutput& output,
    const torch::Tensor& cpu_next_tokens,
    bool speculative_decode) {
  if (output.is_graph_warmup || !speculative_decode) {
    return {};
  }
  if (output.spec_verify_layouts.empty()) {
    CHECK(!cpu_next_tokens.defined() || cpu_next_tokens.dim() != 2)
        << "speculative verify output is missing row metadata";
    return {};
  }
  // Reuse the completed output copy instead of issuing another D2H for metrics.
  SpecMetrics metrics =
      make_spec_metrics(cpu_next_tokens, output.spec_verify_layouts);
  record_spec_metrics(metrics);
  return std::move(metrics.sequence_stats);
}

}  // namespace

void WorkerService::step_rec_model(
    RecForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  CHECK(!options_.enable_task_pipeline())
      << "REC does not support task pipeline";
  CHECK(!options_.enable_schedule_overlap())
      << "REC does not support schedule overlap";
  // RecWorkerImpl captures input by reference. Wait before releasing this
  // owner.
  auto result = std::move(worker_->step_async(input)).get();
  if (!result.has_value()) {
    return;
  }
  const ForwardOutput& output = *result;
  cpu_output.prepared_token = output.prepared_token;
  cpu_output.json_object_errors = output.json_object_errors;
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  auto copy_output_to_host = [&]() {
    // Every D2H, including expert data, runs under the selected output stream.
    cpu_output.expert_load_data =
        safe_to(output.expert_load_data, torch::kCPU, /*non_blocking=*/true);
    cpu_output.sample_output.embeddings =
        safe_to(sample.embeddings,
                torch::dtype(torch::kFloat32).device(torch::kCPU),
                /*non_blocking=*/true);
    auto& mm_embeddings = cpu_output.sample_output.mm_embeddings;
    mm_embeddings.clear();
    mm_embeddings.reserve(sample.mm_embeddings.size());
    for (const auto& sequence : sample.mm_embeddings) {
      std::vector<torch::Tensor> cpu_sequence;
      cpu_sequence.reserve(sequence.size());
      for (const torch::Tensor& embedding : sequence) {
        cpu_sequence.emplace_back(
            safe_to(embedding, torch::kCPU, /*non_blocking=*/true));
      }
      mm_embeddings.emplace_back(std::move(cpu_sequence));
    }
    cpu_output.sample_output.next_tokens =
        safe_to(output.next_tokens_host.defined() ? output.next_tokens_host
                                                  : sample.next_tokens,
                torch::kCPU,
                /*non_blocking=*/true);
    if (cpu_output.sample_output.next_tokens.defined()) {
      cpu_output.sample_output.logprobs =
          safe_to(sample.logprobs, torch::kCPU, /*non_blocking=*/true);
      // Beam results already carry the final tokens and logprobs.
      if (!beam.src_seq_idxes.defined()) {
        cpu_output.sample_output.top_tokens =
            safe_to(sample.top_tokens, torch::kCPU, /*non_blocking=*/true);
        cpu_output.sample_output.top_logprobs =
            safe_to(sample.top_logprobs, torch::kCPU, /*non_blocking=*/true);
      }
    }
    // REC multi-round results can contain beams without sample next_tokens.
    cpu_output.beam_search_output.src_seq_idxes =
        safe_to(beam.src_seq_idxes, torch::kCPU, /*non_blocking=*/true);
    if (beam.src_seq_idxes.defined()) {
      cpu_output.beam_search_output.out_tokens =
          safe_to(beam.out_tokens, torch::kCPU, /*non_blocking=*/true);
      cpu_output.beam_search_output.out_logprobs =
          safe_to(beam.out_logprobs,
                  torch::dtype(torch::kFloat32).device(torch::kCPU),
                  /*non_blocking=*/true);
    }
  };
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (output.ready_event != nullptr) {
    CHECK(stream_->wait_event(output.ready_event))
        << "wait REC output ready event failed";
  }
  copy_output_to_host();
  stream_->synchronize();
  speculative_token_stats =
      record_rec_speculative_metrics(output,
                                     cpu_output.sample_output.next_tokens,
                                     options_.enable_speculative_decode());
}

void WorkerService::poll_rec_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();

  while (true) {
    RecForwardInput input;
    if (!input_shm_manager_->input_read(input, device_)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_rec_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->rec_output_write(
        cpu_output, speculative_token_stats);
    CHECK(write_ok) << "rec worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteRECModel(
    ::google::protobuf::RpcController* controller,
    const proto::RECForwardRequest* request,
    proto::RECForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "rec") {
      controller->SetFailed("REC RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("REC requires a packed input payload");
      return;
    }
    RecForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid REC input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_rec_model(input, cpu_output, speculative_token_stats);
    if (!encode_rec_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize REC model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

}  // namespace xllm

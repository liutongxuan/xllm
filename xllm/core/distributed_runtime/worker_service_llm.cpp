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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/distributed_runtime/worker_service.h"
#include "core/framework/speculative/metrics.h"
#include "core/util/tensor_helper.h"
#include "runtime/forward_params.h"
#include "util/timer.h"

namespace xllm {
namespace {

std::vector<SpeculativeTokenStats> record_llm_speculative_metrics(
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

int32_t get_llm_decode_sequence_count(const LlmForwardInput& input) {
  if (input.sampling_params.sample_idxes.defined()) {
    return static_cast<int32_t>(input.sampling_params.sample_idxes.size(0));
  }
  if (input.input_host_sample_count != kUnknownPackedSampleCount) {
    return input.input_host_sample_count;
  }
  if (!input.runtime.input_host_buffer_has_layout) {
    return 0;
  }
  LlmForwardInput unpacked_input;
  const bool unpacked = detail::unpack_from_input_host_buffer(
      input,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      unpacked_input,
      /*materialize_device_buffer=*/false);
  if (!unpacked || !unpacked_input.sampling_params.sample_idxes.defined()) {
    return 0;
  }
  return static_cast<int32_t>(
      unpacked_input.sampling_params.sample_idxes.size(0));
}

torch::Tensor clone_llm_cpu_view(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU tensor view";
  return tensor.contiguous().clone();
}

void stabilize_llm_host_views(LlmForwardInput& input) {
  input.token_ids_host = clone_llm_cpu_view(input.token_ids_host);
  input.positions_host = clone_llm_cpu_view(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_llm_cpu_view(input.input_params.attention.host.block_tables);
}

bool has_cpu_llm_output(const ForwardOutput& output) {
  const auto on_cpu = [](const torch::Tensor& tensor) {
    return !tensor.defined() || tensor.device().is_cpu();
  };
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  const bool cpu_tensors =
      on_cpu(output.next_tokens_host.defined() ? output.next_tokens_host
                                               : sample.next_tokens) &&
      on_cpu(output.expert_load_data) && on_cpu(sample.embeddings) &&
      on_cpu(sample.logprobs) && on_cpu(sample.top_tokens) &&
      on_cpu(sample.top_logprobs) && on_cpu(beam.src_seq_idxes) &&
      on_cpu(beam.out_tokens) && on_cpu(beam.out_logprobs);
  return cpu_tensors &&
         std::all_of(sample.mm_embeddings.begin(),
                     sample.mm_embeddings.end(),
                     [&](const auto& sequence) {
                       return std::all_of(
                           sequence.begin(), sequence.end(), on_cpu);
                     });
}

}  // namespace

void WorkerService::collect_llm_output(const ForwardOutput& output,
                                       ForwardOutput& cpu_output,
                                       bool cpu_ready_result) {
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
    // Keep optional beam results independent of sample next_tokens.
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
  const bool use_default_stream = !options_.enable_schedule_overlap();
  if (cpu_ready_result) {
    copy_output_to_host();
  } else if (use_default_stream) {
    copy_output_to_host();
    device_.synchronize_default_stream();
  } else {
    c10::StreamGuard stream_guard = stream_->set_stream_guard();
    if (output.ready_event != nullptr) {
      CHECK(stream_->wait_event(output.ready_event))
          << "wait LLM output ready event failed";
    }
    copy_output_to_host();
    stream_->synchronize();
  }
#if defined(USE_NPU)
  if (!use_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }
#endif
}

void WorkerService::step_llm_model(
    LlmForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  const bool task_pipeline = options_.enable_task_pipeline();
  const bool worker_prepare =
      task_pipeline && worker_->task_pipeline_uses_worker_prepare();
  const bool worker_rebuilds_host_views =
      input.runtime.input_host_buffer_has_layout &&
      input.runtime.input_host_buffer.defined() &&
      input.runtime.input_host_buffer.device().is_cpu() &&
      input.runtime.input_host_buffer.is_contiguous() &&
      input.runtime.input_host_buffer.numel() > 0 &&
      !input.runtime.device_tensors_ready;
  if (options_.enable_schedule_overlap() && !worker_rebuilds_host_views &&
      (!task_pipeline || worker_prepare)) {
    // Packed preparation rebuilds views from its retained owner. Device-ready
    // inputs reuse their existing views, which can still borrow shared memory.
    stabilize_llm_host_views(input);
  }
  auto future = worker_->step_async(input);
  if (!options_.enable_schedule_overlap()) {
    auto output = std::move(future).get();
    if (!output.has_value()) {
      return;
    }
    const bool cpu_ready_result =
        task_pipeline && (!worker_prepare || has_cpu_llm_output(*output));
    collect_llm_output(*output, cpu_output, cpu_ready_result);
    speculative_token_stats =
        record_llm_speculative_metrics(*output,
                                       cpu_output.sample_output.next_tokens,
                                       options_.enable_speculative_decode());
    return;
  }
  if (task_pipeline) {
    // Do not publish placeholders until the Worker accepts and owns the input.
    auto ack = std::move(future).get();
    CHECK(!ack.has_value()) << "LLM task Step must return PrepareAck";
  }
  const auto int_options = torch::TensorOptions().device(torch::kCPU);
  if (worker_->is_driver()) {
    const int32_t sequence_count = get_llm_decode_sequence_count(input);
    cpu_output.sample_output.next_tokens = torch::arange(
        -1, -(sequence_count + 1), -1, int_options.dtype(torch::kInt32));
    if (!task_pipeline) {
      std::move(future).deferValue([](auto&&) {});
    }
  }
  cpu_output.expert_load_data =
      torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
}

void WorkerService::poll_llm_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();
  // Graph updates and task preparation must stay on the ordered worker stream.
  const InputDeviceMaterializationPolicy policy =
      options_.enable_task_pipeline() ||
              (options_.enable_schedule_overlap() && options_.enable_graph())
          ? InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE
          : InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ;
  while (true) {
    LlmForwardInput input;
    if (!input_shm_manager_->input_read(input, device_, policy)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_llm_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->llm_output_write(
        cpu_output, speculative_token_stats);
    CHECK(write_ok) << "llm worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteLLMModel(
    ::google::protobuf::RpcController* controller,
    const proto::LLMForwardRequest* request,
    proto::LLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "llm") {
      controller->SetFailed("LLM RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("LLM requires a packed input payload");
      return;
    }
    LlmForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid LLM input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_llm_model(input, cpu_output, speculative_token_stats);
    if (!encode_llm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize LLM model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::GetLastLLMStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* /*request*/,
    proto::LLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "llm" || !options_.enable_schedule_overlap()) {
      controller->SetFailed(
          "LLM last-step results require the matching overlap backend");
      return;
    }
    auto output = std::move(worker_->get_last_step_result_async()).get();
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    if (output.has_value()) {
      const bool task_pipeline = options_.enable_task_pipeline();
      const bool worker_prepare =
          task_pipeline && worker_->task_pipeline_uses_worker_prepare();
      const bool cpu_ready_result =
          task_pipeline && (!worker_prepare || has_cpu_llm_output(*output));
      collect_llm_output(*output, cpu_output, cpu_ready_result);
      speculative_token_stats =
          record_llm_speculative_metrics(*output,
                                         cpu_output.sample_output.next_tokens,
                                         options_.enable_speculative_decode());
    }
    // Embedding-only and beam-only results are valid completed outputs.
    if (!encode_llm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize LLM last-step output");
    }
  });
}

}  // namespace xllm

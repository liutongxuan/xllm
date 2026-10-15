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

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/distributed_runtime/worker_service.h"
#include "core/framework/speculative/metrics.h"
#include "core/util/tensor_helper.h"
#include "runtime/vlm_forward_params.h"
#include "util/timer.h"

namespace xllm {
namespace {

std::vector<SpeculativeTokenStats> record_vlm_speculative_metrics(
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

int32_t get_vlm_decode_sequence_count(const VlmForwardInput& input) {
  if (input.sampling_params.sample_idxes.defined()) {
    return static_cast<int32_t>(input.sampling_params.sample_idxes.size(0));
  }
  if (input.input_host_sample_count != kUnknownPackedSampleCount) {
    return input.input_host_sample_count;
  }
  if (!input.runtime.input_host_buffer_has_layout) {
    return 0;
  }
  VlmForwardInput unpacked_input;
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

torch::Tensor clone_vlm_cpu_view(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU tensor view";
  return tensor.contiguous().clone();
}

void stabilize_vlm_host_views(VlmForwardInput& input) {
  input.token_ids_host = clone_vlm_cpu_view(input.token_ids_host);
  input.positions_host = clone_vlm_cpu_view(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_vlm_cpu_view(input.input_params.attention.host.block_tables);
}

}  // namespace

void WorkerService::collect_vlm_output(const ForwardOutput& output,
                                       ForwardOutput& cpu_output) {
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
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (output.ready_event != nullptr) {
    CHECK(stream_->wait_event(output.ready_event))
        << "wait VLM output ready event failed";
  }
  copy_output_to_host();
  stream_->synchronize();
#if defined(USE_NPU)
  if (options_.enable_schedule_overlap()) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }
#endif
}

void WorkerService::step_vlm_model(
    VlmForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  CHECK(!options_.enable_task_pipeline())
      << "VLM does not support task pipeline";
  if (options_.enable_schedule_overlap()) {
    stabilize_vlm_host_views(input);
  }
  auto future = worker_->step_async(input);
  if (!options_.enable_schedule_overlap()) {
    auto output = std::move(future).get();
    if (output.has_value()) {
      collect_vlm_output(*output, cpu_output);
      speculative_token_stats =
          record_vlm_speculative_metrics(*output,
                                         cpu_output.sample_output.next_tokens,
                                         options_.enable_speculative_decode());
    }
    return;
  }
  const auto int_options = torch::TensorOptions().device(torch::kCPU);
  if (worker_->is_driver()) {
    const int32_t sequence_count = get_vlm_decode_sequence_count(input);
    cpu_output.sample_output.next_tokens = torch::arange(
        -1, -(sequence_count + 1), -1, int_options.dtype(torch::kInt32));
    std::move(future).deferValue([](auto&&) {});
  }
  cpu_output.expert_load_data =
      torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
}

void WorkerService::poll_vlm_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();
  const InputDeviceMaterializationPolicy policy =
      options_.enable_schedule_overlap() && options_.enable_graph()
          ? InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE
          : InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ;
  while (true) {
    VlmForwardInput input;
    if (!input_shm_manager_->input_read(input, device_, policy)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_vlm_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->raw_output_write(
        cpu_output.sample_output.next_tokens,
        cpu_output.sample_output.logprobs,
        cpu_output.sample_output.top_tokens,
        cpu_output.sample_output.top_logprobs,
        cpu_output.sample_output.embeddings,
        cpu_output.sample_output.mm_embeddings,
        speculative_token_stats,
        cpu_output.dit_forward_output.tensors,
        cpu_output.dit_forward_output.text_output,
        cpu_output.expert_load_data,
        cpu_output.prepared_token,
        cpu_output.beam_search_output.src_seq_idxes,
        cpu_output.beam_search_output.out_tokens,
        cpu_output.beam_search_output.out_logprobs,
        cpu_output.json_object_errors);
    CHECK(write_ok) << "vlm worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteVLMModel(
    ::google::protobuf::RpcController* controller,
    const proto::VLMForwardRequest* request,
    proto::VLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "vlm") {
      controller->SetFailed("VLM RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("VLM requires a packed input payload");
      return;
    }
    VlmForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid VLM input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_vlm_model(input, cpu_output, speculative_token_stats);
    if (!encode_vlm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize VLM model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::GetLastVLMStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* /*request*/,
    proto::VLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "vlm" || !options_.enable_schedule_overlap()) {
      controller->SetFailed(
          "VLM last-step results require the matching overlap backend");
      return;
    }
    auto output = std::move(worker_->get_last_step_result_async()).get();
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    if (output.has_value()) {
      collect_vlm_output(*output, cpu_output);
      speculative_token_stats =
          record_vlm_speculative_metrics(*output,
                                         cpu_output.sample_output.next_tokens,
                                         options_.enable_speculative_decode());
    }
    // Embedding-only and beam-only results are valid completed outputs.
    if (!encode_vlm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize VLM last-step output");
    }
  });
}

}  // namespace xllm

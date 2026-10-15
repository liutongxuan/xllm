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
#include "core/util/tensor_helper.h"
#include "runtime/dit_forward_params.h"
#include "util/timer.h"

namespace xllm {
void WorkerService::step_dit_model(const DiTForwardInput& input,
                                   ForwardOutput& cpu_output) {
  CHECK(!options_.enable_task_pipeline())
      << "DiT does not support task pipeline";
  auto result = std::move(worker_->step_async(input)).get();
  if (!result.has_value()) {
    return;
  }
  const auto& output = result->dit_forward_output;
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (result->ready_event != nullptr) {
    CHECK(stream_->wait_event(result->ready_event))
        << "wait DiT output ready event failed";
  }
  auto& tensors = cpu_output.dit_forward_output.tensors;
  tensors.reserve(output.tensors.size());
  for (const torch::Tensor& tensor : output.tensors) {
    tensors.emplace_back(safe_to(tensor, torch::kCPU, /*non_blocking=*/true));
  }
  cpu_output.dit_forward_output.text_output = output.text_output;
  stream_->synchronize();
}

void WorkerService::poll_dit_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();

  while (true) {
    DiTForwardInput input;
    if (!input_shm_manager_->input_read(input)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    step_dit_model(input, cpu_output);
    const bool write_ok =
        output_shm_manager->dit_output_write(cpu_output.dit_forward_output);
    CHECK(write_ok) << "dit worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteDITModel(
    ::google::protobuf::RpcController* controller,
    const proto::DITForwardRequest* request,
    proto::DITForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "dit") {
      controller->SetFailed("DIT RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("DIT requires a packed input payload");
      return;
    }
    DiTForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid DIT input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    step_dit_model(input, cpu_output);
    if (!encode_dit_output(cpu_output, response)) {
      controller->SetFailed("Failed to serialize DIT model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

}  // namespace xllm

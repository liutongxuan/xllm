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

#include "core/distributed_runtime/worker_service.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>

#include "core/runtime/options.h"

namespace xllm {

namespace {

class CountingClosure final : public google::protobuf::Closure {
 public:
  void Run() override {
    if (run_count_.fetch_add(1, std::memory_order_relaxed) == 0) {
      completed_.set_value();
    }
  }

  std::future<void> get_future() { return completed_.get_future(); }

  int32_t run_count() const {
    return run_count_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int32_t> run_count_{0};
  std::promise<void> completed_;
};

template <typename Request, typename Response>
using ModelRpc = void (WorkerService::*)(google::protobuf::RpcController*,
                                         const Request*,
                                         Response*,
                                         google::protobuf::Closure*);

template <typename Request, typename Response>
void expect_rpc_rejection(runtime::Options options,
                          bool initialized,
                          const Request& request,
                          ModelRpc<Request, Response> rpc,
                          const std::string& expected_error) {
  brpc::Controller controller;
  Response response;
  CountingClosure done;
  auto completed = done.get_future();
  // The injected constructor marks the service initialized. A null Worker
  // ensures rejected requests cannot accidentally enter model execution.
  auto service = initialized ? std::make_unique<WorkerService>(
                                   options, torch::Device("npu:0"), nullptr)
                             : std::make_unique<WorkerService>(
                                   options, torch::Device("npu:0"));
  (service.get()->*rpc)(&controller, &request, &response, &done);
  EXPECT_EQ(completed.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  // Destruction drains request tasks before the callback count is inspected.
  service.reset();
  EXPECT_EQ(done.run_count(), 1);
  EXPECT_TRUE(controller.Failed());
  EXPECT_NE(controller.ErrorText().find(expected_error), std::string::npos)
      << controller.ErrorText();
}

void make_packed_inputs(
    std::array<proto::PackedForwardInput, 4>& packed_inputs) {
  LlmForwardInput llm_input;
  VlmForwardInput vlm_input;
  RecForwardInput rec_input;
  DiTForwardInput dit_input;
  ASSERT_TRUE(llm_input.to_packed_proto(&packed_inputs[0]));
  ASSERT_TRUE(vlm_input.to_packed_proto(&packed_inputs[1]));
  ASSERT_TRUE(rec_input.to_packed_proto(&packed_inputs[2]));
  ASSERT_TRUE(dit_input.to_packed_proto(&packed_inputs[3]));
}

template <typename Request, typename Response>
void expect_execute_rejections(const std::string& backend,
                               size_t input_index,
                               ModelRpc<Request, Response> rpc) {
  runtime::Options options;
  options.backend(backend);
  Request request;
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "requires a packed input payload");

  request.mutable_packed_input()->set_payload("invalid");
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "input domain, schema, or layout");

  std::array<proto::PackedForwardInput, 4> packed_inputs;
  ASSERT_NO_FATAL_FAILURE(make_packed_inputs(packed_inputs));
  for (size_t index = 0; index < packed_inputs.size(); ++index) {
    if (index == input_index) {
      continue;
    }
    SCOPED_TRACE(index);
    *request.mutable_packed_input() = packed_inputs[index];
    expect_rpc_rejection(options,
                         /*initialized=*/true,
                         request,
                         rpc,
                         "input domain, schema, or layout");
  }

  expect_rpc_rejection(options,
                       /*initialized=*/false,
                       request,
                       rpc,
                       "Worker is not initialized");
  options.backend(backend == "llm" ? "dit" : "llm");
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "RPC does not match the configured backend");
}

TEST(WorkerServiceModelRpcTest,
     ExecuteLLMRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("llm",
                            /*input_index=*/0,
                            &WorkerService::ExecuteLLMModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteVLMRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("vlm",
                            /*input_index=*/1,
                            &WorkerService::ExecuteVLMModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteRECRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("rec",
                            /*input_index=*/2,
                            &WorkerService::ExecuteRECModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteDITRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("dit",
                            /*input_index=*/3,
                            &WorkerService::ExecuteDITModel);
}

template <typename Response>
void expect_last_step_rejections(const std::string& backend,
                                 ModelRpc<proto::Empty, Response> rpc) {
  runtime::Options options;
  options.backend(backend).enable_schedule_overlap(true);
  proto::Empty request;
  expect_rpc_rejection(options,
                       /*initialized=*/false,
                       request,
                       rpc,
                       "Worker is not initialized");

  options.backend(backend == "llm" ? "vlm" : "llm");
  expect_rpc_rejection(
      options,
      /*initialized=*/true,
      request,
      rpc,
      "last-step results require the matching overlap backend");

  options.backend(backend).enable_schedule_overlap(false);
  expect_rpc_rejection(
      options,
      /*initialized=*/true,
      request,
      rpc,
      "last-step results require the matching overlap backend");
}

TEST(WorkerServiceModelRpcTest, LastLLMStepRequiresInitializedOverlapBackend) {
  expect_last_step_rejections("llm", &WorkerService::GetLastLLMStepResult);
}

TEST(WorkerServiceModelRpcTest, LastVLMStepRequiresInitializedOverlapBackend) {
  expect_last_step_rejections("vlm", &WorkerService::GetLastVLMStepResult);
}

TEST(WorkerServiceShutdownTest, JoinsIdleShmPollingThreadForEveryBackend) {
  for (const std::string& backend : {"llm", "vlm", "rec", "dit"}) {
    SCOPED_TRACE(backend);
    runtime::Options options;
    options.backend(backend);
    auto service =
        std::make_unique<WorkerService>(options, torch::Device("npu:0"));
    const std::string prefix =
        "worker_shutdown_" + std::to_string(getpid()) + "_" + backend;
    bool is_creator = false;
    auto input =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_input",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_INPUT);
    auto output =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_output",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_OUTPUT);
    service->create_polling_shm_thread(std::move(input), std::move(output));
    // No input is ever published: destruction must cancel the read and join,
    // including when shutdown races with the polling thread's first read.
    service.reset();
  }
}

TEST(WorkerServiceShutdownTest, StopsWithoutShmPollingThread) {
  runtime::Options options;
  WorkerService service(options, torch::Device("npu:0"));
}

}  // namespace
}  // namespace xllm

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
#include "core/distributed_runtime/worker_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/stream.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/types.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/kv_transport_codec.h"
#include "core/kv_cache/transfer/prefetch_result.h"

namespace xllm {
namespace {

class WorkerPrefetchSession final
    : public brpc::StreamInputHandler,
      public std::enable_shared_from_this<WorkerPrefetchSession> {
 public:
  WorkerPrefetchSession(Worker* worker,
                        ThreadPool* threadpool,
                        StoragePrefetchRequest request)
      : worker_(worker), threadpool_(threadpool), request_(std::move(request)) {
    CHECK(worker_ != nullptr);
    CHECK(threadpool_ != nullptr);
    CHECK(request_.valid());
  }

  void retain() {
    std::lock_guard<std::mutex> lock(mutex_);
    keepalive_ = shared_from_this();
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    keepalive_.reset();
  }

  void start(brpc::StreamId stream_id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::CREATED) {
        return;
      }
      stream_id_ = stream_id;
      state_ = State::RUNNING_BATCH;
    }
    schedule_batch();
  }

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    if (size != 1 || messages[0]->length() != 1) {
      fail_and_close(id);
      return -1;
    }

    uint8_t control_byte = 0;
    messages[0]->copy_to(&control_byte, sizeof(control_byte));
    const PrefetchControl control = static_cast<PrefetchControl>(control_byte);
    bool run_next = false;
    bool close = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::RUNNING_BATCH && control == PrefetchControl::STOP) {
        stop_after_batch_ = true;
      } else if (state_ != State::WAITING_DECISION) {
        state_ = State::FAILED;
        close = true;
      } else if (control == PrefetchControl::STOP) {
        state_ = State::COMPLETED;
        close = true;
      } else if (control == PrefetchControl::CONTINUE &&
                 last_prefix_hit_units_ ==
                     request_.batch_unit_count(batch_index_) &&
                 batch_index_ + 1 < request_.batch_count()) {
        ++batch_index_;
        state_ = State::RUNNING_BATCH;
        run_next = true;
      } else {
        state_ = State::FAILED;
        close = true;
      }
    }

    if (run_next) {
      schedule_batch();
    } else if (close) {
      brpc::StreamClose(id);
    }
    // A close triggered by anything other than a STOP is a protocol error;
    // a STOP-driven close is the normal completion path.
    const bool protocol_error = close && control != PrefetchControl::STOP;
    return protocol_error ? -1 : 0;
  }

  void on_idle_timeout(brpc::StreamId id) override { fail_and_close(id); }

  void on_failed(brpc::StreamId /*id*/,
                 int /*error_code*/,
                 const std::string& /*error_text*/) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::COMPLETED) {
      state_ = State::FAILED;
    }
  }

  void on_closed(brpc::StreamId /*id*/) override {
    std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = State::CLOSED;
    keepalive_.reset();
  }

 private:
  enum class State : uint8_t {
    CREATED = 0,
    RUNNING_BATCH = 1,
    WAITING_DECISION = 2,
    COMPLETED = 3,
    FAILED = 4,
    CLOSED = 5,
  };

  void schedule_batch() {
    std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
    threadpool_->schedule([self = std::move(self)]() { self->run_batch(); });
  }

  void run_batch() {
    size_t batch_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::RUNNING_BATCH) {
        return;
      }
      batch_index = batch_index_;
    }

    const auto [transfer_begin, transfer_end] =
        request_.batch_transfer_range(batch_index);
    Slice<BlockTransferInfo> all_transfers(request_.transfer_infos);
    Slice<BlockTransferInfo> batch =
        all_transfers.slice(transfer_begin, transfer_end);
    std::vector<uint8_t> logical_hits = worker_->prefetch_kv_blocks(batch);
    const std::optional<uint8_t> prefix_hit_units =
        request_.count_prefix_hit_units(batch_index, logical_hits);
    if (!prefix_hit_units.has_value()) {
      fail_and_close(stream_id_);
      return;
    }

    bool close_after_result = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::RUNNING_BATCH || batch_index != batch_index_) {
        return;
      }
      last_prefix_hit_units_ = *prefix_hit_units;
      close_after_result = stop_after_batch_;
      state_ = close_after_result ? State::COMPLETED : State::WAITING_DECISION;
    }

    butil::IOBuf result;
    result.append(&*prefix_hit_units, sizeof(*prefix_hit_units));
    if (brpc::StreamWrite(stream_id_, result) != 0) {
      fail_and_close(stream_id_);
    } else if (close_after_result) {
      brpc::StreamClose(stream_id_);
    }
  }

  void fail_and_close(brpc::StreamId id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::CLOSED) {
        return;
      }
      state_ = State::FAILED;
    }
    brpc::StreamClose(id);
  }

  Worker* worker_ = nullptr;
  ThreadPool* threadpool_ = nullptr;
  StoragePrefetchRequest request_;
  std::mutex mutex_;
  brpc::StreamId stream_id_ = brpc::INVALID_STREAM_ID;
  size_t batch_index_ = 0;
  size_t last_prefix_hit_units_ = 0;
  bool stop_after_batch_ = false;
  State state_ = State::CREATED;
  std::shared_ptr<WorkerPrefetchSession> keepalive_;
};

}  // namespace

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device)
    : options_(options), initialized_(false), device_(device) {
  device_.set_device();
  device_.init_device_context();
  stream_ = device_.get_stream_from_pool();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/4,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"WorkerService.request");
}

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device,
                             std::unique_ptr<Worker> worker)
    : options_(options),
      initialized_(true),
      device_(device),
      worker_(std::move(worker)) {
  device_.set_device();
  device_.init_device_context();
  stream_ = device_.get_stream_from_pool();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/4,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"WorkerService.request");
}

WorkerService::~WorkerService() {
  // The polling thread may be idle in input_read. Cancel that wait before
  // joining, while the worker and its execution resources are still alive.
  if (input_shm_manager_) {
    input_shm_manager_->stop_input_read();
  }
  if (polling_thread_ && polling_thread_->joinable()) {
    polling_thread_->join();
  }
}

void WorkerService::set_worker(std::unique_ptr<Worker> worker) {
  worker_ = std::move(worker);
  initialized_ = true;
}

void WorkerService::create_polling_shm_thread(
    std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  CHECK(!polling_thread_) << "SHM polling thread has already been created.";
  input_shm_manager_ = std::move(input_shm_manager);
  if (options_.backend() == "llm") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_llm_shm, this, std::move(output_shm_manager));
  } else if (options_.backend() == "vlm") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_vlm_shm, this, std::move(output_shm_manager));
  } else if (options_.backend() == "rec") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_rec_shm, this, std::move(output_shm_manager));
  } else {
    CHECK_EQ(options_.backend(), "dit") << "Unsupported worker backend";
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_dit_shm, this, std::move(output_shm_manager));
  }
}

void WorkerService::Hello(::google::protobuf::RpcController* controller,
                          const proto::Status* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
  return;
}

void WorkerService::InitModel(::google::protobuf::RpcController* controller,
                              const proto::InitModelRequest* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto model_weights_path = request->model_weights_path();
    auto random_seed = request->random_seed();
    auto init_future =
        worker_->init_model_async(model_weights_path,
                                  random_seed,
                                  MasterStatus(request->master_status()));
    bool status = std::move(init_future).get();
    if (!status) {
      response->set_ok(false);
      return;
    }

    response->set_ok(true);
  });
  return;
}

void WorkerService::ProcessGroupTest(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->process_group_test_async();
    std::move(future).get();
    response->set_ok(true);
  });
  return;
}

void WorkerService::ProfileDeviceMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::DeviceMemory* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->estimate_kv_cache_capacity_async();
    std::tuple<int64_t, int64_t> result = std::move(future).get();
    response->set_available_memory(std::get<0>(result));
    response->set_total_memory(std::get<1>(result));
  });
  return;
}

void WorkerService::AllocateKVCache(
    ::google::protobuf::RpcController* controller,
    const proto::AllocateKVCacheRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    const KVCacheShape kv_cache_shape =
        KVCacheShape::from_proto(request->kv_cache_shape());
    auto future = worker_->allocate_kv_cache_async(kv_cache_shape);
    bool status = std::move(future).get();
    response->set_ok(status);
  });
  return;
}

void WorkerService::SetSpeculativeValidateTimePredictor(
    ::google::protobuf::RpcController* controller,
    const proto::SpeculativeValidateTimePredictor* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    SpeculativeProfileRegistry::ValidateTimePredictor predictor;
    predictor.intercept_ms = request->intercept_ms();
    predictor.query_token_ms = request->query_token_ms();
    predictor.query_prefix_ms = request->query_prefix_ms();
    response->set_ok(
        worker_->set_speculative_validate_time_predictor(predictor));
  });
  return;
}

void WorkerService::AllocateKVCacheWithTransfer(
    ::google::protobuf::RpcController* controller,
    const proto::AllocateKVCacheRequest* req,
    proto::Status* resp,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    const KVCacheShape kv_cache_shape =
        KVCacheShape::from_proto(req->kv_cache_shape());
    auto future =
        worker_->allocate_kv_cache_with_transfer_async(kv_cache_shape);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::GetCacheInfo(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::CacheInfo* resp,
                                 ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    uint64_t cluster_id;
    std::string addr;
    uint16_t listen_port;
    worker_->get_cache_info(cluster_id, addr, listen_port);
    resp->set_cluster_id(cluster_id);
    resp->set_addr(addr);
    resp->set_listen_port(listen_port);
  });
  return;
}

void WorkerService::PullKVCache(::google::protobuf::RpcController* controller,
                                const proto::PullKVCacheRequest* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<KVTransferMapping> mappings;
    mappings.reserve(req->mappings_size());
    for (const proto::KVTransferMapping& proto_mapping : req->mappings()) {
      KVTransferMapping mapping;
      mapping.group_id = proto_mapping.group_id();
      mapping.local_ids.assign(proto_mapping.local_ids().begin(),
                               proto_mapping.local_ids().end());
      mapping.remote_ids.assign(proto_mapping.remote_ids().begin(),
                                proto_mapping.remote_ids().end());
      mappings.emplace_back(std::move(mapping));
    }
    auto future =
        worker_->pull_kv_blocks_async(req->cluster_id(), req->addr(), mappings);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::TransferBlocks(
    ::google::protobuf::RpcController* controller,
    const proto::BlockTransferInfos* req,
    proto::TransferStatus* resp,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<BlockTransferInfo> block_transfer_info;
  uint64_t batch_id = 0;
  if (!proto_to_block_transfer_info(*req, block_transfer_info, &batch_id)) {
    resp->set_success_cnt(0);
    controller->SetFailed("Invalid block transfer request");
    LOG(ERROR) << "Invalid block transfer request.";
    return;
  }

  resp->set_success_cnt(
      worker_->transfer_kv_blocks(batch_id, block_transfer_info));
  return;
}

void WorkerService::PrefetchFromStorage(
    google::protobuf::RpcController* controller,
    const proto::PrefetchRequest* req,
    proto::Status* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

  StoragePrefetchRequest request;
  if (!request.from_proto(*req)) {
    resp->set_ok(false);
    LOG(ERROR) << "Invalid Mooncake prefetch request.";
    return;
  }

  auto session = std::make_shared<WorkerPrefetchSession>(
      worker_.get(), &copy_threadpool_, std::move(request));

  brpc::StreamId stream_id;
  brpc::StreamOptions stream_options;
  stream_options.idle_timeout_ms = -1;
  stream_options.handler = session.get();
  session->retain();
  if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
    session->release();
    resp->set_ok(false);
    LOG(ERROR) << "Failed to accept stream!";
    return;
  }

  resp->set_ok(true);
  if (google::protobuf::Closure* response_done = done_guard.release()) {
    response_done->Run();
  }
  session->start(stream_id);
}

void WorkerService::LinkCluster(::google::protobuf::RpcController* controller,
                                const proto::ClusterInfo* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status = worker_->link_cluster(cluster_ids, addrs, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::UnlinkCluster(::google::protobuf::RpcController* controller,
                                  const proto::ClusterInfo* req,
                                  proto::Status* resp,
                                  ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status = worker_->unlink_cluster(cluster_ids, addrs, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::LinkP2P(::google::protobuf::RpcController* controller,
                            const proto::P2PLinkWorkerRequest* req,
                            proto::Status* resp,
                            ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->link_p2p(req->remote_addr());
    resp->set_ok(status);
  });
  return;
}

void WorkerService::UnlinkP2P(::google::protobuf::RpcController* controller,
                              const proto::P2PLinkWorkerRequest* req,
                              proto::Status* resp,
                              ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->unlink_p2p(req->remote_addr());
    resp->set_ok(status);
  });
  return;
}

void WorkerService::Sleep(::google::protobuf::RpcController* controller,
                          const proto::SleepRequest* req,
                          proto::Status* resp,
                          ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->sleep(MasterStatus(req->master_status()));
    resp->set_ok(status);
  });

  return;
}

void WorkerService::Wakeup(::google::protobuf::RpcController* controller,
                           const proto::WakeupRequest* req,
                           proto::Status* resp,
                           ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    WakeupOptions options;
    options.master_status = MasterStatus(req->master_status());
    options.remote_addrs.assign(req->remote_addrs().begin(),
                                req->remote_addrs().end());
    // Unmarshal weight segments
    for (const auto& seg_list : req->src_weight_segments()) {
      std::vector<WeightSegment> segments;
      segments.reserve(seg_list.segments_size());
      for (const auto& proto_seg : seg_list.segments()) {
        segments.emplace_back(proto_seg.offset(), proto_seg.size());
      }
      options.src_weight_segments.push_back(std::move(segments));
    }
    bool status = worker_->wakeup(options);
    resp->set_ok(status);
  });

  return;
}

void WorkerService::StartProfile(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::Status* resp,
                                 ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->start_profile();
    resp->set_ok(status);
  });

  return;
}

void WorkerService::StopProfile(::google::protobuf::RpcController* controller,
                                const proto::Empty* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->stop_profile();
    resp->set_ok(status);
  });

  return;
}

void WorkerService::GetActiveActivationMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* req,
    proto::ActivationMemory* resp,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->get_active_activation_memory_async();
    int64_t active_activation_memory = std::move(future).get();
    resp->set_active_activation_memory(active_activation_memory);
  });
  return;
}
}  // namespace xllm

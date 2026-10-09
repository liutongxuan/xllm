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

#include "core/distributed_runtime/xtensor_controller.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/xtensor/page_allocator.h"
#include "core/framework/xtensor/xtensor_allocator.h"

namespace xllm {

XTensorController::XTensorController(
    Options options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(std::move(options)),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {}

void XTensorController::get_xtensor_info(
    std::vector<size_t>& worker_free_phy_pages,
    std::unordered_map<std::string, std::vector<WeightSegment>>&
        model_weight_segments) const {
  if (!options_.enabled) {
    return;
  }

  // Worker 0 shares the master's process, so these queries need no RPC.
  auto& page_allocator = PageAllocator::get_instance();
  if (page_allocator.is_initialized()) {
    worker_free_phy_pages = page_allocator.get_all_worker_free_pages();
  }

  auto& xtensor_allocator = XTensorAllocator::get_instance();
  model_weight_segments = xtensor_allocator.get_all_model_weight_segments();
}

bool XTensorController::sleep(MasterStatus master_status) {
  if (!options_.enabled) {
    LOG(WARNING) << "sleep requires --enable_xtensor=true";
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to sleep.";
    return false;
  }

  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  LOG(INFO) << "Starting to sleep model " << options_.model_id
            << ". Worker clients count: " << worker_clients.size();

  // Release weight and KV cache pages before changing worker model state.
  auto& page_allocator = PageAllocator::get_instance();
  if (!page_allocator.sleep_model(options_.model_id)) {
    LOG(ERROR) << "PageAllocator sleep_model failed, aborting sleep flow";
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());
  for (const auto& worker : worker_clients) {
    futures.emplace_back(worker->sleep_async(master_status));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Sleep failed.";
      return false;
    }
  }
  return true;
}

bool XTensorController::wakeup(const WakeupOptions& options) {
  if (!options_.enabled) {
    LOG(WARNING) << "wakeup requires --enable_xtensor=true";
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to wakeup.";
    return false;
  }

  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  LOG(INFO) << "Starting to wakeup model " << options_.model_id
            << ". Worker clients count: " << worker_clients.size();

  // Restore weight and KV cache pages before workers load or transfer weights.
  auto& page_allocator = PageAllocator::get_instance();
  if (!page_allocator.wakeup_model(options_.model_id)) {
    LOG(ERROR) << "PageAllocator wakeup_model failed, aborting wakeup flow";
    return false;
  }

  LOG(INFO) << "Waking up model " << options_.model_id
            << ", remote_addrs.size()=" << options.remote_addrs.size();
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());

  if (!options.remote_addrs.empty() &&
      options.remote_addrs.size() == worker_clients.size()) {
    // Each worker pulls weights from the source at the same global rank.
    for (size_t i = 0; i < worker_clients.size(); ++i) {
      WakeupOptions per_worker_options;
      per_worker_options.master_status = options.master_status;
      per_worker_options.remote_addrs = {options.remote_addrs[i]};
      if (i < options.src_weight_segments.size()) {
        per_worker_options.src_weight_segments = {
            options.src_weight_segments[i]};
      }
      futures.emplace_back(worker_clients[i]->wakeup_async(per_worker_options));
    }
  } else {
    for (const auto& worker : worker_clients) {
      futures.emplace_back(worker->wakeup_async(options));
    }
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Wakeup failed.";
      return false;
    }
  }
  LOG(INFO) << "Wakeup finished for model " << options_.model_id << ".";
  return true;
}

bool XTensorController::get_xtensor_offsets_for_blocks(
    int32_t dp_rank,
    const std::vector<int32_t>& block_ids,
    uint64_t slot_size,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) const {
  if (!options_.enabled) {
    return false;
  }

  const uint64_t block_size_bytes = slot_size * options_.block_size / 2;
  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.get_xtensor_offsets(dp_rank,
                                     options_.model_id,
                                     block_ids,
                                     block_size_bytes,
                                     layer_offsets)) {
    LOG(ERROR) << "get_xtensor_offsets_for_blocks via RPC failed for dp_rank="
               << dp_rank << ", model_id=" << options_.model_id;
    return false;
  }

  VLOG(1) << "get_xtensor_offsets_for_blocks: dp_rank=" << dp_rank
          << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();
  return true;
}

}  // namespace xllm

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

#pragma once

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/kv_cache_transfer/prefetch_result.h"
#include "core/framework/model/model_input_params.h"

namespace xllm {

// Non-owning worker transfer operations consumed by the hierarchical cache.
class KVCacheTransferOperations final {
 public:
  using Transfer = std::function<std::vector<folly::SemiFuture<uint32_t>>(
      uint32_t,
      const std::vector<BlockTransferInfo>&)>;
  using Load = std::function<
      void(uint32_t, uint64_t, const std::vector<BlockTransferInfo>&)>;
  using Prefetch =
      std::function<void(uint32_t,
                         std::shared_ptr<const StoragePrefetchRequest>,
                         PrefetchResult::StopPredicate,
                         PrefetchResult::DoneCallback)>;

  KVCacheTransferOperations() = default;
  KVCacheTransferOperations(Transfer transfer, Load load, Prefetch prefetch)
      : transfer_(std::move(transfer)),
        load_(std::move(load)),
        prefetch_(std::move(prefetch)) {}

  template <typename TargetEngine>
  static KVCacheTransferOperations bind(TargetEngine& engine) {
    Transfer transfer;
    if constexpr (requires(uint32_t rank,
                           const std::vector<BlockTransferInfo>& infos) {
                    engine.transfer_kv_blocks(rank, infos);
                  }) {
      transfer = [&engine](uint32_t rank, const auto& infos) {
        return engine.transfer_kv_blocks(rank, infos);
      };
    }
    Load load;
    if constexpr (requires(uint32_t rank,
                           uint64_t batch_id,
                           const std::vector<BlockTransferInfo>& infos) {
                    engine.transfer_kv_blocks(rank, batch_id, infos);
                  }) {
      load = [&engine](uint32_t rank, uint64_t batch_id, const auto& infos) {
        engine.transfer_kv_blocks(rank, batch_id, infos);
      };
    }
    Prefetch prefetch;
    if constexpr (requires(
                      uint32_t rank,
                      std::shared_ptr<const StoragePrefetchRequest> request,
                      PrefetchResult::StopPredicate stop,
                      PrefetchResult::DoneCallback done) {
                    engine.prefetch_from_storage(rank, request, stop, done);
                  }) {
      prefetch = [&engine](uint32_t rank, auto request, auto stop, auto done) {
        engine.prefetch_from_storage(
            rank, std::move(request), std::move(stop), std::move(done));
      };
    }
    return KVCacheTransferOperations(
        std::move(transfer), std::move(load), std::move(prefetch));
  }

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t rank,
      const std::vector<BlockTransferInfo>& infos) const {
    CHECK(transfer_) << "KV block transfer capability is unavailable";
    return transfer_(rank, infos);
  }
  void transfer_kv_blocks(uint32_t rank,
                          uint64_t batch_id,
                          const std::vector<BlockTransferInfo>& infos) const {
    CHECK(load_) << "KV block load capability is unavailable";
    load_(rank, batch_id, infos);
  }
  void prefetch_from_storage(
      uint32_t rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop,
      PrefetchResult::DoneCallback done) const {
    CHECK(prefetch_) << "Storage prefetch capability is unavailable";
    prefetch_(rank, std::move(request), std::move(stop), std::move(done));
  }

 private:
  Transfer transfer_;
  Load load_;
  Prefetch prefetch_;
};

}  // namespace xllm

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

#include <glog/logging.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"

namespace xllm {

// Only cluster cache exchange capabilities; no model or batch execution API.
class PDExecution final {
 public:
  using CacheInfo = std::function<void(std::vector<uint64_t>&,
                                       std::vector<std::string>&,
                                       std::vector<uint16_t>&)>;
  using Pull = std::function<bool(int32_t,
                                  int32_t,
                                  const std::vector<uint64_t>&,
                                  const std::vector<std::string>&,
                                  int32_t,
                                  const std::vector<KVTransferMapping>&)>;
  using Connect = std::function<bool(const std::vector<uint64_t>&,
                                     const std::vector<std::string>&,
                                     const std::vector<uint16_t>&,
                                     int32_t,
                                     int32_t)>;
  using LayerOffsets =
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>;
  using Offsets =
      std::function<bool(int32_t, const std::vector<int32_t>&, LayerOffsets&)>;

  PDExecution() = default;
  PDExecution(CacheInfo cache_info,
              Pull pull,
              Connect link,
              Connect unlink,
              Offsets offsets = {})
      : cache_info_(std::move(cache_info)),
        pull_(std::move(pull)),
        link_(std::move(link)),
        unlink_(std::move(unlink)),
        offsets_(std::move(offsets)) {}

  template <typename TargetEngine>
  static PDExecution bind(TargetEngine& engine) {
    CacheInfo cache_info;
    if constexpr (requires(std::vector<uint64_t>& ids,
                           std::vector<std::string>& addrs,
                           std::vector<uint16_t>& ports) {
                    engine.get_cache_info(ids, addrs, ports);
                  }) {
      cache_info = [&engine](auto& ids, auto& addrs, auto& ports) {
        engine.get_cache_info(ids, addrs, ports);
      };
    }
    Pull pull;
    if constexpr (requires(int32_t size,
                           int32_t rank,
                           const std::vector<uint64_t>& ids,
                           const std::vector<std::string>& addrs,
                           const std::vector<KVTransferMapping>& mappings) {
                    engine.pull_kv_blocks(
                        size, rank, ids, addrs, rank, mappings);
                  }) {
      pull = [&engine](int32_t size,
                       int32_t src_rank,
                       const auto& ids,
                       const auto& addrs,
                       int32_t dst_rank,
                       const auto& mappings) {
        return engine.pull_kv_blocks(
            size, src_rank, ids, addrs, dst_rank, mappings);
      };
    }
    Connect link;
    if constexpr (requires(const std::vector<uint64_t>& ids,
                           const std::vector<std::string>& addrs,
                           const std::vector<uint16_t>& ports,
                           int32_t size) {
                    engine.link_cluster(ids, addrs, ports, size, size);
                  }) {
      link = [&engine](const auto& ids,
                       const auto& addrs,
                       const auto& ports,
                       int32_t size,
                       int32_t split_size) {
        return engine.link_cluster(ids, addrs, ports, size, split_size);
      };
    }
    Connect unlink;
    if constexpr (requires(const std::vector<uint64_t>& ids,
                           const std::vector<std::string>& addrs,
                           const std::vector<uint16_t>& ports,
                           int32_t size) {
                    engine.unlink_cluster(ids, addrs, ports, size, size);
                  }) {
      unlink = [&engine](const auto& ids,
                         const auto& addrs,
                         const auto& ports,
                         int32_t size,
                         int32_t split_size) {
        return engine.unlink_cluster(ids, addrs, ports, size, split_size);
      };
    }
    Offsets offsets;
    if constexpr (requires(int32_t rank,
                           const std::vector<int32_t>& blocks,
                           LayerOffsets& layers) {
                    engine.get_xtensor_offsets_for_blocks(rank, blocks, layers);
                  }) {
      offsets = [&engine](int32_t rank, const auto& blocks, auto& layers) {
        return engine.get_xtensor_offsets_for_blocks(rank, blocks, layers);
      };
    }
    return PDExecution(std::move(cache_info),
                       std::move(pull),
                       std::move(link),
                       std::move(unlink),
                       std::move(offsets));
  }

  bool supports_cache_registration() const {
    return static_cast<bool>(cache_info_);
  }

  void validate_cluster_exchange() const {
    CHECK(cache_info_) << "PD requires cache registration";
    CHECK(pull_) << "PD requires KV block pulling";
    CHECK(link_) << "PD requires cluster linking";
    CHECK(unlink_) << "PD requires cluster unlinking";
  }

  static PDExecution checked(PDExecution execution) {
    execution.validate_cluster_exchange();
    return execution;
  }

  void get_cache_info(std::vector<uint64_t>& ids,
                      std::vector<std::string>& addrs,
                      std::vector<uint16_t>& ports) const {
    CHECK(cache_info_) << "Cache registration capability is unavailable";
    cache_info_(ids, addrs, ports);
  }
  bool pull_kv_blocks(int32_t size,
                      int32_t src_rank,
                      const std::vector<uint64_t>& ids,
                      const std::vector<std::string>& addrs,
                      int32_t dst_rank,
                      const std::vector<KVTransferMapping>& mappings) const {
    CHECK(pull_) << "PD cache pull capability is unavailable";
    return pull_(size, src_rank, ids, addrs, dst_rank, mappings);
  }
  bool link_cluster(const std::vector<uint64_t>& ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports,
                    int32_t size,
                    int32_t split_size = 1) const {
    CHECK(link_) << "PD cluster link capability is unavailable";
    return link_(ids, addrs, ports, size, split_size);
  }
  bool unlink_cluster(const std::vector<uint64_t>& ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports,
                      int32_t size,
                      int32_t split_size = 1) const {
    CHECK(unlink_) << "PD cluster unlink capability is unavailable";
    return unlink_(ids, addrs, ports, size, split_size);
  }
  bool get_xtensor_offsets_for_blocks(int32_t rank,
                                      const std::vector<int32_t>& blocks,
                                      LayerOffsets& layers) const {
    return offsets_ && offsets_(rank, blocks, layers);
  }

 private:
  CacheInfo cache_info_;
  Pull pull_;
  Connect link_;
  Connect unlink_;
  Offsets offsets_;
};

}  // namespace xllm

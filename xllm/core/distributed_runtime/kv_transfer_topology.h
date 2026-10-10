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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace xllm {

struct DPWorkerRange {
  size_t begin = 0;
  size_t count = 0;
};

struct KVWorkerRoute {
  size_t src_rank = 0;
  size_t dst_rank = 0;
};

// Transfer participation uses the entire DP worker group, including CP
// workers. The attention TP width alone is not the DP group's rank stride.
class KVTransferTopology final {
 public:
  static std::optional<DPWorkerRange> get_dp_worker_range(size_t worker_count,
                                                          int32_t dp_size,
                                                          int32_t dp_rank) {
    if (worker_count == 0 || dp_size <= 0 || dp_rank < 0 ||
        dp_rank >= dp_size) {
      return std::nullopt;
    }
    const size_t dp_count = static_cast<size_t>(dp_size);
    if (worker_count % dp_count != 0) {
      return std::nullopt;
    }
    const size_t workers_per_dp = worker_count / dp_count;
    return DPWorkerRange{static_cast<size_t>(dp_rank) * workers_per_dp,
                         workers_per_dp};
  }

  // PULL does not reshard caches: each destination worker reads the same
  // relative rank in the selected source DP group. Heterogeneous PD uses PUSH.
  static std::optional<std::vector<KVWorkerRoute>> get_pull_worker_routes(
      size_t src_worker_count,
      int32_t src_dp_size,
      int32_t src_dp_rank,
      size_t dst_worker_count,
      int32_t dst_dp_size,
      int32_t dst_dp_rank) {
    const auto src =
        get_dp_worker_range(src_worker_count, src_dp_size, src_dp_rank);
    const auto dst =
        get_dp_worker_range(dst_worker_count, dst_dp_size, dst_dp_rank);
    if (!src.has_value() || !dst.has_value() || src_dp_size != dst_dp_size ||
        src->count != dst->count) {
      return std::nullopt;
    }

    std::vector<KVWorkerRoute> routes;
    routes.reserve(dst->count);
    for (size_t local_rank = 0; local_rank < dst->count; ++local_rank) {
      routes.emplace_back(
          KVWorkerRoute{src->begin + local_rank, dst->begin + local_rank});
    }
    return routes;
  }
};

}  // namespace xllm

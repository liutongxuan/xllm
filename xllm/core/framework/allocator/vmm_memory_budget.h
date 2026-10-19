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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

#include "core/platform/allocator_memory_stats.h"

namespace xllm {

// Protects allocator execution peaks independently of elastic physical backing.
// Backends without allocator statistics retain the driver low-watermark
// fallback.
class VmmMemoryBudget final {
 public:
  void observe(
      size_t free_bytes,
      size_t committed_bytes,
      std::optional<AllocatorMemoryStats> allocator_stats = std::nullopt) {
    const size_t available =
        outside_pool_available(free_bytes, committed_bytes);
    low_watermark_ = low_watermark_.has_value()
                         ? std::min(*low_watermark_, available)
                         : available;
    observe_allocator_memory(allocator_stats);
  }

  size_t headroom(size_t free_bytes,
                  size_t committed_bytes,
                  size_t configured_reserve,
                  std::optional<AllocatorMemoryStats> allocator_stats =
                      std::nullopt) const {
    observe_allocator_memory(allocator_stats);
    if (allocator_stats.has_value()) {
      // Resident allocator caches already reduce driver free memory. Reserve
      // only the missing bytes needed to recover the largest observed peak.
      const size_t recovery =
          peak_reserved_bytes_ - allocator_stats->reserved_bytes;
      return saturating_add(configured_reserve, recovery);
    }
    if (!low_watermark_.has_value()) {
      return configured_reserve;
    }
    const size_t available =
        outside_pool_available(free_bytes, committed_bytes);
    const size_t recovery =
        available > *low_watermark_ ? available - *low_watermark_ : 0;
    // Resident allocator caches already reduce driver free memory. Reserve
    // only the additional bytes needed if those caches have been released.
    return saturating_add(configured_reserve, recovery);
  }

  void observe_allocator_memory(
      std::optional<AllocatorMemoryStats> allocator_stats) const {
    if (allocator_stats.has_value()) {
      // Backend peak counters may be reset by callers. Preserve the local
      // high-water mark, and accept current residency as a peak lower bound.
      peak_reserved_bytes_ =
          std::max(peak_reserved_bytes_,
                   std::max(allocator_stats->reserved_bytes,
                            allocator_stats->peak_reserved_bytes));
    }
  }

  void reset() {
    low_watermark_.reset();
    peak_reserved_bytes_ = 0;
  }

 private:
  static size_t saturating_add(size_t lhs, size_t rhs) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    return lhs > maximum - rhs ? maximum : lhs + rhs;
  }

  static size_t outside_pool_available(size_t free_bytes,
                                       size_t committed_bytes) {
    return saturating_add(free_bytes, committed_bytes);
  }

  std::optional<size_t> low_watermark_;
  mutable size_t peak_reserved_bytes_ = 0;
};

}  // namespace xllm

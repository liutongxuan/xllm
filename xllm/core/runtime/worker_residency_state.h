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
#include <mutex>
#include <optional>
#include <utility>

#include "core/common/types.h"

namespace xllm {

// Residency RPCs can outlive a caller timeout. Serialize their work and
// remember successful transitions so retrying an unknown result cannot reload
// weights.
class WorkerResidencyState final {
 public:
  template <typename Operation>
  bool initialize(MasterStatus status, Operation&& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status != MasterStatus::WAKEUP && !is_sleep_status(status)) {
      return false;
    }
    if (!std::forward<Operation>(operation)()) {
      return false;
    }
    status_ = status;
    transition_id_ = 0;
    sleep_status_ = is_sleep_status(status)
                        ? std::optional<MasterStatus>(status)
                        : std::nullopt;
    last_wakeup_.reset();
    return true;
  }

  template <typename Operation>
  bool sleep(MasterStatus status,
             uint64_t transition_id,
             Operation&& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status_.has_value() || !is_sleep_status(status)) {
      return false;
    }
    if (transition_id < transition_id_) {
      return false;
    }
    if (transition_id == transition_id_) {
      // A completed sleep remains acknowledged after its matching wakeup.
      // Otherwise a late retry could put an already awake worker to sleep.
      return sleep_status_.has_value() && *sleep_status_ == status;
    }
    if (*status_ != MasterStatus::WAKEUP) {
      return false;
    }
    if (!std::forward<Operation>(operation)()) {
      return false;
    }
    status_ = status;
    transition_id_ = transition_id;
    sleep_status_ = status;
    last_wakeup_.reset();
    return true;
  }

  template <typename Operation>
  bool wakeup(const WakeupOptions& options, Operation&& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status_.has_value() || !is_sleep_status(options.master_status) ||
        options.transition_id != transition_id_ || !sleep_status_.has_value() ||
        *sleep_status_ != options.master_status) {
      return false;
    }
    if (*status_ == MasterStatus::WAKEUP) {
      return last_wakeup_.has_value() &&
             same_wakeup_options(*last_wakeup_, options);
    }
    if (*status_ != options.master_status) {
      return false;
    }
    // Copy before loading weights so an allocation failure cannot lose the
    // successful request identity after the device operation has completed.
    WakeupOptions completed_options = options;
    if (!std::forward<Operation>(operation)()) {
      return false;
    }
    last_wakeup_ = std::move(completed_options);
    status_ = MasterStatus::WAKEUP;
    return true;
  }

 private:
  static bool is_sleep_status(MasterStatus status) {
    return status == MasterStatus::LIGHT_SLEEP ||
           status == MasterStatus::DEEP_SLEEP;
  }

  static bool same_wakeup_options(const WakeupOptions& lhs,
                                  const WakeupOptions& rhs) {
    if (lhs.master_status != rhs.master_status ||
        lhs.transition_id != rhs.transition_id ||
        lhs.remote_addrs != rhs.remote_addrs ||
        lhs.src_weight_segments.size() != rhs.src_weight_segments.size()) {
      return false;
    }
    for (size_t source = 0; source < lhs.src_weight_segments.size(); ++source) {
      const auto& lhs_segments = lhs.src_weight_segments[source];
      const auto& rhs_segments = rhs.src_weight_segments[source];
      if (lhs_segments.size() != rhs_segments.size()) {
        return false;
      }
      for (size_t segment = 0; segment < lhs_segments.size(); ++segment) {
        if (lhs_segments[segment].offset != rhs_segments[segment].offset ||
            lhs_segments[segment].size != rhs_segments[segment].size) {
          return false;
        }
      }
    }
    return true;
  }

  std::mutex mutex_;
  std::optional<MasterStatus> status_;
  uint64_t transition_id_ = 0;
  std::optional<MasterStatus> sleep_status_;
  std::optional<WakeupOptions> last_wakeup_;
};

}  // namespace xllm

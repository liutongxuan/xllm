/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/batch_sequence_ordering.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>

#include "core/framework/batch/batch_storage.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"

namespace xllm {

void BatchSequenceOrdering::prepare([[maybe_unused]] BatchStorage& storage) {
#if defined(USE_NPU)
  // this shuffle operation is mainly used for npu with 24 cores
  // and specific mla op implementation
  constexpr size_t kNumNpuCores = 24;
  if (::xllm::KernelConfig::get_instance().enable_customize_mla_kernel() &&
      ::xllm::ParallelConfig::get_instance().enable_dp_balance() &&
      storage.sequence_plan().sequences().size() > kNumNpuCores) {
    std::vector<uint32_t> kv_cache_tokens_num;
    kv_cache_tokens_num.reserve(storage.sequence_plan().sequences().size());
    for (auto& seq : storage.sequence_plan().sequences()) {
      kv_cache_tokens_num.push_back(seq->kv_state().kv_cache_tokens_num());
    }
    auto seq_index_shift = cal_seq_exchange_index(kv_cache_tokens_num);

    std::vector<size_t> source_indices(storage.sequence_plan().size());
    for (const auto& [source, target] : seq_index_shift) {
      source_indices[target] = source;
    }
    storage.sequence_plan().reorder(source_indices);
  }
#else
  // TODO: implement dp_balance_shuffle_seqs for non-npu devices
  static bool warning = true;
  if (warning) {
    LOG(WARNING)
        << "dp_balance_shuffle_seqs is not implemented for current device";
    warning = false;
  }
#endif
}

std::unordered_map<uint32_t, uint32_t>
BatchSequenceOrdering::cal_seq_exchange_index(
    std::vector<uint32_t>& kv_cache_tokens_num) {
  constexpr size_t kNumNpuCores = 24;
  const size_t num_seqs = kv_cache_tokens_num.size();
  const size_t base_per_core = num_seqs / kNumNpuCores;
  const size_t remainder = num_seqs % kNumNpuCores;

  // find the indices of the remainder biggest elements
  std::vector<uint32_t> indices(num_seqs);
  std::iota(indices.begin(), indices.end(), 0);
  if (remainder > 0) {
    std::nth_element(indices.begin(),
                     indices.end() - remainder,
                     indices.end(),
                     [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
                       return kv_cache_tokens_num[a] < kv_cache_tokens_num[b];
                     });
  }

  std::vector<uint32_t> base_indices(indices.begin(),
                                     indices.end() - remainder);
  std::vector<uint32_t> remainder_indices(indices.end() - remainder,
                                          indices.end());

  // sort base_indices in descending order
  std::sort(base_indices.begin(),
            base_indices.end(),
            [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
              return kv_cache_tokens_num[a] > kv_cache_tokens_num[b];
            });

  // allocate a long and a short request to each core, to ensuring
  // load balance among all cores
  std::vector<std::vector<uint32_t>> base_assignment(
      kNumNpuCores, std::vector<uint32_t>(base_per_core));
  for (size_t i = 0; i < base_indices.size(); ++i) {
    const size_t col = i / kNumNpuCores;
    const size_t row = (col % 2 == 0) ? (i % kNumNpuCores)
                                      : (kNumNpuCores - 1 - (i % kNumNpuCores));
    base_assignment[row][col] = base_indices[i];
  }

  // record the index map, first one is original index,
  // second one is the target index to be exchanged to
  std::unordered_map<uint32_t, uint32_t> index_shift;
  // add base part data
  for (size_t i = 0; i < kNumNpuCores; ++i) {
    for (size_t j = 0; j < base_per_core; ++j) {
      const uint32_t idx = base_assignment[i][j];
      index_shift[idx] = static_cast<uint32_t>(i + j * kNumNpuCores);
    }
  }
  // add remainder part data
  for (size_t i = 0; i < remainder; ++i) {
    index_shift[remainder_indices[i]] =
        static_cast<uint32_t>(i + kNumNpuCores * base_per_core);
  }

  return index_shift;
}

}  // namespace xllm

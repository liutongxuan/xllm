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

#include "runtime/params_utils.h"

#include <glog/logging.h>

namespace xllm {

torch::Tensor choose_lm_head_selected_token_idxes(
    const torch::Tensor& selected_token_idxes,
    const ModelInputParams& input_params,
    const ParallelArgs& parallel_args,
    int64_t hidden_num_rows,
    const torch::Device& device) {
  const auto& mapping = parallel_args.mapping_data();
  if (!selected_token_idxes.defined() || selected_token_idxes.numel() == 0 ||
      mapping.empty() || !mapping.contains("attnDp") ||
      !mapping["attnDp"].contains("rank") ||
      input_params.parallel.dp_global_token_nums.size() <= 1 ||
      hidden_num_rows <= 0) {
    return selected_token_idxes;
  }

  const int64_t dp_rank = mapping["attnDp"]["rank"].get<int64_t>();
  CHECK_GE(dp_rank, 0) << "invalid attnDp rank";
  CHECK_LT(
      dp_rank,
      static_cast<int64_t>(input_params.parallel.dp_global_token_nums.size()))
      << "attnDp rank exceeds dp_global_token_nums";

  const int64_t local_selected_rows = selected_token_idxes.numel();
  const int64_t local_token_count =
      input_params.parallel.dp_global_token_nums.at(dp_rank);
  if (hidden_num_rows == local_token_count ||
      hidden_num_rows == local_selected_rows) {
    return selected_token_idxes;
  }

  int64_t dp_offset = 0;
  for (int64_t i = 0; i < dp_rank; ++i) {
    dp_offset += input_params.parallel.dp_global_token_nums[i];
  }

  torch::Tensor selected_cpu =
      selected_token_idxes.to(torch::dtype(torch::kLong).device(torch::kCPU));
  torch::Tensor logical_selected_cpu = selected_cpu + dp_offset;

  const auto& padding_idx = input_params.parallel.dp_ep_padding_data
                                .lm_head_skip_padding_token_indices();
  if (padding_idx.defined() && padding_idx.numel() > 0 &&
      hidden_num_rows > padding_idx.numel()) {
    torch::Tensor padding_cpu =
        padding_idx.to(torch::dtype(torch::kLong).device(torch::kCPU));
    const int64_t max_logical_selected =
        logical_selected_cpu.max().item<int64_t>();
    if (max_logical_selected < padding_cpu.numel()) {
      return padding_cpu.index_select(/*dim=*/0, logical_selected_cpu)
          .to(torch::dtype(selected_token_idxes.scalar_type()).device(device),
              /*non_blocking=*/false)
          .contiguous();
    }
  }

  if (dp_offset == 0) {
    return selected_token_idxes;
  }

  const int64_t max_selected_idx = selected_cpu.max().item<int64_t>();
  if (max_selected_idx + dp_offset >= hidden_num_rows) {
    return selected_token_idxes;
  }

  torch::Tensor remapped =
      selected_token_idxes.to(device, /*non_blocking=*/false).contiguous();
  return (remapped + dp_offset).to(remapped.scalar_type()).contiguous();
}

}  // namespace xllm

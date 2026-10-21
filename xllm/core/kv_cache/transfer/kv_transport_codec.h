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

#include <cstdint>
#include <vector>

#include "framework/model/model_params_view.h"

namespace xllm {
namespace proto {
class BlockTransferInfos;
}  // namespace proto

bool proto_to_block_transfer_info(
    const proto::BlockTransferInfos& input,
    std::vector<BlockTransferInfo>& transfer_infos,
    uint64_t* batch_id);

bool block_transfer_info_to_proto(
    const std::vector<BlockTransferInfo>& transfer_infos,
    proto::BlockTransferInfos* output);

bool block_transfer_info_to_proto(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& transfer_infos,
    proto::BlockTransferInfos* output);

}  // namespace xllm

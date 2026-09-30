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

#include "core/framework/batch/batch_storage.h"

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <utility>

namespace xllm {

void BatchStorage::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  set_batch_id();
  sequence_plan_.add(sequence, allowed_max_token);

  const auto& input_embedding = sequence->get_input_embedding();
  if (input_embedding.defined()) {
    input_embeddings_vec_.emplace_back(input_embedding);
  }

  update_forward_type(sequence);
}

void BatchStorage::update_forward_type(Sequence* sequence) {
  const SequenceStage stage = sequence->stage();
  switch (batch_forward_type_.value()) {
    case BatchForwardType::PREFILL:
      if (stage == SequenceStage::CHUNKED_PREFILL) {
        batch_forward_type_ = BatchForwardType::CHUNKED_PREFILL;
      } else if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::CHUNKED_PREFILL:
      if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::DECODE:
      if (stage != SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::MIXED:
      break;
    case BatchForwardType::EMPTY:
      batch_forward_type_ = BatchForwardType(static_cast<int32_t>(stage));
      break;
  }
}

void BatchStorage::set_batch_id() {
  static std::atomic<uint64_t> next_batch_id{1};
  while (batch_id_ == UNINITIALIZED_BATCH_ID) {
    batch_id_ = next_batch_id.fetch_add(1, std::memory_order_relaxed);
  }
}

void BatchStorage::reserve(size_t sequence_count, size_t group_count) {
  sequence_plan_.reserve(sequence_count);
  input_embeddings_vec_.reserve(sequence_count);
  sequence_groups_.reserve(group_count);
}

void BatchStorage::add(SequencesGroup* sequence_group) {
  CHECK(sequence_group != nullptr);
  CHECK(!sequence_group->sequences().empty());
  set_batch_id();
  sequence_groups_.emplace_back(sequence_group);
}

bool BatchStorage::has_partial_finished_beam_group() const {
  if (sequence_groups_.empty()) {
    return false;
  }

  for (auto* seq_group : sequence_groups_) {
    if (!seq_group->check_beam_search()) {
      continue;
    }

    const auto& sequences = seq_group->sequences();
    if (sequences.empty()) {
      continue;
    }

    const size_t finished_cnt = static_cast<size_t>(
        std::count_if(sequences.begin(), sequences.end(), [](const auto& seq) {
          return seq->finished();
        }));
    if (finished_cnt > 0 && finished_cnt < sequences.size()) {
      return true;
    }
  }
  return false;
}

void BatchStorage::refresh_sequences_from_groups() {
  if (sequence_groups_.empty()) {
    return;
  }
  BatchSequencePlan next_plan;
  size_t sequence_count = 0;
  for (const auto* group : sequence_groups_) {
    sequence_count += group->sequences().size();
  }
  next_plan.reserve(sequence_count);
  for (const auto* group : sequence_groups_) {
    for (const auto& sequence : group->sequences()) {
      next_plan.add(sequence.get(), std::numeric_limits<uint32_t>::max());
    }
  }
  sequence_plan_ = std::move(next_plan);
}

void BatchStorage::refresh_forward_type(
    const std::vector<Sequence*>& sequences) {
  batch_forward_type_ = BatchForwardType();
  for (auto* sequence : sequences) {
    update_forward_type(sequence);
  }
}

size_t BatchStorage::num_group_sequences() const {
  size_t count = 0;
  for (const auto* group : sequence_groups_) {
    count += group->sequences().size();
  }
  return count;
}

Sequence* BatchStorage::group_sequence(size_t index) const {
  for (const auto* group : sequence_groups_) {
    if (index < group->sequences().size()) {
      return group->sequences()[index].get();
    }
    index -= group->sequences().size();
  }
  LOG(FATAL) << "Group sequence index out of range";
  return nullptr;
}

BatchInputData BatchStorage::input_data(const BatchSequencePlan& plan) {
  return {plan.sequences(),
          sequence_groups_,
          plan.budgets(),
          input_embeddings_vec_,
          mm_data_vec_,
          &swap_block_transfer_infos_,
          batch_id_,
          batch_forward_type_};
}

std::vector<Sequence*> BatchStorage::group_sequences() const {
  std::vector<Sequence*> sequences;
  sequences.reserve(num_group_sequences());
  for (const auto* group : sequence_groups_) {
    for (const auto& sequence : group->sequences()) {
      sequences.emplace_back(sequence.get());
    }
  }
  return sequences;
}

}  // namespace xllm

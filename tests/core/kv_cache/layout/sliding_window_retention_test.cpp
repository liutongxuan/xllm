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

#include "core/kv_cache/layout/sliding_window_retention.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(SlidingWindowRetentionTest, RestorationKeepsPartialWindowStartBlock) {
  const SlidingWindowRetention retention(/*token_stride=*/4,
                                         /*window_tokens=*/8,
                                         /*speculative_tokens=*/0);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/0), 0u);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/8), 0u);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/11), 0u);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/12), 1u);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/13), 1u);
}

TEST(SlidingWindowRetentionTest, ReclamationKeepsHistoryForNextForward) {
  const SlidingWindowRetention retention(/*token_stride=*/4,
                                         /*window_tokens=*/8,
                                         /*speculative_tokens=*/0);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/10,
                                        /*logical_blocks=*/4),
            0u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/11,
                                        /*logical_blocks=*/4),
            1u);
  // Restoration keeps an extra token at this unaligned boundary.
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/11), 0u);
}

TEST(SlidingWindowRetentionTest, SpeculativeRollbackDelaysReclamation) {
  const SlidingWindowRetention retention(/*token_stride=*/4,
                                         /*window_tokens=*/8,
                                         /*speculative_tokens=*/2);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/7,
                                        /*logical_blocks=*/4),
            0u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/9,
                                        /*logical_blocks=*/4),
            0u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/12,
                                        /*logical_blocks=*/4),
            0u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/13,
                                        /*logical_blocks=*/4),
            1u);
}

TEST(SlidingWindowRetentionTest, ReclamationCannotExceedLogicalHistory) {
  const SlidingWindowRetention retention(/*token_stride=*/4,
                                         /*window_tokens=*/8,
                                         /*speculative_tokens=*/0);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/100,
                                        /*logical_blocks=*/0),
            0u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/100,
                                        /*logical_blocks=*/2),
            2u);
}

TEST(SlidingWindowRetentionTest, ZeroWindowKeepsSingleTokenHistory) {
  const SlidingWindowRetention retention(/*token_stride=*/4,
                                         /*window_tokens=*/0,
                                         /*speculative_tokens=*/0);
  EXPECT_EQ(retention.restore_begin_block(/*restored_tokens=*/8), 1u);
  EXPECT_EQ(retention.release_end_block(/*cached_tokens=*/8,
                                        /*logical_blocks=*/3),
            2u);
}

TEST(SlidingWindowRetentionTest, RejectsZeroTokenStride) {
  EXPECT_DEATH((SlidingWindowRetention(/*token_stride=*/0,
                                       /*window_tokens=*/8,
                                       /*speculative_tokens=*/0)),
               "token_stride");
}

}  // namespace
}  // namespace xllm

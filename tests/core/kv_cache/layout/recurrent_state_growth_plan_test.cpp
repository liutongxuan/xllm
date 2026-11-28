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

#include "core/kv_cache/layout/recurrent_state_growth_plan.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(RecurrentStateGrowthPlanTest, InitialExecutionNeedsOneSlot) {
  for (const bool decode_step : {false, true}) {
    const RecurrentStateGrowthPlan plan(
        /*checkpoint_stride=*/4,
        /*enable_prefix_cache=*/true,
        RecurrentStateProgress{.cached_tokens = 12,
                               .decode_step = decode_step});
    EXPECT_FALSE(plan.checkpoint_index().has_value());
    EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 1u);
    EXPECT_FALSE(plan.can_advance_with(/*allocated_slots=*/0));
    EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/1));
  }
}

TEST(RecurrentStateGrowthPlanTest, PrivatePrefillRotatesAtUnpublishedBoundary) {
  const RecurrentStateGrowthPlan plan(
      /*checkpoint_stride=*/4,
      /*enable_prefix_cache=*/true,
      RecurrentStateProgress{.live_slot = RecurrentLiveSlot::PRIVATE,
                             .cached_tokens = 8,
                             .published_checkpoint_index = 1});
  ASSERT_EQ(plan.checkpoint_index(), 2u);
  EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 1u);
  EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/true), 0u);
  EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/0));
}

TEST(RecurrentStateGrowthPlanTest, SharedRestoreNeedsPrivateOutputOnCacheHit) {
  const RecurrentStateGrowthPlan plan(
      /*checkpoint_stride=*/4,
      /*enable_prefix_cache=*/true,
      RecurrentStateProgress{.live_slot = RecurrentLiveSlot::SHARED,
                             .cached_tokens = 8,
                             .published_checkpoint_index = 1});
  EXPECT_EQ(plan.checkpoint_index(), 2u);
  EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 1u);
  EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/true), 1u);
  EXPECT_FALSE(plan.can_advance_with(/*allocated_slots=*/0));
  EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/1));
}

TEST(RecurrentStateGrowthPlanTest,
     SharedRestoreNeedsPrivateOutputBetweenBoundaries) {
  const RecurrentStateGrowthPlan plan(
      /*checkpoint_stride=*/4,
      /*enable_prefix_cache=*/true,
      RecurrentStateProgress{.live_slot = RecurrentLiveSlot::SHARED,
                             .cached_tokens = 7});
  EXPECT_FALSE(plan.checkpoint_index().has_value());
  EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 1u);
  EXPECT_FALSE(plan.can_advance_with(/*allocated_slots=*/0));
}

TEST(RecurrentStateGrowthPlanTest,
     PartialAndPublishedBoundariesReusePrivateSlot) {
  for (const size_t tokens : {0u, 4u, 7u}) {
    const RecurrentStateGrowthPlan plan(
        /*checkpoint_stride=*/4,
        /*enable_prefix_cache=*/true,
        RecurrentStateProgress{.live_slot = RecurrentLiveSlot::PRIVATE,
                               .cached_tokens = tokens,
                               .published_checkpoint_index = 1});
    EXPECT_FALSE(plan.checkpoint_index().has_value());
    EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 0u);
    EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/0));
  }
}

TEST(RecurrentStateGrowthPlanTest, DecodeReusesLiveSlotAcrossBoundaries) {
  for (const RecurrentLiveSlot live_slot :
       {RecurrentLiveSlot::PRIVATE, RecurrentLiveSlot::SHARED}) {
    const RecurrentStateGrowthPlan plan(
        /*checkpoint_stride=*/4,
        /*enable_prefix_cache=*/true,
        RecurrentStateProgress{
            .live_slot = live_slot, .cached_tokens = 12, .decode_step = true});
    EXPECT_FALSE(plan.checkpoint_index().has_value());
    EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 0u);
    EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/0));
  }
}

TEST(RecurrentStateGrowthPlanTest, DisabledPrefixCacheReusesLiveSlot) {
  for (const RecurrentLiveSlot live_slot :
       {RecurrentLiveSlot::PRIVATE, RecurrentLiveSlot::SHARED}) {
    const RecurrentStateGrowthPlan plan(
        /*checkpoint_stride=*/4,
        /*enable_prefix_cache=*/false,
        RecurrentStateProgress{.live_slot = live_slot, .cached_tokens = 12});
    EXPECT_FALSE(plan.checkpoint_index().has_value());
    EXPECT_EQ(plan.additional_slots(/*checkpoint_cached=*/false), 0u);
    EXPECT_TRUE(plan.can_advance_with(/*allocated_slots=*/0));
  }
}

TEST(RecurrentStateGrowthPlanTest,
     WarmupDoesNotPublishButProtectsSharedSource) {
  const RecurrentStateGrowthPlan private_plan(
      /*checkpoint_stride=*/4,
      /*enable_prefix_cache=*/true,
      RecurrentStateProgress{.live_slot = RecurrentLiveSlot::PRIVATE,
                             .cached_tokens = 8,
                             .graph_warmup = true});
  EXPECT_FALSE(private_plan.checkpoint_index().has_value());
  EXPECT_EQ(private_plan.additional_slots(/*checkpoint_cached=*/false), 0u);
  const RecurrentStateGrowthPlan shared_plan(
      /*checkpoint_stride=*/4,
      /*enable_prefix_cache=*/true,
      RecurrentStateProgress{.live_slot = RecurrentLiveSlot::SHARED,
                             .cached_tokens = 8,
                             .graph_warmup = true});
  EXPECT_FALSE(shared_plan.checkpoint_index().has_value());
  EXPECT_EQ(shared_plan.additional_slots(/*checkpoint_cached=*/false), 1u);
  EXPECT_FALSE(shared_plan.can_advance_with(/*allocated_slots=*/0));
}

TEST(RecurrentStateGrowthPlanTest, CheckpointStrideIsIndependentPerPartition) {
  const RecurrentStateProgress progress{.live_slot = RecurrentLiveSlot::PRIVATE,
                                        .cached_tokens = 12,
                                        .published_checkpoint_index = 2};
  const RecurrentStateGrowthPlan first(/*checkpoint_stride=*/4,
                                       /*enable_prefix_cache=*/true,
                                       progress);
  const RecurrentStateGrowthPlan second(/*checkpoint_stride=*/6,
                                        /*enable_prefix_cache=*/true,
                                        progress);
  EXPECT_EQ(first.checkpoint_index(), 3u);
  EXPECT_EQ(first.additional_slots(/*checkpoint_cached=*/false), 1u);
  EXPECT_FALSE(second.checkpoint_index().has_value());
  EXPECT_EQ(second.additional_slots(/*checkpoint_cached=*/false), 0u);
}

TEST(RecurrentStateGrowthPlanTest, RejectsZeroCheckpointStride) {
  EXPECT_DEATH((RecurrentStateGrowthPlan(/*checkpoint_stride=*/0,
                                         /*enable_prefix_cache=*/true,
                                         RecurrentStateProgress{})),
               "checkpoint_stride");
}

}  // namespace
}  // namespace xllm

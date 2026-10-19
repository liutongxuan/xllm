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

#include "core/framework/allocator/vmm_memory_budget.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

TEST(VmmMemoryBudgetTest, PreserveConfiguredReserveBeforeWarmup) {
  VmmMemoryBudget budget;
  EXPECT_EQ(budget.headroom(/*free_bytes=*/600,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, ExcludePhysicalPoolGrowthFromExecutionWatermark) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/600, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/400, /*committed_bytes=*/400);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/200,
                            /*committed_bytes=*/600,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, DoNotCountResidentExecutionCachesTwice) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/300,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, ReserveObservedExecutionPeakAfterCacheRelease) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            250);
}

TEST(VmmMemoryBudgetTest, RetainLargestObservedExecutionFootprint) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/600, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/450, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/0),
            200);
}

TEST(VmmMemoryBudgetTest, PreserveWatermarkWhenIdlePhysicalPagesAreTrimmed) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/0,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, SaturateReserveWithoutUnsignedOverflow) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/0, /*committed_bytes=*/0);
  const size_t maximum = std::numeric_limits<size_t>::max();
  EXPECT_EQ(budget.headroom(maximum,
                            /*committed_bytes=*/1,
                            /*configured_reserve=*/1),
            maximum);
}

TEST(VmmMemoryBudgetTest, ResetDropsPreviousDeviceWatermark) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  budget.reset();
  EXPECT_EQ(budget.headroom(/*free_bytes=*/600,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest,
     PreserveLargerAllocatorPeakAfterOutsideMemoryRelease) {
  VmmMemoryBudget budget;
  budget.observe(600, 0, AllocatorMemoryStats{200, 200});
  EXPECT_EQ(budget.headroom(800, 0, 50, AllocatorMemoryStats{0, 200}), 250);
  // A later execution can need more workspace than the first execution.
  budget.observe(800, 0, AllocatorMemoryStats{400, 400});
  EXPECT_EQ(budget.headroom(1200, 0, 50, AllocatorMemoryStats{0, 400}), 450);
}

TEST(VmmMemoryBudgetTest, TrackAllocatorCacheAccumulationAcrossExecutions) {
  VmmMemoryBudget budget;
  budget.observe(800, 0, AllocatorMemoryStats{200, 200});
  budget.observe(600, 0, AllocatorMemoryStats{400, 400});
  EXPECT_EQ(budget.headroom(700, 0, 50, AllocatorMemoryStats{300, 400}), 150);
  EXPECT_EQ(budget.headroom(1000, 0, 50, AllocatorMemoryStats{0, 400}), 450);
}

TEST(VmmMemoryBudgetTest, DoNotChargeResidentAllocatorCachesTwice) {
  VmmMemoryBudget budget;
  budget.observe(600, 0, AllocatorMemoryStats{400, 400});
  EXPECT_EQ(budget.headroom(600, 0, 50, AllocatorMemoryStats{400, 400}), 50);
  EXPECT_EQ(budget.headroom(800, 0, 50, AllocatorMemoryStats{200, 400}), 250);
}

TEST(VmmMemoryBudgetTest, BackendPeakResetDoesNotLowerRetainedPeak) {
  VmmMemoryBudget budget;
  budget.observe(600, 0, AllocatorMemoryStats{400, 400});
  budget.observe(900, 0, AllocatorMemoryStats{100, 100});
  EXPECT_EQ(budget.headroom(1000, 0, 50, AllocatorMemoryStats{0, 0}), 450);
  // Peak observations on the admission path also survive backend resets.
  EXPECT_EQ(budget.headroom(500, 0, 50, AllocatorMemoryStats{500, 500}), 50);
  EXPECT_EQ(budget.headroom(1000, 0, 50, AllocatorMemoryStats{0, 0}), 550);
}

TEST(VmmMemoryBudgetTest, CurrentAllocatorResidencyIsALowerBoundOnPeak) {
  VmmMemoryBudget budget;
  EXPECT_EQ(budget.headroom(600, 0, 50, AllocatorMemoryStats{400, 100}), 50);
  EXPECT_EQ(budget.headroom(1000, 0, 50, AllocatorMemoryStats{0, 0}), 450);
}

TEST(VmmMemoryBudgetTest, OutsideMemoryReleaseDoesNotInflateAllocatorReserve) {
  VmmMemoryBudget budget;
  budget.observe(600, 200, AllocatorMemoryStats{200, 200});
  EXPECT_EQ(budget.headroom(800, 200, 50, AllocatorMemoryStats{0, 200}), 250);
  EXPECT_EQ(budget.headroom(1600, 200, 50, AllocatorMemoryStats{0, 200}), 250);
  // Missing statistics preserve the lifetime driver-watermark fallback.
  EXPECT_EQ(budget.headroom(1600, 200, 50), 1050);
}

TEST(VmmMemoryBudgetTest, SaturateAllocatorReserveWithoutUnsignedOverflow) {
  VmmMemoryBudget budget;
  const size_t maximum = std::numeric_limits<size_t>::max();
  EXPECT_EQ(budget.headroom(0, 0, 1, AllocatorMemoryStats{0, maximum}),
            maximum);
}

TEST(VmmMemoryBudgetTest, ResetDropsRetainedAllocatorPeak) {
  VmmMemoryBudget budget;
  budget.observe(600, 0, AllocatorMemoryStats{400, 400});
  budget.reset();
  EXPECT_EQ(budget.headroom(1000, 0, 50, AllocatorMemoryStats{0, 0}), 50);
}

}  // namespace
}  // namespace xllm

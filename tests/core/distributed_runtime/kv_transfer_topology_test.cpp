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

#include "core/distributed_runtime/kv_transfer_topology.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace xllm {
namespace {

TEST(KVTransferTopologyTest, PullIncludesEveryDeviceInSingleNodeGroup) {
  const auto routes = KVTransferTopology::get_pull_worker_routes(
      /*src_worker_count=*/4,
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      /*dst_worker_count=*/4,
      /*dst_dp_size=*/1,
      /*dst_dp_rank=*/0);
  ASSERT_TRUE(routes.has_value());
  ASSERT_EQ(routes->size(), 4u);
  for (size_t rank = 0; rank < routes->size(); ++rank) {
    EXPECT_EQ((*routes)[rank].src_rank, rank);
    EXPECT_EQ((*routes)[rank].dst_rank, rank);
  }
}

TEST(KVTransferTopologyTest, PullSelectsIndependentSourceAndDestinationDP) {
  const auto routes = KVTransferTopology::get_pull_worker_routes(
      /*src_worker_count=*/8,
      /*src_dp_size=*/2,
      /*src_dp_rank=*/1,
      /*dst_worker_count=*/8,
      /*dst_dp_size=*/2,
      /*dst_dp_rank=*/0);
  ASSERT_TRUE(routes.has_value());
  ASSERT_EQ(routes->size(), 4u);
  for (size_t rank = 0; rank < routes->size(); ++rank) {
    EXPECT_EQ((*routes)[rank].src_rank, rank + 4);
    EXPECT_EQ((*routes)[rank].dst_rank, rank);
  }
}

TEST(KVTransferTopologyTest, TransferUsesFullDPStrideAndIncludesCPWorkers) {
  // DP=2, CP=2, attention TP=2: each DP group contains four workers.
  const auto workers = KVTransferTopology::get_dp_worker_range(
      /*worker_count=*/8, /*dp_size=*/2, /*dp_rank=*/1);
  ASSERT_TRUE(workers.has_value());
  EXPECT_EQ(workers->begin, 4u);
  EXPECT_EQ(workers->count, 4u);

  const auto routes = KVTransferTopology::get_pull_worker_routes(
      /*src_worker_count=*/8,
      /*src_dp_size=*/2,
      /*src_dp_rank=*/0,
      /*dst_worker_count=*/8,
      /*dst_dp_size=*/2,
      /*dst_dp_rank=*/1);
  ASSERT_TRUE(routes.has_value());
  ASSERT_EQ(routes->size(), 4u);
  for (size_t rank = 0; rank < routes->size(); ++rank) {
    EXPECT_EQ((*routes)[rank].src_rank, rank);
    EXPECT_EQ((*routes)[rank].dst_rank, rank + 4);
  }
}

TEST(KVTransferTopologyTest, RejectsInvalidDPGroups) {
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/0, /*dp_size=*/1, /*dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/4, /*dp_size=*/0, /*dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/4, /*dp_size=*/-1, /*dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/4, /*dp_size=*/3, /*dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/4, /*dp_size=*/2, /*dp_rank=*/-1)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_dp_worker_range(
                   /*worker_count=*/4, /*dp_size=*/2, /*dp_rank=*/2)
                   .has_value());
}

TEST(KVTransferTopologyTest, PullRejectsHeterogeneousTopologies) {
  EXPECT_FALSE(KVTransferTopology::get_pull_worker_routes(
                   /*src_worker_count=*/4,
                   /*src_dp_size=*/1,
                   /*src_dp_rank=*/0,
                   /*dst_worker_count=*/4,
                   /*dst_dp_size=*/2,
                   /*dst_dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_pull_worker_routes(
                   /*src_worker_count=*/4,
                   /*src_dp_size=*/2,
                   /*src_dp_rank=*/0,
                   /*dst_worker_count=*/8,
                   /*dst_dp_size=*/2,
                   /*dst_dp_rank=*/0)
                   .has_value());
}

TEST(KVTransferTopologyTest, PullRejectsInvalidSourceAndDestinationRanks) {
  EXPECT_FALSE(KVTransferTopology::get_pull_worker_routes(
                   /*src_worker_count=*/4,
                   /*src_dp_size=*/2,
                   /*src_dp_rank=*/2,
                   /*dst_worker_count=*/4,
                   /*dst_dp_size=*/2,
                   /*dst_dp_rank=*/0)
                   .has_value());
  EXPECT_FALSE(KVTransferTopology::get_pull_worker_routes(
                   /*src_worker_count=*/4,
                   /*src_dp_size=*/2,
                   /*src_dp_rank=*/0,
                   /*dst_worker_count=*/4,
                   /*dst_dp_size=*/2,
                   /*dst_dp_rank=*/-1)
                   .has_value());
}

}  // namespace
}  // namespace xllm

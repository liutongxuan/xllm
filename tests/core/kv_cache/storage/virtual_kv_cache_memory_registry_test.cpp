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

#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/kv_cache/storage/virtual_kv_cache_tensor_allocator.h"
#include "core/platform/vmm_api.h"
#include "tests/npu_test_environment.h"

namespace xllm {
namespace {

constexpr size_t kPoolPages = 4;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kLatentSize = 512;
constexpr int64_t kRopeSize = 64;

void fill_block_pattern(const torch::Tensor& tensor,
                        int64_t block_id,
                        uint8_t pattern) {
  const size_t block_bytes =
      static_cast<size_t>(tensor.stride(/*dim=*/0)) * tensor.element_size();
  auto* block_ptr = static_cast<uint8_t*>(tensor.data_ptr()) +
                    static_cast<size_t>(block_id) * block_bytes;
  const std::vector<uint8_t> expected(block_bytes, pattern);
  ASSERT_EQ(aclrtMemcpy(block_ptr,
                        block_bytes,
                        expected.data(),
                        block_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
}

void expect_block_pattern(const torch::Tensor& tensor,
                          int64_t block_id,
                          uint8_t pattern) {
  SCOPED_TRACE(block_id);
  const size_t block_bytes =
      static_cast<size_t>(tensor.stride(/*dim=*/0)) * tensor.element_size();
  const auto* block_ptr = static_cast<const uint8_t*>(tensor.data_ptr()) +
                          static_cast<size_t>(block_id) * block_bytes;
  std::vector<uint8_t> actual(block_bytes);
  ASSERT_EQ(aclrtMemcpy(actual.data(),
                        block_bytes,
                        block_ptr,
                        block_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(actual, std::vector<uint8_t>(block_bytes, pattern));
}

class KVCacheMemoryEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("virtual_kv_cache_memory_registry_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new KVCacheMemoryEnvironment);

class VirtualKVCacheMemoryRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint32_t device_count = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
    if (device_count == 0) {
      GTEST_SKIP() << "No NPU device available";
    }
    ASSERT_EQ(aclrtSetDevice(/*device_id=*/0), ACL_SUCCESS);
    page_size_ = vmm::get_recommended_granularity(device_.index());
    ASSERT_GT(page_size_, 0);
    auto& pool = PhysicalPagePool::get_instance();
    pool.init(device_, pool_pages(), page_size_);
    resources_initialized_ = true;
    ASSERT_EQ(pool.num_available(), pool_pages());
    registry_.init(device_, page_size_);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    registry_.clear();
    auto& pool = PhysicalPagePool::get_instance();
    EXPECT_EQ(pool.num_available(), pool_pages());
    EXPECT_TRUE(pool.reset());
  }

  void create_model(const std::string& model_id, int64_t num_layers) {
    const std::vector<int64_t> dims{static_cast<int64_t>(page_size_)};
    registry_.create_k_tensors(model_id, dims, torch::kUInt8, num_layers);
    registry_.create_v_tensors(model_id, dims, torch::kUInt8, num_layers);
  }

  virtual size_t pool_pages() const { return kPoolPages; }

  const torch::Device device_{"npu:0"};
  VirtualKVCacheMemoryRegistry registry_;
  size_t page_size_ = 0;
  bool resources_initialized_ = false;
};

class VirtualMLAKVCacheMemoryRegistryTest final
    : public VirtualKVCacheMemoryRegistryTest {
 protected:
  size_t pool_pages() const override { return 18; }

  int64_t blocks_per_page() const {
    return static_cast<int64_t>(page_size_ /
                                (kBlockSize * kRopeSize * sizeof(uint16_t)));
  }

  std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
  create_mla_model(const std::string& model_id,
                   int64_t num_blocks,
                   int64_t num_layers) {
    return {registry_.create_k_tensors(model_id,
                                       {num_blocks, kBlockSize, 1, kLatentSize},
                                       torch::kBFloat16,
                                       num_layers),
            registry_.create_v_tensors(model_id,
                                       {num_blocks, kBlockSize, 1, kRopeSize},
                                       torch::kBFloat16,
                                       num_layers)};
  }
};

TEST_F(VirtualKVCacheMemoryRegistryTest,
       ModelsKeepIndependentMappingsAndMetadata) {
  create_model("first", /*num_layers=*/1);
  create_model("second", /*num_layers=*/1);
  ASSERT_TRUE(registry_.map("first", {0}));
  ASSERT_TRUE(registry_.map("second", {0}));
  const auto first_offsets = registry_.get_global_offsets_for_block(
      "first", /*layer_id=*/0, /*block_id=*/0, page_size_);
  const auto second_offsets = registry_.get_global_offsets_for_block(
      "second", /*layer_id=*/0, /*block_id=*/0, page_size_);
  EXPECT_NE(first_offsets.first, UINT64_MAX);
  EXPECT_NE(first_offsets.second, UINT64_MAX);
  EXPECT_NE(first_offsets.first, second_offsets.first);
  EXPECT_NE(first_offsets.second, second_offsets.second);
  EXPECT_EQ(registry_.num_layers("first"), 1);
  EXPECT_FALSE(registry_.num_layers("missing").has_value());

  registry_.erase("first");
  EXPECT_FALSE(registry_.num_layers("first").has_value());
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 2);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "second", /*layer_id=*/0, /*block_id=*/0, page_size_),
            second_offsets);
  EXPECT_FALSE(registry_.map("first", {0}));
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "first", /*layer_id=*/0, /*block_id=*/0, page_size_),
            std::make_pair(UINT64_MAX, UINT64_MAX));
}

TEST_F(VirtualKVCacheMemoryRegistryTest,
       ShortageDoesNotPartiallyMapAnotherModel) {
  create_model("busy", /*num_layers=*/1);
  create_model("large", /*num_layers=*/2);
  ASSERT_TRUE(registry_.map("busy", {0}));
  const auto busy_offsets = registry_.get_global_offsets_for_block(
      "busy", /*layer_id=*/0, /*block_id=*/0, page_size_);
  EXPECT_FALSE(registry_.map("large", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 2);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "busy", /*layer_id=*/0, /*block_id=*/0, page_size_),
            busy_offsets);
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    EXPECT_EQ(registry_.get_global_offsets_for_block(
                  "large", layer_id, /*block_id=*/0, page_size_),
              std::make_pair(UINT64_MAX, UINT64_MAX));
  }

  ASSERT_TRUE(registry_.unmap("busy", {0}));
  ASSERT_TRUE(registry_.map("large", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 0);
  registry_.clear();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), kPoolPages);
  registry_.init(device_, page_size_);
  create_model("fresh", /*num_layers=*/1);
  EXPECT_TRUE(registry_.map("fresh", {0}));
}

TEST_F(VirtualKVCacheMemoryRegistryTest,
       TensorAllocatorUsesItsInjectedRegistry) {
  auto allocator = create_virtual_kv_cache_tensor_allocator(
      registry_, "model", /*num_layers=*/2);
  const std::vector<int64_t> shape{static_cast<int64_t>(page_size_ / 2)};
  const torch::Tensor first_key = allocator->allocate(
      KVCacheTensorRole::KEY, shape, torch::kBFloat16, device_);
  const torch::Tensor second_key = allocator->allocate(
      KVCacheTensorRole::KEY, shape, torch::kBFloat16, device_);
  const torch::Tensor first_value = allocator->allocate(
      KVCacheTensorRole::VALUE, shape, torch::kBFloat16, device_);
  const torch::Tensor second_value = allocator->allocate(
      KVCacheTensorRole::VALUE, shape, torch::kBFloat16, device_);
  EXPECT_NE(first_key.data_ptr(), second_key.data_ptr());
  EXPECT_NE(first_value.data_ptr(), second_value.data_ptr());
  EXPECT_EQ(first_key.sizes().vec(), shape);
  EXPECT_EQ(first_value.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(first_value.device(), device_);
  EXPECT_EQ(registry_.num_layers("model"), 2);
  EXPECT_TRUE(registry_.map("model", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 0);
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       LatentAndRopeBoundariesKeepIndependentMappings) {
  ASSERT_EQ(page_size_ % (kBlockSize * kLatentSize * sizeof(uint16_t)), 0);
  const int64_t blocks_per_key_page = static_cast<int64_t>(
      page_size_ / (kBlockSize * kLatentSize * sizeof(uint16_t)));
  ASSERT_GT(blocks_per_key_page, 1);
  const int64_t blocks_per_group = blocks_per_page();
  auto [keys, values] =
      create_mla_model("mla", 2 * blocks_per_group, /*num_layers=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  const offset_t next_offset = static_cast<offset_t>(page_size_);
  const offset_t invalid_offset = static_cast<offset_t>(2 * page_size_);
  EXPECT_FALSE(registry_.map("mla", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  ASSERT_TRUE(registry_.map("mla", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  ASSERT_TRUE(registry_.map("mla", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);

  const std::vector<int64_t> blocks = {0,
                                       blocks_per_key_page - 1,
                                       blocks_per_key_page,
                                       blocks_per_group - 1,
                                       blocks_per_group,
                                       2 * blocks_per_group - 1};
  for (size_t i = 0; i < blocks.size(); ++i) {
    fill_block_pattern(keys.front(), blocks[i], static_cast<uint8_t>(20 + i));
    fill_block_pattern(
        values.front(), blocks[i], static_cast<uint8_t>(100 + i));
  }
  EXPECT_FALSE(registry_.unmap("mla", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t i = 0; i < blocks.size(); ++i) {
    expect_block_pattern(keys.front(), blocks[i], static_cast<uint8_t>(20 + i));
    expect_block_pattern(
        values.front(), blocks[i], static_cast<uint8_t>(100 + i));
  }
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "mla", /*layer_id=*/0, /*block_id=*/0, page_size_),
            std::make_pair(UINT64_MAX, UINT64_MAX));

  ASSERT_TRUE(registry_.unmap("mla", {0}));
  EXPECT_EQ(pool.num_available(), 9);
  expect_block_pattern(keys.front(), blocks_per_group, /*pattern=*/24);
  expect_block_pattern(values.front(), blocks_per_group, /*pattern=*/104);
  expect_block_pattern(keys.front(), 2 * blocks_per_group - 1, /*pattern=*/25);
  expect_block_pattern(
      values.front(), 2 * blocks_per_group - 1, /*pattern=*/105);
  ASSERT_TRUE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 0);
  fill_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/50);
  fill_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/150);
  expect_block_pattern(keys.front(), blocks_per_group, /*pattern=*/24);
  expect_block_pattern(values.front(), blocks_per_group, /*pattern=*/104);
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       ShortageDoesNotPartiallyMapLatentRopeOrAnotherLayer) {
  create_model("busy", /*num_layers=*/1);
  auto [keys, values] =
      create_mla_model("mla", blocks_per_page(), /*num_layers=*/2);
  ASSERT_TRUE(registry_.map("busy", {0}));
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_EQ(pool.num_available(), 16);
  const auto busy_offsets = registry_.get_global_offsets_for_block(
      "busy", /*layer_id=*/0, /*block_id=*/0, page_size_);

  EXPECT_FALSE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 16);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "busy", /*layer_id=*/0, /*block_id=*/0, page_size_),
            busy_offsets);
  ASSERT_TRUE(registry_.unmap("mla", {0}));
  EXPECT_EQ(pool.num_available(), 16);
  ASSERT_TRUE(registry_.unmap("busy", {0}));
  ASSERT_TRUE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    fill_block_pattern(
        keys[layer], blocks_per_page() - 1, static_cast<uint8_t>(30 + layer));
    fill_block_pattern(values[layer],
                       blocks_per_page() - 1,
                       static_cast<uint8_t>(130 + layer));
  }
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    expect_block_pattern(
        keys[layer], blocks_per_page() - 1, static_cast<uint8_t>(30 + layer));
    expect_block_pattern(values[layer],
                         blocks_per_page() - 1,
                         static_cast<uint8_t>(130 + layer));
  }
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       IncompleteLogicalPagesDoNotChangeExistingMappings) {
  ASSERT_GT(blocks_per_page(), 1);
  auto [keys, values] =
      create_mla_model("mla", 2 * blocks_per_page() - 1, /*num_layers=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  const offset_t partial_offset = static_cast<offset_t>(page_size_);
  EXPECT_FALSE(registry_.map("mla", {0, partial_offset}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  ASSERT_TRUE(registry_.map("mla", {0}));
  fill_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/51);
  fill_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/151);
  EXPECT_FALSE(registry_.map("mla", {partial_offset}));
  EXPECT_FALSE(registry_.unmap("mla", {0, partial_offset}));
  EXPECT_EQ(pool.num_available(), 9);
  expect_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/51);
  expect_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/151);
}

}  // namespace
}  // namespace xllm

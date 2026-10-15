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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_shared_memory_manager.h"

namespace xllm {
namespace {

class ForwardSharedMemoryOutputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& execution_config = ExecutionConfig::get_instance();
    previous_contiguous_ = execution_config.use_contiguous_input_buffer();
    execution_config.use_contiguous_input_buffer(false);
    previous_eplb_ = EPLBConfig::get_instance().enable_eplb();
    EPLBConfig::get_instance().enable_eplb(false);
    const std::string prefix =
        "forward_output_" + std::to_string(getpid()) + "_" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    const std::string name = ForwardSharedMemoryManager::create_unique_name(
        prefix, /*dp_group=*/0, ForwardType::RAW_OUTPUT, /*rank=*/0);
    bool is_creator = false;
    writer_ = std::make_unique<ForwardSharedMemoryManager>(
        name, kSegmentBytes, is_creator, ForwardType::RAW_OUTPUT);
    ASSERT_TRUE(is_creator);
    reader_ = std::make_unique<ForwardSharedMemoryManager>(
        name, kSegmentBytes, is_creator, ForwardType::RAW_OUTPUT);
    ASSERT_FALSE(is_creator);
  }

  void TearDown() override {
    reader_.reset();
    writer_.reset();
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
    EPLBConfig::get_instance().enable_eplb(previous_eplb_);
  }

  uint64_t published_version() const {
    return static_cast<const ControlMetadata*>(writer_->base_address())
        ->version;
  }

  static constexpr size_t kSegmentBytes = 4096;
  bool previous_contiguous_ = false;
  bool previous_eplb_ = false;
  std::unique_ptr<ForwardSharedMemoryManager> writer_;
  std::unique_ptr<ForwardSharedMemoryManager> reader_;
};

TEST_F(ForwardSharedMemoryOutputTest,
       LlmPreservesSpeculativeTokensAndJsonErrors) {
  ForwardOutput source;
  source.sample_output.next_tokens =
      torch::tensor({{10, 11, -1}, {20, 21, 22}}, torch::kInt64);
  source.sample_output.logprobs =
      torch::tensor({{-0.1f, -0.2f, -0.3f}, {-0.4f, -0.5f, -0.6f}});
  source.sample_output.embeddings =
      torch::arange(12, torch::kFloat32).reshape({2, 3, 2});
  source.json_object_errors = {{"request#0", "invalid prior token"}};
  ASSERT_TRUE(writer_->llm_output_write(source, {{1, 3}, {2, 3}}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 2);
  ASSERT_EQ(output.outputs[0].tokens.size(), 2);
  EXPECT_EQ(output.outputs[0].tokens[1].id, 11);
  EXPECT_FLOAT_EQ(output.outputs[0].tokens[1].logprob.value(), -0.2f);
  EXPECT_EQ(output.outputs[0].tokens[1].embeddings,
            (std::vector<float>{2.0f, 3.0f}));
  EXPECT_EQ(output.outputs[1].tokens.size(), 3);
  EXPECT_EQ(output.outputs[0].speculative_token_stats.accepted_tokens, 1);
  EXPECT_EQ(output.outputs[1].speculative_token_stats.proposed_tokens, 3);
  ASSERT_EQ(output.json_object_errors.size(), 1);
  EXPECT_EQ(output.json_object_errors[0].sample_sequence_id, "request#0");
  EXPECT_EQ(output.json_object_errors[0].message, "invalid prior token");
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmPreservesMultimodalOutputAndTopTokens) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  source.sample_output.logprobs = torch::tensor({-0.5f});
  source.sample_output.top_tokens = torch::tensor({{7, 9}}, torch::kInt64);
  source.sample_output.top_logprobs = torch::tensor({{-0.5f, -0.8f}});
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
  EXPECT_EQ(output.outputs[0].tokens[0].top_tokens,
            (std::vector<int64_t>{7, 9}));
  EXPECT_EQ(output.outputs[0].tokens[0].top_logprobs,
            (std::vector<float>{-0.5f, -0.8f}));
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest, RecPreservesBeamOutput) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7, 8}, torch::kInt32);
  source.beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
  source.beam_search_output.out_logprobs = torch::tensor({-0.2f, -0.3f});
  ASSERT_TRUE(writer_->rec_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->rec_output_read(output));
  EXPECT_EQ(output.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(output.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(output.out_logprobs, (std::vector<float>{-0.2f, -0.3f}));
  ASSERT_EQ(output.outputs.size(), 2);
  EXPECT_EQ(output.outputs[1].tokens[0].id, 8);
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmPreservesSpeculativeTokensWithMultimodalOutput) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({{7, 8, -1}}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {{1, 3}}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 2);
  EXPECT_EQ(output.outputs[0].tokens[1].id, 8);
  EXPECT_EQ(output.outputs[0].speculative_token_stats.proposed_tokens, 3);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmEmptySpeculativeColumnsPreserveMultimodalRows) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({1, 0}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_TRUE(output.outputs[0].tokens.empty());
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmEmptyTokenVectorPreservesMultimodalRows) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({0}, torch::kInt32);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, -1);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       ZeroSpeculativeRowsRejectMultimodalRowsWithoutPublishing) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({0, 2}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f})}};
  EXPECT_FALSE(writer_->llm_output_write(source, {}));
  EXPECT_FALSE(writer_->vlm_output_write(source, {}));
  EXPECT_FALSE(writer_->rec_output_write(source, {}));
  EXPECT_EQ(published_version(), 0);
}

TEST_F(ForwardSharedMemoryOutputTest, RecBeamOnlyOutputNeedsNoSampleRows) {
  ForwardOutput source;
  source.beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
  source.beam_search_output.out_logprobs = torch::tensor({-0.2f, -0.3f});
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->rec_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_EQ(output.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(output.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(output.out_logprobs, (std::vector<float>{-0.2f, -0.3f}));
}

TEST_F(ForwardSharedMemoryOutputTest, EmptyTokenOutputClearsPreviousPayload) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_FALSE(output.outputs.empty());
  ASSERT_TRUE(writer_->llm_output_write(ForwardOutput(), {}));
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_TRUE(output.json_object_errors.empty());
  EXPECT_TRUE(output.expert_load_data.empty());
  EXPECT_EQ(output.prepared_token, -1);
}

TEST_F(ForwardSharedMemoryOutputTest, DiTPreservesTextOnlyOutput) {
  DiTForwardOutput source;
  source.text_output = {"diffusion text", ""};
  ASSERT_TRUE(writer_->dit_output_write(source));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_EQ(output.dit_forward_output.text_output, source.text_output);
}

TEST_F(ForwardSharedMemoryOutputTest,
       LlmAllowsMissingExpertDataWhenEplbIsEnabled) {
  EPLBConfig::get_instance().enable_eplb(true);
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  source.prepared_token = 81;
  ASSERT_TRUE(writer_->llm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.expert_load_data.empty());
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest, VlmPreservesEplbExpertLoads) {
  EPLBConfig::get_instance().enable_eplb(true);
  ForwardOutput source;
  source.expert_load_data = torch::tensor({{1, 2}, {3, 4}}, torch::kInt64);
  source.prepared_token = 19;
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  EXPECT_EQ(output.expert_load_data, (std::vector<int64_t>{1, 2, 3, 4}));
  EXPECT_EQ(output.prepared_token, 19);
}

TEST_F(ForwardSharedMemoryOutputTest, TokenReadsClearPreviousDiTAndBeamData) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"old text"};
  output.dit_forward_output.tensors = {torch::ones({2})};
  output.src_seq_idxes = {1};
  output.out_tokens = {3};
  output.out_logprobs = {-0.5f};
  output.json_object_errors = {{"old", "old error"}};
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.dit_forward_output.text_output.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_TRUE(output.src_seq_idxes.empty());
  EXPECT_TRUE(output.out_tokens.empty());
  EXPECT_TRUE(output.out_logprobs.empty());
  EXPECT_TRUE(output.json_object_errors.empty());
}

TEST_F(ForwardSharedMemoryOutputTest, EmptyDiTReadClearsPreviousOutput) {
  DiTForwardOutput source;
  source.text_output = {"first output"};
  source.tensors = {torch::ones({2})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  ASSERT_FALSE(output.dit_forward_output.tensors.empty());
  ASSERT_TRUE(writer_->dit_output_write(DiTForwardOutput()));
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_TRUE(output.dit_forward_output.text_output.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
}

TEST_F(ForwardSharedMemoryOutputTest,
       DiTTensorsSurviveOverwriteAndSegmentRelease) {
  const torch::Tensor expected = torch::tensor({1.0f, 2.0f, 3.0f});
  DiTForwardOutput source;
  source.tensors = {expected};
  source.text_output = {"first"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  ASSERT_EQ(output.dit_forward_output.tensors.size(), 1);
  source.tensors = {torch::zeros({3})};
  source.text_output = {"other"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  reader_.reset();
  writer_.reset();
  EXPECT_TRUE(torch::equal(output.dit_forward_output.tensors[0], expected));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"first"}));
}

TEST_F(ForwardSharedMemoryOutputTest, VlmTensorsSurviveSubsequentOutputWrites) {
  const torch::Tensor expected = torch::tensor({1.0f, 2.0f});
  ForwardOutput source;
  source.sample_output.mm_embeddings = {{expected}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  source.sample_output.mm_embeddings = {{torch::zeros({2})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0], expected));
}

TEST_F(ForwardSharedMemoryOutputTest,
       InvalidTokenWritesPreservePublishedOutput) {
  ForwardOutput valid;
  valid.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(valid, {}));
  const uint64_t version = published_version();
  ForwardOutput invalid = valid;
  invalid.sample_output.logprobs = torch::tensor({-0.5f, -0.2f});
  EXPECT_FALSE(writer_->llm_output_write(invalid, {}));
  invalid = valid;
  invalid.sample_output.mm_embeddings = {{torch::Tensor()}};
  EXPECT_FALSE(writer_->vlm_output_write(invalid, {}));
  invalid = valid;
  invalid.beam_search_output.out_tokens = torch::tensor({8}, torch::kInt32);
  EXPECT_FALSE(writer_->rec_output_write(invalid, {}));
  EXPECT_EQ(published_version(), version);
  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
}

TEST_F(ForwardSharedMemoryOutputTest,
       OverflowDiTWritePreservesPublishedOutput) {
  DiTForwardOutput valid;
  valid.text_output = {"first"};
  ASSERT_TRUE(writer_->dit_output_write(valid));
  const uint64_t version = published_version();
  DiTForwardOutput overflow;
  overflow.text_output = {std::string(kSegmentBytes, 'x')};
  EXPECT_FALSE(writer_->dit_output_write(overflow));
  EXPECT_EQ(published_version(), version);
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output, valid.text_output);
}

TEST_F(ForwardSharedMemoryOutputTest, InvalidDiTTensorDoesNotPublishOutput) {
  DiTForwardOutput source;
  source.tensors = {torch::Tensor()};
  EXPECT_FALSE(writer_->dit_output_write(source));
  source.tensors = {torch::tensor(1.0f)};
  EXPECT_FALSE(writer_->dit_output_write(source));
  EXPECT_EQ(published_version(), 0);
}

TEST_F(ForwardSharedMemoryOutputTest,
       DiTReaderRejectsTokenOutputWithoutMutation) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"previous"};
  EXPECT_FALSE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"previous"}));
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenReaderRejectsDiTOutputWithoutMutation) {
  DiTForwardOutput source;
  source.text_output = {"diffusion"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  output.prepared_token = 81;
  EXPECT_FALSE(reader_->vlm_output_read(output));
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest, MalformedCountFailsWithoutMutation) {
  ForwardOutput source;
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  const uint64_t invalid_count = std::numeric_limits<uint64_t>::max();
  char* payload =
      static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
  std::memcpy(payload, &invalid_count, sizeof(invalid_count));
  RawForwardOutput output;
  output.prepared_token = 81;
  EXPECT_FALSE(reader_->rec_output_read(output));
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest,
       MalformedReadAllowsSubsequentOutputToBeConsumed) {
  ASSERT_TRUE(writer_->llm_output_write(ForwardOutput(), {}));
  const uint64_t invalid_count = std::numeric_limits<uint64_t>::max();
  char* payload =
      static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
  std::memcpy(payload, &invalid_count, sizeof(invalid_count));
  RawForwardOutput output;
  EXPECT_FALSE(reader_->llm_output_read(output));
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
}

TEST_F(ForwardSharedMemoryOutputTest, MalformedTensorSizeFailsWithoutMutation) {
  DiTForwardOutput source;
  source.tensors = {torch::ones({2})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  // RAW_OUTPUT framing, tensor vector count, rank, shape, and dtype precede
  // the tensor byte count. Corrupt only that count in an otherwise valid frame.
  constexpr size_t kTensorBytesOffset =
      sizeof(ControlMetadata) + 6 * sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(bool) + sizeof(int32_t) + sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(int8_t);
  char* byte_count =
      static_cast<char*>(writer_->base_address()) + kTensorBytesOffset;
  const uint64_t invalid_bytes = kSegmentBytes;
  std::memcpy(byte_count, &invalid_bytes, sizeof(invalid_bytes));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"previous"};
  EXPECT_FALSE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"previous"}));
}

}  // namespace
}  // namespace xllm

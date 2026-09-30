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

#include "rec_scheduler.h"

#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <utility>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/request/rec_type.h"

namespace xllm {

namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    (void)text;
    (void)ids;
    (void)add_special_tokens;
    return false;
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    (void)ids;
    (void)skip_special_tokens;
    return "";
  }
  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    (void)token;
    return std::nullopt;
  }
  std::string id_to_token(int32_t id) const override {
    (void)id;
    return "";
  }
  size_t vocab_size() const override { return 0; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class ControllablePrefetchPool final : public BlockManagerPool {
 public:
  ControllablePrefetchPool(const Options& options, int32_t dp_size)
      : BlockManagerPool(options, dp_size) {}

  bool has_storage_prefetch() const override { return prefetch_enabled_; }

  void prefetch_from_storage(std::shared_ptr<Request> request,
                             PrefetchDoneCallback done) override {
    ++prefetch_calls_;
    if (prefetch_ready_) {
      done(std::move(request));
      return;
    }
    pending_.emplace_back(std::move(request), std::move(done));
  }

  void enable_prefetch() { prefetch_enabled_ = true; }
  void set_prefetch_ready(bool ready) {
    prefetch_ready_ = ready;
    if (!ready) {
      return;
    }
    auto pending = std::move(pending_);
    pending_.clear();
    for (auto& [request, done] : pending) {
      done(std::move(request));
    }
  }
  size_t prefetch_calls() const { return prefetch_calls_; }

 private:
  bool prefetch_enabled_ = false;
  bool prefetch_ready_ = true;
  size_t prefetch_calls_ = 0;
  std::vector<std::pair<std::shared_ptr<Request>, PrefetchDoneCallback>>
      pending_;
};

class FakeRecEngine final {
 public:
  explicit FakeRecEngine(
      int32_t num_blocks,
      int32_t block_size,
      int32_t dp_size = 1,
      RecExecutionConfig config = RecExecutionConfig(BatchInputType::ONEREC))
      : config_(std::move(config)) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = false;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ =
        std::make_unique<ControllablePrefetchPool>(opt, dp_size);
  }
  ForwardOutput step(RecBatchGroup& batch) {
    step_calls_.fetch_add(1, std::memory_order_relaxed);
    if (step_callback_) {
      step_callback_(batch);
    }
    if (finish_batches_) {
      for (RecBatch& rec_batch : batch) {
        rec_batch.finish();
      }
    }
    return ForwardOutput();
  }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  const RecExecutionConfig& execution_config() const { return config_; }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const {
    static ModelArgs args;
    return args;
  }
  const TokenizerArgs& tokenizer_args() const {
    static TokenizerArgs args;
    return args;
  }
  std::vector<int64_t> get_active_activation_memory() const { return {}; }
  int32_t step_calls() const {
    return step_calls_.load(std::memory_order_relaxed);
  }
  void finish_batches() { finish_batches_ = true; }
  void set_step_callback(std::function<void(RecBatchGroup&)> callback) {
    step_callback_ = std::move(callback);
  }
  void enable_prefetch() { fake_block_manager_->enable_prefetch(); }
  void set_prefetch_ready(bool ready) {
    fake_block_manager_->set_prefetch_ready(ready);
  }
  size_t prefetch_calls() const {
    return fake_block_manager_->prefetch_calls();
  }

 private:
  const RecExecutionConfig config_;
  std::atomic<int32_t> step_calls_{0};
  bool finish_batches_ = false;
  std::function<void(RecBatchGroup&)> step_callback_;
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<ControllablePrefetchPool> fake_block_manager_;
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

RecScheduler::Options CreateOptions(int32_t max_tokens_per_batch = 10000,
                                    int32_t max_seqs_per_batch = 256,
                                    int32_t dp_size = 1,
                                    bool enable_schedule_overlap = false,
                                    int32_t rec_worker_max_concurrency = 1) {
  RecScheduler::Options opt;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.enable_schedule_overlap_ = enable_schedule_overlap;
  opt.disable_log_stats_ = true;
  opt.rec_worker_max_concurrency_ = rec_worker_max_concurrency;
  return opt;
}

std::vector<std::shared_ptr<Request>> GenRequests(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    RecType rec_type,
    int32_t max_context_len = 30000) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_EQ(prompt_lens.size(), max_tokens.size());
  requests.reserve(prompt_lens.size());
  for (size_t i = 0; i < prompt_lens.size(); ++i) {
    std::vector<int32_t> prompt_token_ids(prompt_lens[i], 0);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = false;
    scheduler_param.priority = RequestPriority::NORMAL;
    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);
    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           static_cast<size_t>(prompt_lens[i]) + 30000,
                           1,
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);
    req_state.rec_type = rec_type;
    req_state.output_func = [](const RequestOutput& /*output*/) {
      return true;
    };
    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }
  return requests;
}

class TestableRecScheduler final : public RecScheduler {
 public:
  using RecScheduler::RecScheduler;

  RecBatchGroup prepare_batch_test() { return prepare_rec_batch(); }

  const std::vector<std::shared_ptr<Request>>& get_running_requests() const {
    return running_requests_;
  }
};

}  // namespace

TEST(RecSchedulerTest, AddRequestSuccess) {
  auto engine = std::make_unique<FakeRecEngine>(32, 32);
  auto opt = CreateOptions();
  RecScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64}, {10}, RecType::kOneRec);
  std::shared_ptr<Request> req = requests[0];
  EXPECT_TRUE(scheduler.add_request(req));
}

TEST(RecSchedulerTest, PrepareBatchEmptyWhenNoRequests) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  auto engine = std::make_unique<FakeRecEngine>(32, 32);
  auto opt = CreateOptions();
  TestableRecScheduler scheduler(engine.get(), opt);
  RecBatchGroup batches = scheduler.prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  EXPECT_TRUE(batches[0].empty());
}

TEST(RecSchedulerTest, PrepareBatchOneRecSchedulesRequest) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeRecEngine>(64, 32);
  auto opt = CreateOptions(10000, 256);
  TestableRecScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64, 64}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  RecBatchGroup batches = scheduler.prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  bool has_non_empty = false;
  for (const auto& b : batches) {
    if (!b.empty()) {
      has_non_empty = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_empty);
  EXPECT_EQ(scheduler.get_running_requests().size(), 2u);
  EXPECT_EQ(batches.front().input_type(), BatchInputType::ONEREC);
  EXPECT_TRUE(batches.front().uses_group_input());
}

TEST(RecSchedulerTest, PrepareBatchRespectsTokenBudget) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeRecEngine>(64, 32);
  auto opt = CreateOptions(50, 1);
  TestableRecScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({40, 40}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  scheduler.prepare_batch_test();
  EXPECT_LE(scheduler.get_running_requests().size(), 1u);
}

TEST(RecSchedulerTest, StepCompletesWithRequest) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeRecEngine>(64, 32);
  auto opt = CreateOptions(10000, 256);
  RecScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  scheduler.add_request(requests[0]);
  EXPECT_NO_THROW(scheduler.step(absl::Milliseconds(500)));
  EXPECT_EQ(engine->step_calls(), 1);
}

TEST(RecSchedulerTest, GenerateCompletesRecRequestsAndResponses) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeRecEngine>(64, 32);
  engine->finish_batches();
  auto opt = CreateOptions(10000, 256);
  RecScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  std::promise<RequestOutput> response;
  std::future<RequestOutput> response_future = response.get_future();
  requests[0]->state().output_func = [&response](const RequestOutput& output) {
    response.set_value(output);
    return true;
  };
  ASSERT_TRUE(scheduler.add_request(requests[0]));

  scheduler.generate();

  EXPECT_EQ(engine->step_calls(), 1);
  EXPECT_TRUE(requests[0]->finished());
  ASSERT_EQ(response_future.wait_for(std::chrono::milliseconds(0)),
            std::future_status::ready);
  const RequestOutput output = response_future.get();
  EXPECT_TRUE(output.finished);
  ASSERT_TRUE(output.status.has_value());
  EXPECT_TRUE(output.status->ok());
}

TEST(RecSchedulerTest, RejectsRequestsWithoutRecDomain) {
  FakeRecEngine engine(/*num_blocks=*/32, /*block_size=*/32);
  RecScheduler scheduler(&engine, CreateOptions());
  auto requests = GenRequests({8}, {4}, RecType::kNone);

  EXPECT_FALSE(scheduler.add_request(requests.front()));
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(RecSchedulerTest, RejectsRequestsForAnotherRecModelKind) {
  const RecExecutionConfig contracts[] = {
      RecExecutionConfig(BatchInputType::SEQUENCE),
      RecExecutionConfig(BatchInputType::ONEREC),
  };
  for (const auto& config : contracts) {
    FakeRecEngine engine(
        /*num_blocks=*/32, /*block_size=*/4, /*dp_size=*/1, config);
    RecScheduler scheduler(&engine, CreateOptions());
    const RecType other_kind = config.rec_type() == RecType::kOneRec
                                   ? RecType::kLlmRec
                                   : RecType::kOneRec;
    auto requests = GenRequests({8}, {4}, other_kind);
    EXPECT_FALSE(scheduler.add_request(requests.front()));
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
    EXPECT_FALSE(scheduler.has_pending_prefetch());
    EXPECT_EQ(engine.prefetch_calls(), 0u);
  }
}

TEST(RecSchedulerTest, SelectsContractBeforeRequestsAndKeepsItsSnapshot) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  const RecExecutionConfig contracts[] = {
      RecExecutionConfig(BatchInputType::SEQUENCE),
      RecExecutionConfig(BatchInputType::ONEREC),
      *RecExecutionConfig::resolve(
          "qwen3", /*decode_rounds=*/3, /*enable_prefill_only=*/false),
      *RecExecutionConfig::resolve(
          "onerec", /*decode_rounds=*/4, /*enable_prefill_only=*/false),
  };
  for (const auto& config : contracts) {
    FakeRecEngine engine(
        /*num_blocks=*/64, /*block_size=*/4, /*dp_size=*/1, config);
    TestableRecScheduler scheduler(&engine, CreateOptions());
    RecBatchGroup empty = scheduler.prepare_batch_test();
    ASSERT_EQ(empty.size(), 1u);
    EXPECT_TRUE(empty.front().empty());
    EXPECT_EQ(empty.front().execution_config(), config);
    RecConfig::get_instance().max_decode_rounds(config.is_multi_round() ? 0
                                                                        : 9);
    auto requests = GenRequests({8}, {2}, config.rec_type());
    ASSERT_TRUE(scheduler.add_request(requests.front()));
    RecBatchGroup batches = scheduler.prepare_batch_test();
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front().execution_config(), config);
    EXPECT_EQ(batches.front().input_type(), config.input_type());
    EXPECT_EQ(batches.front().uses_group_input(), config.uses_group_input());
  }
}

TEST(RecSchedulerTest, XAttentionKvAllocationUsesConfiguredRoundsSnapshot) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  const auto config = RecExecutionConfig::resolve(
      "onerec", /*decode_rounds=*/6, /*enable_prefill_only=*/false);
  ASSERT_TRUE(config.has_value());
  FakeRecEngine engine(
      /*num_blocks=*/64, /*block_size=*/4, /*dp_size=*/1, config.value());
  TestableRecScheduler scheduler(&engine, CreateOptions());
  RecConfig::get_instance().max_decode_rounds(100);
  auto requests = GenRequests({8}, {2}, RecType::kOneRec);
  Sequence* sequence = requests.front()->sequences().front().get();
  ASSERT_TRUE(scheduler.add_request(requests.front()));
  RecBatchGroup batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front().input_type(), BatchInputType::ONEREC_XATTENTION);
  // One decoder BOS + six rounds require two blocks of four tokens.
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 2u);
}

TEST(RecSchedulerTest, CachedLlmRecRequestsPreserveRemainingComputeBudget) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  FakeRecEngine engine(/*num_blocks=*/64,
                       /*block_size=*/4,
                       /*dp_size=*/1,
                       RecExecutionConfig(BatchInputType::SEQUENCE));
  TestableRecScheduler scheduler(&engine,
                                 CreateOptions(/*max_tokens_per_batch=*/2));
  auto requests = GenRequests({8}, {4}, RecType::kLlmRec);
  Sequence* sequence = requests.front()->sequences().front().get();
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(sequence, /*num_tokens=*/12));
  sequence->kv_state().set_kv_cache_tokens_num(6);
  ASSERT_TRUE(scheduler.add_request(requests.front()));

  RecBatchGroup batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front().sequence(0), sequence);
  EXPECT_EQ(batches.front().input_type(), BatchInputType::SEQUENCE);
  EXPECT_FALSE(batches.front().uses_group_input());
  EXPECT_EQ(batches.front().get_allowed_max_tokens(),
            std::vector<uint32_t>({2}));
  EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 6u);
  EXPECT_EQ(scheduler.get_running_requests().size(), 1u);
}

TEST(RecSchedulerTest, LlmRecSchedulesAcrossConfiguredDataParallelRanks) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  FakeRecEngine engine(/*num_blocks=*/64,
                       /*block_size=*/4,
                       /*dp_size=*/2,
                       RecExecutionConfig(BatchInputType::SEQUENCE));
  TestableRecScheduler scheduler(&engine,
                                 CreateOptions(/*max_tokens_per_batch=*/64,
                                               /*max_seqs_per_batch=*/4,
                                               /*dp_size=*/2));
  auto requests = GenRequests({8, 8}, {4, 4}, RecType::kLlmRec);
  for (std::shared_ptr<Request>& request : requests) {
    ASSERT_TRUE(scheduler.add_request(request));
  }

  RecBatchGroup batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 2u);
  EXPECT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[1].size(), 1u);
  EXPECT_EQ(requests[0]->sequences().front()->dp_rank(), 0);
  EXPECT_EQ(requests[1]->sequences().front()->dp_rank(), 1);
}

TEST(RecSchedulerTest, QueueCapacityIncludesPendingStoragePrefetch) {
  FakeRecEngine engine(/*num_blocks=*/64, /*block_size=*/4);
  engine.enable_prefetch();
  engine.set_prefetch_ready(false);
  auto options = CreateOptions();
  options.request_queue_size(1);
  TestableRecScheduler scheduler(&engine, options);
  auto requests = GenRequests({8, 8}, {4, 4}, RecType::kOneRec);

  ASSERT_TRUE(scheduler.add_request(requests[0]));
  EXPECT_EQ(engine.prefetch_calls(), 0u);
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
  EXPECT_FALSE(scheduler.add_request(requests[1]));
  EXPECT_TRUE(scheduler.has_pending_prefetch());
  EXPECT_TRUE(scheduler.prepare_batch_test().front().empty());
  EXPECT_EQ(engine.prefetch_calls(), 1u);

  engine.set_prefetch_ready(true);
  EXPECT_TRUE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
  RecBatchGroup batches = scheduler.prepare_batch_test();

  EXPECT_FALSE(batches.front().empty());
  EXPECT_FALSE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  EXPECT_TRUE(scheduler.add_request(requests[1]));
}

TEST(RecSchedulerTest, CancelledPrefetchReleasesAdmissionCapacity) {
  FakeRecEngine engine(/*num_blocks=*/64, /*block_size=*/4);
  engine.enable_prefetch();
  engine.set_prefetch_ready(false);
  auto options = CreateOptions();
  options.request_queue_size(1);
  TestableRecScheduler scheduler(&engine, options);
  auto requests = GenRequests({8, 8}, {4, 4}, RecType::kOneRec);
  ASSERT_TRUE(scheduler.add_request(requests[0]));
  requests[0]->set_cancel();

  EXPECT_TRUE(scheduler.prepare_batch_test().front().empty());

  EXPECT_EQ(engine.prefetch_calls(), 0u);
  EXPECT_FALSE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  EXPECT_TRUE(scheduler.add_request(requests[1]));
}

TEST(RecSchedulerTest, DestructorWaitsForInFlightRecExecution) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  FakeRecEngine engine(/*num_blocks=*/64, /*block_size=*/4);
  std::promise<void> execution_started;
  std::future<void> started = execution_started.get_future();
  std::promise<void> release_execution;
  std::shared_future<void> release = release_execution.get_future().share();
  engine.set_step_callback(
      [&execution_started, release](RecBatchGroup& /*batches*/) {
        execution_started.set_value();
        release.wait();
      });
  auto scheduler = std::make_unique<RecScheduler>(
      &engine,
      CreateOptions(/*max_tokens_per_batch=*/64,
                    /*max_seqs_per_batch=*/4,
                    /*dp_size=*/1,
                    /*enable_schedule_overlap=*/false,
                    /*rec_worker_max_concurrency=*/2));
  auto requests = GenRequests({8}, {4}, RecType::kOneRec);
  ASSERT_TRUE(scheduler->add_request(requests[0]));
  scheduler->step(absl::Milliseconds(50));
  EXPECT_EQ(started.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  auto destroyed = std::async(
      std::launch::async,
      [scheduler = std::move(scheduler)]() mutable { scheduler.reset(); });

  EXPECT_EQ(destroyed.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  release_execution.set_value();
  EXPECT_EQ(destroyed.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  destroyed.get();
  EXPECT_EQ(engine.step_calls(), 1);
}

}  // namespace xllm

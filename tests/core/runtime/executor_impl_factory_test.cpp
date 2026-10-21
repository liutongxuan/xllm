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

#include "core/runtime/executor_impl_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xllm {
namespace {

template <int32_t Priority>
class FactoryTestExecutorImpl final : public ExecutorImpl {
 public:
  FactoryTestExecutorImpl() = default;

  FactoryTestExecutorImpl(CausalLM* /*model*/,
                          const ModelArgs& /*args*/,
                          const torch::Device& /*device*/,
                          const runtime::Options& /*options*/) {}

  ModelOutput run(const torch::Tensor& /*tokens*/,
                  const torch::Tensor& /*positions*/,
                  std::vector<KVCache>& /*kv_caches*/,
                  const ModelInputParams& /*params*/) override {
    return ModelOutput{};
  }
};

using DefaultFactoryTestExecutorImpl = FactoryTestExecutorImpl<0>;
using PriorityFactoryTestExecutorImpl = FactoryTestExecutorImpl<10>;
using EqualPriorityFactoryTestExecutorImpl = FactoryTestExecutorImpl<5>;

template <int32_t Priority>
ExecutorImplFactory::Creator create_test_creator() {
  return [](CausalLM* /*model*/,
            const ModelArgs& /*args*/,
            const torch::Device& /*device*/,
            const runtime::Options& /*options*/) {
    return std::make_unique<FactoryTestExecutorImpl<Priority>>();
  };
}

std::unique_ptr<ExecutorImpl> create_executor(
    const std::string& executor_type) {
  return ExecutorImplFactory::get_instance().create_executor_impl(
      /*model=*/nullptr,
      /*args=*/ModelArgs{},
      /*device=*/torch::Device(torch::kCPU),
      /*options=*/runtime::Options{},
      /*executor_type=*/executor_type);
}

}  // namespace

REGISTER_EXECUTOR("executor_factory_macro_default",
                  DefaultFactoryTestExecutorImpl);
REGISTER_EXECUTOR("executor_factory_macro_priority",
                  PriorityFactoryTestExecutorImpl,
                  10);

namespace {

TEST(ExecutorImplFactoryTest, HigherPriorityReplacesLowerPriority) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_lower_then_higher",
                                       create_test_creator<0>(),
                                       /*priority=*/0));
  EXPECT_TRUE(factory.register_creator("executor_factory_lower_then_higher",
                                       create_test_creator<10>(),
                                       /*priority=*/10));

  auto executor = create_executor("executor_factory_lower_then_higher");
  EXPECT_NE(dynamic_cast<PriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, LowerPriorityDoesNotReplaceHigherPriority) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_higher_then_lower",
                                       create_test_creator<10>(),
                                       /*priority=*/10));
  EXPECT_FALSE(factory.register_creator("executor_factory_higher_then_lower",
                                        create_test_creator<0>(),
                                        /*priority=*/0));

  auto executor = create_executor("executor_factory_higher_then_lower");
  EXPECT_NE(dynamic_cast<PriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, EqualPriorityKeepsFirstRegistration) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_equal_priority",
                                       create_test_creator<5>(),
                                       /*priority=*/5));
  EXPECT_FALSE(factory.register_creator("executor_factory_equal_priority",
                                        create_test_creator<10>(),
                                        /*priority=*/5));

  auto executor = create_executor("executor_factory_equal_priority");
  EXPECT_NE(dynamic_cast<EqualPriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest,
     RegisterExecutorSupportsDefaultAndExplicitPriority) {
  auto default_executor = create_executor("executor_factory_macro_default");
  EXPECT_NE(
      dynamic_cast<DefaultFactoryTestExecutorImpl*>(default_executor.get()),
      nullptr);

  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_FALSE(factory.register_creator("executor_factory_macro_priority",
                                        create_test_creator<0>(),
                                        /*priority=*/0));
  auto priority_executor = create_executor("executor_factory_macro_priority");
  EXPECT_NE(
      dynamic_cast<PriorityFactoryTestExecutorImpl*>(priority_executor.get()),
      nullptr);
}

TEST(ExecutorImplFactoryTest, UnknownExecutorTypeThrows) {
  EXPECT_THROW(create_executor("executor_factory_unknown_executor_type"),
               std::runtime_error);
}

}  // namespace
}  // namespace xllm

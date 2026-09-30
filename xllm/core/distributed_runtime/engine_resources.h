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

#include <concepts>
#include <functional>
#include <utility>
#include <vector>

#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {

using ActivationMemoryReader = std::function<std::vector<int64_t>()>;

// Non-owning resources shared by scheduling components. The concrete engine
// must outlive every consumer of this view and its callbacks.
class EngineResources final {
 public:
  EngineResources(const ModelArgs& model_args,
                  const Tokenizer* tokenizer,
                  BlockManagerPool* block_manager_pool,
                  ActivationMemoryReader activation_memory_reader)
      : model_args_(&model_args),
        tokenizer_(tokenizer),
        block_manager_pool_(block_manager_pool),
        activation_memory_reader_(std::move(activation_memory_reader)) {}

  template <typename TargetEngine>
    requires requires(TargetEngine& engine) {
      { engine.model_args() } -> std::same_as<const ModelArgs&>;
      { engine.tokenizer() } -> std::convertible_to<const Tokenizer*>;
      { engine.block_manager_pool() } -> std::convertible_to<BlockManagerPool*>;
      {
        engine.get_active_activation_memory()
      } -> std::same_as<std::vector<int64_t>>;
    }
  static EngineResources bind(TargetEngine& engine) {
    return EngineResources(
        engine.model_args(),
        engine.tokenizer(),
        engine.block_manager_pool(),
        [&engine] { return engine.get_active_activation_memory(); });
  }

  const ModelArgs& model_args() const { return *model_args_; }
  const Tokenizer* tokenizer() const { return tokenizer_; }
  BlockManagerPool* block_manager_pool() const { return block_manager_pool_; }
  const ActivationMemoryReader& activation_memory_reader() const {
    return activation_memory_reader_;
  }

 private:
  const ModelArgs* model_args_;
  const Tokenizer* tokenizer_;
  BlockManagerPool* block_manager_pool_;
  ActivationMemoryReader activation_memory_reader_;
};

}  // namespace xllm

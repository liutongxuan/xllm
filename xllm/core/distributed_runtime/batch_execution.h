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

#include <glog/logging.h>

#include <concepts>
#include <functional>
#include <utility>

#include "core/distributed_runtime/engine_resources.h"
#include "core/framework/batch/batch_group.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/forward_params.h"

namespace xllm {

// A value capability for ordinary sequence batches. It cannot bind an engine
// whose execution contract accepts only RecBatchGroup or DiTBatch.
class BatchExecution final {
 public:
  using Step = std::function<ForwardOutput(BatchGroup&)>;
  using Consume = std::function<void(BatchGroup&)>;
  using GraphShapeReader = std::function<runtime::DecodeGraphExecutionShape()>;
  using PredictorSetter = std::function<bool(
      const SpeculativeProfileRegistry::ValidateTimePredictor&)>;

  BatchExecution(EngineResources resources,
                 Step step,
                 Consume consume,
                 GraphShapeReader graph_shape_reader = {},
                 PredictorSetter predictor_setter = {})
      : resources_(std::move(resources)),
        step_(std::move(step)),
        consume_(std::move(consume)),
        graph_shape_reader_(std::move(graph_shape_reader)),
        predictor_setter_(std::move(predictor_setter)) {
    CHECK(step_);
    CHECK(consume_);
  }

  template <typename TargetEngine>
    requires requires(TargetEngine& engine, BatchGroup& batches) {
      EngineResources::bind(engine);
      { engine.step(batches) } -> std::same_as<ForwardOutput>;
      { engine.update_last_step_result(batches) } -> std::same_as<void>;
    }
  static BatchExecution bind(TargetEngine& engine) {
    GraphShapeReader graph_shape_reader;
    if constexpr (requires { engine.decode_graph_execution_shape(); }) {
      graph_shape_reader = [&engine] {
        return engine.decode_graph_execution_shape();
      };
    }
    PredictorSetter predictor_setter;
    if constexpr (requires(
                      const SpeculativeProfileRegistry::ValidateTimePredictor&
                          predictor) {
                    engine.set_speculative_validate_time_predictor(predictor);
                  }) {
      predictor_setter = [&engine](const auto& predictor) {
        return engine.set_speculative_validate_time_predictor(predictor);
      };
    }
    return BatchExecution(
        EngineResources::bind(engine),
        [&engine](BatchGroup& batches) { return engine.step(batches); },
        [&engine](BatchGroup& batches) {
          engine.update_last_step_result(batches);
        },
        std::move(graph_shape_reader),
        std::move(predictor_setter));
  }

  const EngineResources& resources() const { return resources_; }
  const ModelArgs& model_args() const { return resources_.model_args(); }
  BlockManagerPool* block_manager_pool() const {
    return resources_.block_manager_pool();
  }
  ForwardOutput step(BatchGroup& batches) const { return step_(batches); }
  void update_last_step_result(BatchGroup& batches) const { consume_(batches); }
  runtime::DecodeGraphExecutionShape decode_graph_execution_shape() const {
    return graph_shape_reader_ ? graph_shape_reader_()
                               : runtime::DecodeGraphExecutionShape{};
  }
  bool set_speculative_validate_time_predictor(
      const SpeculativeProfileRegistry::ValidateTimePredictor& predictor)
      const {
    return predictor_setter_ && predictor_setter_(predictor);
  }

 private:
  EngineResources resources_;
  Step step_;
  Consume consume_;
  GraphShapeReader graph_shape_reader_;
  PredictorSetter predictor_setter_;
};

}  // namespace xllm

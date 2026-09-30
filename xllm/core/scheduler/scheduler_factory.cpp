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

#include "scheduler/scheduler_factory.h"

#include "core/distributed_runtime/rec_engine.h"
#include "core/framework/config/scheduler_config.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/rec_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

namespace xllm {

SchedulerKind select_scheduler_kind(
    const ContinuousScheduler::Options& options) {
  if (options.enable_disagg_pd()) {
    return SchedulerKind::DISAGG_PD;
  }

  if (::xllm::SchedulerConfig::get_instance().use_zero_evict()) {
    return SchedulerKind::ZERO_EVICTION;
  }

  return SchedulerKind::CONTINUOUS;
}

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    BatchExecution execution,
    ContinuousScheduler::Options options,
    PDExecution pd_execution,
    XTensorInfoProvider xtensor_info_provider) {
  switch (select_scheduler_kind(options)) {
    case SchedulerKind::DISAGG_PD:
      pd_execution.validate_cluster_exchange();
      return std::make_unique<DisaggPDScheduler>(
          std::move(execution),
          options,
          std::move(pd_execution),
          std::move(xtensor_info_provider));
    case SchedulerKind::ZERO_EVICTION:
      return std::make_unique<ZeroEvictionScheduler>(
          std::move(execution),
          options,
          std::move(pd_execution),
          std::move(xtensor_info_provider));
    case SchedulerKind::CONTINUOUS:
      return std::make_unique<ContinuousScheduler>(
          std::move(execution),
          options,
          std::move(pd_execution),
          std::move(xtensor_info_provider));
  }

  return std::make_unique<ContinuousScheduler>(
      std::move(execution),
      options,
      std::move(pd_execution),
      std::move(xtensor_info_provider));
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

std::unique_ptr<RecScheduler> create_rec_scheduler(
    RecEngine* engine,
    RecScheduler::Options options) {
  CHECK(engine != nullptr);
  return std::make_unique<RecScheduler>(
      engine, std::move(options), engine->execution_config());
}

}  // namespace xllm

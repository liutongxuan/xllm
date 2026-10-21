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

#include <torch/types.h>

#include <string>
#include <vector>

namespace xllm {

namespace proto {
class DiTForwardOutput;
}  // namespace proto

class DiTForwardOutput final {
 public:
  bool to_proto(proto::DiTForwardOutput* proto_output) const;
  bool from_proto(const proto::DiTForwardOutput& proto_output);

  // Generated tensors for image/audio models.
  std::vector<torch::Tensor> tensors;
  // Optional generated audio tensors for video+audio models.
  std::vector<torch::Tensor> audio_tensors;
  // Generated text for text diffusion models such as Cola-DLM.
  std::vector<std::string> text_output;
};

}  // namespace xllm

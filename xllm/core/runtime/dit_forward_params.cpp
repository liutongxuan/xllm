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

#include "runtime/dit_forward_params.h"

#include <glog/logging.h>

#include <utility>

#include "util/utils.h"
#include "worker.pb.h"

namespace xllm {

namespace {

bool serialize_named_tensors(
    const std::vector<NamedTensor>& sources,
    google::protobuf::RepeatedPtrField<proto::NamedTensor>* proto_sources) {
  proto_sources->Reserve(static_cast<int32_t>(sources.size()));
  for (const NamedTensor& source : sources) {
    proto::NamedTensor* proto_source = proto_sources->Add();
    proto_source->set_name(source.name);
    if (!source.tensor.defined() || !source.tensor.device().is_cpu() ||
        !source.tensor.is_contiguous() ||
        !util::torch_to_proto(source.tensor, proto_source->mutable_tensor())) {
      LOG(ERROR) << "Failed to serialize named tensor: " << source.name;
      return false;
    }
  }
  return true;
}

template <typename Sources>
bool deserialize_named_tensors(
    const google::protobuf::RepeatedPtrField<proto::NamedTensor>& proto_sources,
    Sources& sources) {
  sources.entries().reserve(proto_sources.size());
  for (const proto::NamedTensor& proto_source : proto_sources) {
    torch::Tensor tensor = util::proto_to_torch(proto_source.tensor());
    if (!tensor.defined()) {
      LOG(ERROR) << "Failed to deserialize named tensor: "
                 << proto_source.name();
      return false;
    }
    sources.add(proto_source.name(), std::move(tensor));
  }
  return true;
}

}  // namespace

bool DiTForwardInput::to_proto(proto::DiTForwardInput* proto_input) const {
  if (proto_input == nullptr) {
    return false;
  }

  proto::DiTForwardInput encoded_input;
  encoded_input.set_batch_size(batch_size);
  encoded_input.mutable_prompts()->Add(prompts.begin(), prompts.end());
  encoded_input.mutable_prompts_2()->Add(prompts_2.begin(), prompts_2.end());
  encoded_input.mutable_negative_prompts()->Add(negative_prompts.begin(),
                                                negative_prompts.end());
  encoded_input.mutable_negative_prompts_2()->Add(negative_prompts_2.begin(),
                                                  negative_prompts_2.end());
  if (!serialize_named_tensors(image_sources.entries(),
                               encoded_input.mutable_image_sources()) ||
      !serialize_named_tensors(tensor_sources.entries(),
                               encoded_input.mutable_tensor_sources())) {
    return false;
  }
  if (!audio_prompt_text.empty()) {
    encoded_input.set_audio_prompt_text(audio_prompt_text);
  }
  if (!generation_params.to_proto(encoded_input.mutable_generation_params())) {
    LOG(ERROR) << "Failed to serialize generation params";
    return false;
  }
  proto_input->Swap(&encoded_input);
  return true;
}

bool DiTForwardInput::from_proto(const proto::DiTForwardInput& proto_input) {
  DiTForwardInput decoded_input;
  decoded_input.batch_size = proto_input.batch_size();
  decoded_input.prompts.assign(proto_input.prompts().begin(),
                               proto_input.prompts().end());
  decoded_input.prompts_2.assign(proto_input.prompts_2().begin(),
                                 proto_input.prompts_2().end());
  decoded_input.negative_prompts.assign(proto_input.negative_prompts().begin(),
                                        proto_input.negative_prompts().end());
  decoded_input.negative_prompts_2.assign(
      proto_input.negative_prompts_2().begin(),
      proto_input.negative_prompts_2().end());
  if (!deserialize_named_tensors(proto_input.image_sources(),
                                 decoded_input.image_sources) ||
      !deserialize_named_tensors(proto_input.tensor_sources(),
                                 decoded_input.tensor_sources)) {
    return false;
  }
  if (!decoded_input.generation_params.from_proto(
          proto_input.generation_params())) {
    LOG(ERROR) << "Failed to deserialize generation params";
    return false;
  }
  if (proto_input.has_audio_prompt_text()) {
    decoded_input.audio_prompt_text = proto_input.audio_prompt_text();
  }
  *this = std::move(decoded_input);
  return true;
}

bool DiTForwardOutput::to_proto(proto::DiTForwardOutput* proto_output) const {
  if (proto_output == nullptr) {
    return false;
  }

  proto::DiTForwardOutput encoded_output;
  auto* proto_tensors = encoded_output.mutable_tensors()->mutable_tensors();
  proto_tensors->Reserve(static_cast<int32_t>(tensors.size()));
  for (const torch::Tensor& tensor : tensors) {
    if (!tensor.defined() || !tensor.device().is_cpu() ||
        !tensor.is_contiguous() ||
        !util::torch_to_proto(tensor, proto_tensors->Add())) {
      LOG(ERROR) << "Failed to serialize DiT output tensor";
      return false;
    }
  }
  encoded_output.mutable_text_output()->Add(text_output.begin(),
                                            text_output.end());
  proto_output->Swap(&encoded_output);
  return true;
}

bool DiTForwardOutput::from_proto(const proto::DiTForwardOutput& proto_output) {
  DiTForwardOutput decoded_output;
  const auto& proto_tensors = proto_output.tensors().tensors();
  decoded_output.tensors.reserve(proto_tensors.size());
  for (const proto::Tensor& proto_tensor : proto_tensors) {
    torch::Tensor tensor = util::proto_to_torch(proto_tensor);
    if (!tensor.defined()) {
      LOG(ERROR) << "Failed to deserialize DiT output tensor";
      return false;
    }
    decoded_output.tensors.emplace_back(std::move(tensor));
  }
  decoded_output.text_output.assign(proto_output.text_output().begin(),
                                    proto_output.text_output().end());
  *this = std::move(decoded_output);
  return true;
}

}  // namespace xllm

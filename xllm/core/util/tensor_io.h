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
#include <torch/torch.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace xllm {

// Pickle-format tensor I/O, compatible with torch.save/torch.load in Python.
//
// These helpers need <torch/serialize.h>, which torch/types.h does not bring
// in. They live in their own header so that tensor_helper.h, which is reached
// by several hundred translation units, does not pull that surface into all of
// them.

inline std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}

inline torch::Tensor load_tensor(std::string filename) {
  std::vector<char> f = get_the_bytes(filename);
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor my_tensor = x.toTensor();
  return my_tensor;
}

// save torch tensor to .pt file as pickle format, which is same as torch.save
// in python. .pt file can be loaded by torch.load in python. file_path must end
// with ".pt".
inline void save_tensor_as_pickle(const torch::Tensor& tensor,
                                  const std::string& file_path) {
  std::vector<char> pickled = torch::pickle_save(tensor);
  std::ofstream ofs(file_path, std::ios::binary);
  CHECK(ofs.good()) << "Cannot open file: " << file_path;
  ofs.write(pickled.data(), static_cast<std::streamsize>(pickled.size()));
  CHECK(ofs.good()) << "Write failed to: " << file_path;
}

}  // namespace xllm

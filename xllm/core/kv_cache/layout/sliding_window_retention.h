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

#include <algorithm>
#include <cstddef>

namespace xllm {

// Rolling-window boundaries in logical block positions. Physical availability,
// shared ownership and prefix-checkpoint publication belong to their callers.
class SlidingWindowRetention final {
 public:
  // Legacy zero-window options use a single-token window.
  SlidingWindowRetention(size_t token_stride,
                         size_t window_tokens,
                         size_t speculative_tokens)
      : token_stride_(token_stride),
        window_tokens_(std::max<size_t>(window_tokens, 1)),
        speculative_tokens_(speculative_tokens) {
    CHECK_GT(token_stride_, 0u);
  }

  // Restore the block containing the window's first token, including a partial
  // block at an unaligned restore cursor. This precedes the next forward.
  size_t restore_begin_block(size_t restored_tokens) const {
    return (restored_tokens - std::min(restored_tokens, window_tokens_)) /
           token_stride_;
  }

  // Exclusive end of the expired prefix. The next forward needs window - 1
  // history tokens; speculative rollback keeps additional tokens resident.
  size_t release_end_block(size_t cached_tokens, size_t logical_blocks) const {
    if (cached_tokens < window_tokens_ ||
        cached_tokens - window_tokens_ < speculative_tokens_) {
      return 0;
    }
    const size_t expired_tokens =
        cached_tokens - window_tokens_ - speculative_tokens_ + 1;
    return std::min(expired_tokens / token_stride_, logical_blocks);
  }

 private:
  size_t token_stride_;
  size_t window_tokens_;
  size_t speculative_tokens_;
};

}  // namespace xllm

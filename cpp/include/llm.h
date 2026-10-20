// RKLLM 封装：加载 Qwen2.5-1.5B，流式回调，统计 TTFT。
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include <rkllm.h>

namespace mm {

class LLMEngine {
 public:
  struct Reply {
    std::string text;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    int segments = 0;
  };

  LLMEngine(const std::string& model_path, int max_new_tokens = 48,
            int max_context_len = 2048);
  ~LLMEngine();

  LLMEngine(const LLMEngine&) = delete;
  LLMEngine& operator=(const LLMEngine&) = delete;

  Reply run(const std::string& prompt);

 private:
  static int on_result(RKLLMResult* result, void* userdata, LLMCallState state);

  LLMHandle handle_ = nullptr;
  std::mutex run_mtx_;
  std::string buf_;
  double t0_ = 0.0;
  double t_first_ = 0.0;
  int segments_ = 0;
};

}  // namespace mm

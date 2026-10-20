#include "llm.h"

#include <cstring>
#include <stdexcept>

#include "common.h"

namespace mm {

LLMEngine::LLMEngine(const std::string& model_path, int max_new_tokens, int max_context_len) {
  RKLLMParam param = rkllm_createDefaultParam();
  param.model_path = model_path.c_str();
  param.max_new_tokens = max_new_tokens;
  param.max_context_len = max_context_len;
  // 结构化 Prompt 要的是稳定复现，不是创造性
  param.top_k = 1;
  param.top_p = 0.95f;
  param.temperature = 0.8f;
  param.repeat_penalty = 1.1f;
  param.skip_special_token = true;
  param.extend_param.base_domain_id = 0;
  param.extend_param.embed_flash = 1;
  // rkllm v1.2.3 起 RKLLMExtendParam 新增 n_batch，要求 1~100；
  // 默认值 0 会让 rkllm_init 直接崩，必须显式设置。
  param.extend_param.n_batch = 1;

  const int ret = rkllm_init(&handle_, &param, &LLMEngine::on_result);
  if (ret != 0 || handle_ == nullptr) {
    throw std::runtime_error("rkllm_init failed, ret=" + std::to_string(ret));
  }
}

LLMEngine::~LLMEngine() {
  if (handle_ != nullptr) {
    rkllm_destroy(handle_);
    handle_ = nullptr;
  }
}

int LLMEngine::on_result(RKLLMResult* result, void* userdata, LLMCallState state) {
  auto* self = static_cast<LLMEngine*>(userdata);
  if (self == nullptr) return 0;
  if (state == RKLLM_RUN_NORMAL) {
    if (self->t_first_ == 0.0) self->t_first_ = now_ms();
    ++self->segments_;
    if (result != nullptr && result->text != nullptr) self->buf_ += result->text;
  } else if (state == RKLLM_RUN_ERROR) {
    self->buf_ += "[推理出错]";
  }
  return 0;
}

LLMEngine::Reply LLMEngine::run(const std::string& prompt) {
  std::lock_guard<std::mutex> lock(run_mtx_);  // LLM 是独占资源
  buf_.clear();
  segments_ = 0;
  t_first_ = 0.0;
  t0_ = now_ms();

  RKLLMInput input;
  std::memset(&input, 0, sizeof(input));
  input.input_type = RKLLM_INPUT_PROMPT;
  input.role = "user";
  input.prompt_input = const_cast<char*>(prompt.c_str());

  RKLLMInferParam infer;
  std::memset(&infer, 0, sizeof(infer));
  infer.mode = RKLLM_INFER_GENERATE;
  infer.keep_history = 0;  // 每帧观测独立，不保留 KV cache

  const int ret = rkllm_run(handle_, &input, &infer, this);
  const double t_end = now_ms();

  Reply r;
  r.text = buf_;
  r.ttft_ms = t_first_ > 0 ? t_first_ - t0_ : 0.0;
  r.total_ms = t_end - t0_;
  r.segments = segments_;
  if (ret != 0) throw std::runtime_error("rkllm_run failed, ret=" + std::to_string(ret));
  return r;
}

}  // namespace mm

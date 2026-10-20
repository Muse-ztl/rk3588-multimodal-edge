// Minimal C ABI bridge over the Rockchip RKLLM runtime.
//
// Why a bridge instead of ctypes-ing RKLLM structs directly:
//   RKLLMParam / RKLLMInput / RKLLMInferParam layouts change between runtime
//   releases (v1.0 lacked extend_param & skip_special_token, v1.2.3 added
//   n_batch, ...). Compiling against the header that ships with the target
//   runtime keeps the ABI correct and lets Python stay version-agnostic.
//
// Build (on board):
//   bash scripts/build_llm_bridge.sh
//
// ABI:
//   int  mmllm_init(const char* model_path, int max_new_tokens, int max_context_len,
//                   mmllm_text_cb cb, void* user, char* err, int errlen);
//   int  mmllm_run(const char* prompt);
//   int  mmllm_abort(void);
//   void mmllm_destroy(void);
//   const char* mmllm_last_error(void);

#include <cstddef>
#include <cstdint>

#include <rkllm.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace {

using TextCb = void (*)(const char* chunk, void* user);

LLMHandle g_handle = nullptr;
TextCb g_cb = nullptr;
void* g_user = nullptr;
std::mutex g_init_mtx;
std::mutex g_run_mtx;
std::string g_err;

int on_result(RKLLMResult* result, void* /*userdata*/, LLMCallState state) {
  switch (state) {
    case RKLLM_RUN_NORMAL:
      if (g_cb != nullptr && result != nullptr && result->text != nullptr) {
        g_cb(result->text, g_user);
      }
      break;
    case RKLLM_RUN_WAITING:
      break;
    case RKLLM_RUN_FINISH:
      break;
    case RKLLM_RUN_ERROR:
      g_err = "rkllm: inference error callback";
      break;
    default:
      break;
  }
  return 0;
}

void set_err(char* buf, int len, const std::string& msg) {
  if (buf != nullptr && len > 0) {
    std::snprintf(buf, static_cast<size_t>(len), "%s", msg.c_str());
  }
  g_err = msg;
}

}  // namespace

extern "C" {

int mmllm_init(const char* model_path, int max_new_tokens, int max_context_len,
               TextCb cb, void* user, char* err, int errlen) {
  std::lock_guard<std::mutex> lk(g_init_mtx);
  if (g_handle != nullptr) {
    return 0;
  }
  if (model_path == nullptr) {
    set_err(err, errlen, "model_path is null");
    return -1;
  }

  g_cb = cb;
  g_user = user;

  RKLLMParam param = rkllm_createDefaultParam();
  param.model_path = model_path;
  param.max_new_tokens = max_new_tokens;
  param.max_context_len = max_context_len;
  // 结构化 Prompt 场景下要的是稳定复现，不是创造性：top_k=1 贪心解码。
  param.top_k = 1;
  param.top_p = 0.95f;
  param.temperature = 0.8f;
  param.repeat_penalty = 1.1f;
  param.skip_special_token = true;
  param.extend_param.base_domain_id = 0;
  param.extend_param.embed_flash = 1;
  // rkllm v1.2.3 起 RKLLMExtendParam 新增 n_batch，要求 1~100；
  // rkllm_createDefaultParam() 给的默认值是 0，不显式设置会在 init 阶段崩。
  param.extend_param.n_batch = 1;

  LLMHandle handle = nullptr;
  const int ret = rkllm_init(&handle, &param, on_result);
  if (ret != 0 || handle == nullptr) {
    set_err(err, errlen, "rkllm_init failed, ret=" + std::to_string(ret));
    g_handle = nullptr;
    return ret == 0 ? -1 : ret;
  }
  g_handle = handle;
  return 0;
}

int mmllm_run(const char* prompt) {
  if (g_handle == nullptr) {
    g_err = "rkllm: model not initialised";
    return -1;
  }
  if (prompt == nullptr) {
    g_err = "rkllm: prompt is null";
    return -2;
  }

  std::lock_guard<std::mutex> lk(g_run_mtx);  // 单实例推理，天然互斥

  RKLLMInput input;
  std::memset(&input, 0, sizeof(input));
  input.input_type = RKLLM_INPUT_PROMPT;
  input.role = "user";
  input.prompt_input = const_cast<char*>(prompt);

  RKLLMInferParam infer;
  std::memset(&infer, 0, sizeof(infer));
  infer.mode = RKLLM_INFER_GENERATE;
  infer.keep_history = 0;  // 每帧观测独立，不保留 KV cache

  const int ret = rkllm_run(g_handle, &input, &infer, nullptr);
  if (ret != 0) {
    g_err = "rkllm_run failed, ret=" + std::to_string(ret);
  }
  return ret;
}

int mmllm_abort(void) {
  if (g_handle == nullptr) {
    return -1;
  }
  return rkllm_abort(g_handle);
}

void mmllm_destroy(void) {
  std::lock_guard<std::mutex> lk(g_init_mtx);
  if (g_handle != nullptr) {
    rkllm_destroy(g_handle);
    g_handle = nullptr;
  }
}

const char* mmllm_last_error(void) {
  return g_err.c_str();
}

}  // namespace

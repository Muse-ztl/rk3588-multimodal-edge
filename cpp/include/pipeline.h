// CV / LLM 异构调度：NPU 采样线程 + 异步 LLM 线程，单槽位背压。
#pragma once

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "llm.h"
#include "rfdetr_split.h"

namespace mm {

struct LLMTrace {
  long frame_id = 0;
  double obs_ts = 0.0;
  std::string text;
  double ttft_ms = 0.0;
  double total_ms = 0.0;
  double e2e_ms = 0.0;
  std::string risk_level;
  std::vector<std::string> rules;
};

struct Stats {
  long frames_read = 0;
  long infer_frames = 0;
  std::vector<double> cv_ms, npu_ms, head_ms;
  long frames_while_llm_busy = 0;
  long observations_published = 0;
  long observations_consumed = 0;
  long llm_calls = 0;
  std::vector<LLMTrace> traces;
  std::string affinity;
};

class Pipeline {
 public:
  Pipeline(SplitRFDETR& detector, LLMEngine* llm, const std::string& video,
           const std::string& out_dir, int cv_stride = 1, int viz_every = 15,
           bool loop_video = true, std::vector<int> cv_cores = {});

  Stats run(double duration_s = 0.0);

 private:
  void cv_loop(double duration_s);
  void llm_loop();
  void dump_viz(const cv::Mat& frame, const Observation& obs, const std::string& name);

  SplitRFDETR& detector_;
  LLMEngine* llm_;
  std::string video_;
  std::string out_dir_;
  int cv_stride_;
  int viz_every_;
  bool loop_video_;
  std::vector<int> cv_cores_;

  std::mutex slot_mtx_;
  std::condition_variable slot_cv_;
  Observation slot_;
  long slot_seq_ = 0;
  bool has_slot_ = false;
  bool stop_ = false;
  std::atomic<bool> llm_busy_{false};

  Stats stats_;
  std::ofstream trace_file_;
};

}  // namespace mm

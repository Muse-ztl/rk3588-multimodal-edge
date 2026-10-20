#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

#include "scene.h"

namespace mm {
namespace {

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(std::min(1.0, std::max(0.0, p / 100.0)) *
                                         (v.size() - 1));
  return v[idx];
}

double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0;
  for (double x : v) s += x;
  return s / v.size();
}

std::string pin_cores(const std::vector<int>& cores) {
#if defined(__linux__)
  if (cores.empty()) return "cpu_affinity=unchanged";
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : cores) CPU_SET(c, &set);
  if (sched_setaffinity(0, sizeof(set), &set) == 0) {
    std::ostringstream os;
    os << "cpu_affinity=[";
    for (size_t i = 0; i < cores.size(); ++i) os << (i ? "," : "") << cores[i];
    os << "]";
    return os.str();
  }
  return "cpu_affinity=failed";
#else
  (void)cores;
  return "cpu_affinity=unsupported";
#endif
}

std::string risk_color_hex(const std::string& level) {
  if (level == "danger") return "#dc3c3c";
  if (level == "warning") return "#f0aa28";
  return "#50c850";
}

cv::Scalar risk_color(const std::string& level) {
  if (level == "danger") return cv::Scalar(60, 60, 220);
  if (level == "warning") return cv::Scalar(40, 170, 240);
  return cv::Scalar(80, 200, 80);
}

}  // namespace

Pipeline::Pipeline(SplitRFDETR& detector, LLMEngine* llm, const std::string& video,
                   const std::string& out_dir, int cv_stride, int viz_every,
                   bool loop_video, std::vector<int> cv_cores)
    : detector_(detector),
      llm_(llm),
      video_(video),
      out_dir_(out_dir),
      cv_stride_(std::max(1, cv_stride)),
      viz_every_(std::max(1, viz_every)),
      loop_video_(loop_video),
      cv_cores_(std::move(cv_cores)) {
  trace_file_.open(out_dir_ + "/llm_traces.jsonl", std::ios::out | std::ios::trunc);
}

void Pipeline::dump_viz(const cv::Mat& frame, const Observation& obs, const std::string& name) {
  cv::Mat img = frame.clone();
  const int w = img.cols, h = img.rows;
  const cv::Scalar color = risk_color(obs.risk_level);

  // 危险区
  cv::rectangle(img, cv::Point(static_cast<int>(0.20 * w), static_cast<int>(0.55 * h)),
                cv::Point(static_cast<int>(0.80 * w), h), cv::Scalar(0, 200, 255), 2);
  cv::putText(img, "DANGER ZONE", cv::Point(static_cast<int>(0.20 * w) + 6,
                                            static_cast<int>(0.55 * h) + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);

  for (const auto& d : obs.dets) {
    cv::rectangle(img, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                  cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
    std::ostringstream label;
    label << d.label << " " << std::fixed << std::setprecision(2) << d.score;
    cv::putText(img, label.str(), cv::Point(static_cast<int>(d.x1) + 2,
                                            std::max(12, static_cast<int>(d.y1) - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
  }

  // 顶部信息条（ASCII：OpenCV 内置字体不支持中文，中文语义见 llm_traces.jsonl）
  cv::Mat bar = img.clone();
  cv::rectangle(bar, cv::Point(0, 0), cv::Point(w, 28), cv::Scalar(20, 20, 20), -1);
  cv::addWeighted(bar, 0.6, img, 0.4, 0, img);
  std::ostringstream head;
  head << "frame " << obs.frame_id << " | risk=" << obs.risk_level
       << " | dets=" << obs.dets.size();
  cv::putText(img, head.str(), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
              cv::LINE_AA);

  cv::imwrite(out_dir_ + "/" + name, img);
}

void Pipeline::cv_loop(double duration_s) {
  stats_.affinity = pin_cores(cv_cores_);
  cv::VideoCapture cap(video_);
  if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + video_);
  const double t_begin = now_ms();
  long frame_id = 0;

  while (!stop_) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      if (!loop_video_) break;
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    ++frame_id;
    ++stats_.frames_read;

    if (duration_s > 0 && (now_ms() - t_begin) / 1000.0 > duration_s) break;
    if ((frame_id - 1) % cv_stride_ != 0) continue;

    if (llm_busy_.load()) ++stats_.frames_while_llm_busy;

    auto dets = detector_.infer(frame);
    Observation obs = analyze(dets, frame.cols, frame.rows);
    obs.frame_id = frame_id;
    obs.ts = now_s();

    {
      std::lock_guard<std::mutex> lock(slot_mtx_);
      slot_ = std::move(obs);
      ++slot_seq_;
      has_slot_ = true;
      ++stats_.observations_published;
    }
    slot_cv_.notify_one();

    ++stats_.infer_frames;
    const Timing& t = detector_.timing();
    stats_.cv_ms.push_back(t.total_ms);
    stats_.npu_ms.push_back(t.backbone_ms);
    stats_.head_ms.push_back(t.head_ms);

    if (stats_.infer_frames % viz_every_ == 0) {
      std::ostringstream name;
      name << "frame_" << std::setw(5) << std::setfill('0') << frame_id << ".jpg";
      dump_viz(frame, slot_, name.str());
    }
  }
  cap.release();
}

void Pipeline::llm_loop() {
  long seen_seq = 0;
  while (!stop_) {
    Observation obs;
    {
      std::unique_lock<std::mutex> lock(slot_mtx_);
      slot_cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
        return stop_ || (slot_seq_ > seen_seq && has_slot_);
      });
      if (stop_ && slot_seq_ <= seen_seq) return;
      if (slot_seq_ <= seen_seq || !has_slot_) continue;
      seen_seq = slot_seq_;
      obs = slot_;
      ++stats_.observations_consumed;
    }

    if (llm_ == nullptr) continue;

    const std::string prompt = build_prompt(obs);
    llm_busy_.store(true);
    LLMEngine::Reply reply;
    try {
      reply = llm_->run(prompt);
    } catch (const std::exception& e) {
      llm_busy_.store(false);
      fprintf(stderr, "[llm] run failed: %s\n", e.what());
      continue;
    }
    llm_busy_.store(false);

    LLMTrace tr;
    tr.frame_id = obs.frame_id;
    tr.obs_ts = obs.ts;
    tr.text = reply.text;
    tr.ttft_ms = reply.ttft_ms;
    tr.total_ms = reply.total_ms;
    tr.e2e_ms = (now_s() - obs.ts) * 1000.0;
    tr.risk_level = obs.risk_level;
    tr.rules = obs.rules;

    {
      std::lock_guard<std::mutex> lock(slot_mtx_);
      stats_.traces.push_back(tr);
      ++stats_.llm_calls;
    }

    std::ostringstream line;
    line << "{\"frame_id\":" << tr.frame_id << ",\"text\":\"" << tr.text << "\",\"ttft_ms\":"
         << std::fixed << std::setprecision(1) << tr.ttft_ms << ",\"total_ms\":" << tr.total_ms
         << ",\"e2e_ms\":" << tr.e2e_ms << ",\"risk_level\":\"" << tr.risk_level << "\"}";
    trace_file_ << line.str() << "\n";
    trace_file_.flush();
    printf("[llm] frame=%ld risk=%s ttft=%.0fms total=%.0fms | %s\n", tr.frame_id,
           tr.risk_level.c_str(), tr.ttft_ms, tr.total_ms, tr.text.c_str());
  }
}

Stats Pipeline::run(double duration_s) {
  std::thread cv_thread([&] { cv_loop(duration_s); });
  std::thread llm_thread([&] { llm_loop(); });
  cv_thread.join();
  stop_ = true;
  slot_cv_.notify_all();
  llm_thread.join();
  trace_file_.close();

  // 汇总 report.json
  std::vector<double> ttft, total, e2e;
  for (const auto& t : stats_.traces) {
    ttft.push_back(t.ttft_ms);
    total.push_back(t.total_ms);
    e2e.push_back(t.e2e_ms);
  }
  std::ofstream rep(out_dir_ + "/report.json");
  rep << std::fixed << std::setprecision(1) << "{\n"
      << "  \"affinity\": \"" << stats_.affinity << "\",\n"
      << "  \"frames_read\": " << stats_.frames_read << ",\n"
      << "  \"infer_frames\": " << stats_.infer_frames << ",\n"
      << "  \"cv_total_p50_ms\": " << pct(stats_.cv_ms, 50) << ",\n"
      << "  \"cv_total_p90_ms\": " << pct(stats_.cv_ms, 90) << ",\n"
      << "  \"backbone_npu_p50_ms\": " << pct(stats_.npu_ms, 50) << ",\n"
      << "  \"head_cpu_p50_ms\": " << pct(stats_.head_ms, 50) << ",\n"
      << "  \"cv_fps_effective\": " << (stats_.cv_ms.empty() ? 0.0 : 1000.0 / mean(stats_.cv_ms))
      << ",\n"
      << "  \"llm_calls\": " << stats_.llm_calls << ",\n"
      << "  \"llm_ttft_p50_ms\": " << pct(ttft, 50) << ",\n"
      << "  \"llm_total_p50_ms\": " << pct(total, 50) << ",\n"
      << "  \"llm_total_p90_ms\": " << pct(total, 90) << ",\n"
      << "  \"e2e_p50_ms\": " << pct(e2e, 50) << ",\n"
      << "  \"e2e_p90_ms\": " << pct(e2e, 90) << ",\n"
      << "  \"observations_published\": " << stats_.observations_published << ",\n"
      << "  \"observations_consumed\": " << stats_.observations_consumed << ",\n"
      << "  \"observations_dropped\": "
      << (stats_.observations_published - stats_.observations_consumed) << ",\n"
      << "  \"frames_while_llm_busy\": " << stats_.frames_while_llm_busy << "\n"
      << "}\n";
  rep.close();
  return stats_;
}

}  // namespace mm

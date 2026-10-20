// Shared types for the RK3588 multimodal pipeline.
#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace mm {

struct Detection {
  float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  float score = 0.f;
  int class_id = 0;
  std::string label;
};

struct Timing {
  double preprocess_ms = 0.0;
  double backbone_ms = 0.0;
  double head_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

struct Observation {
  long frame_id = 0;
  double ts = 0.0;
  int width = 0;
  int height = 0;
  std::vector<Detection> dets;
  std::string location_text;   // "person(0.92)@中央近处，..."
  std::vector<std::string> positions;  // 与 dets 一一对应的位置串
  std::vector<std::string> rules;
  std::string risk_level = "normal";  // normal / warning / danger
};

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline double now_s() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

const char* coco_name(int idx);

}  // namespace mm

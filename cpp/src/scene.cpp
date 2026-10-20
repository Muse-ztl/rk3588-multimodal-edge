#include "scene.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace mm {
namespace {

bool is_person(const std::string& l) { return l == "person"; }

bool is_vehicle(const std::string& l) {
  return l == "car" || l == "bus" || l == "truck" || l == "motorcycle" ||
         l == "bicycle" || l == "train";
}

float center_x(const Detection& d) { return (d.x1 + d.x2) * 0.5f; }
float center_y(const Detection& d) { return (d.y1 + d.y2) * 0.5f; }

std::string position_text(const Detection& d, int w, int h) {
  const float nx = center_x(d) / std::max(w, 1);
  const float bottom = d.y2 / std::max(h, 1);
  const char* lr = nx < 1.f / 3.f ? "左侧" : (nx < 2.f / 3.f ? "中央" : "右侧");
  const char* depth = bottom > 0.75f ? "近处" : (bottom > 0.45f ? "中景" : "远处");
  return std::string(lr) + depth;
}

bool in_zone(const Detection& d, int w, int h, const Zone& z) {
  const float nx = center_x(d) / std::max(w, 1);
  const float ny = center_y(d) / std::max(h, 1);
  return nx >= z.x1 && nx <= z.x2 && ny >= z.y1 && ny <= z.y2;
}

float iou(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  const float aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float bb = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  return inter / (aa + bb - inter + 1e-9f);
}

}  // namespace

Observation analyze(const std::vector<Detection>& dets, int w, int h, const Zone& zone) {
  Observation obs;
  obs.width = w;
  obs.height = h;
  obs.dets = dets;

  std::vector<Detection> persons, vehicles;
  for (const auto& d : dets) {
    if (is_person(d.label)) persons.push_back(d);
    if (is_vehicle(d.label)) vehicles.push_back(d);
  }

  std::ostringstream loc;
  const size_t n = std::min<size_t>(dets.size(), 8);
  obs.positions.reserve(dets.size());
  for (const auto& d : dets) obs.positions.push_back(position_text(d, w, h));
  for (size_t i = 0; i < n; ++i) {
    if (i) loc << "，";
    const std::string pos = position_text(dets[i], w, h);
    loc << dets[i].label << "(" << std::fixed << std::setprecision(2) << dets[i].score
        << ")@" << pos;
  }
  obs.location_text = n ? loc.str() : "无目标";

  bool danger = false, warning = false;
  for (const auto& p : persons) {
    if (in_zone(p, w, h, zone)) {
      obs.rules.emplace_back("人员进入危险区域");
      danger = true;
      break;
    }
  }
  bool near = false;
  for (const auto& p : persons) {
    for (const auto& v : vehicles) {
      if (iou(p, v) > 0.05f) {
        near = true;
        break;
      }
    }
    if (near) break;
  }
  if (near) {
    obs.rules.emplace_back("人车距离过近");
    warning = true;
  }
  if (persons.size() >= 4) {
    obs.rules.emplace_back("区域人员聚集");
    warning = true;
  }
  if (persons.empty() && !vehicles.empty()) obs.rules.emplace_back("区域内有车辆活动但无人");

  obs.risk_level = danger ? "danger" : (warning ? "warning" : "normal");
  return obs;
}

std::string build_prompt(const Observation& obs, const std::string& question) {
  // Prompt 长度直接决定 TTFT（prefill 在 NPU 上跑），所以只留 4 条摘要，
  // 不放 few-shot。实测这版比"6 条完整描述 + few-shot"快 3 倍以上。
  std::ostringstream dets;
  const size_t n = std::min<size_t>(obs.dets.size(), 4);
  for (size_t i = 0; i < n; ++i) {
    if (i) dets << "、";
    const std::string pos = i < obs.positions.size() ? obs.positions[i] : "";
    dets << obs.dets[i].label << std::fixed << std::setprecision(2) << obs.dets[i].score
         << "(" << pos << ")";
  }
  if (n == 0) dets << "无目标";

  std::ostringstream rules;
  const size_t rn = std::min<size_t>(obs.rules.size(), 3);
  for (size_t i = 0; i < rn; ++i) {
    if (i) rules << "；";
    rules << obs.rules[i];
  }
  if (rn == 0) rules << "无";

  const std::string task = question.empty() ? "用一句中文预警当前风险" : question;

  std::ostringstream p;
  p << "你是边缘设备上的安全监测助手，只依据观测作答。\n"
    << "观测：" << dets.str() << "\n"
    << "规则：" << rules.str() << "｜风险=" << obs.risk_level << "\n"
    << "任务：" << task << "，不超过25字，不要解释。\n"
    << "输出：";
  return p.str();
}

}  // namespace mm

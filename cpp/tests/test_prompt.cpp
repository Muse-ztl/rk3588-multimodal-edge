#include "scene.h"

#include <string>
#include <vector>

#include "test_framework.h"

using namespace mm;

namespace {

Observation obs_with(int n_dets, int n_rules, const std::string& risk = "warning") {
  Observation o;
  o.width = 1000;
  o.height = 1000;
  o.risk_level = risk;
  const char* labels[] = {"person", "car", "bus", "truck", "bicycle", "dog"};
  for (int i = 0; i < n_dets && i < 6; ++i) {
    Detection d;
    d.label = labels[i];
    d.score = 0.5f + 0.05f * i;
    o.dets.push_back(d);
    o.positions.push_back("中央近处");
  }
  for (int i = 0; i < n_rules; ++i) o.rules.push_back("rule" + std::to_string(i));
  return o;
}

}  // namespace

TEST(prompt_keeps_only_top_four_detections) {
  const auto p = build_prompt(obs_with(6, 1));
  CHECK(p.find("person") != std::string::npos);
  CHECK(p.find("car") != std::string::npos);
  CHECK(p.find("bus") != std::string::npos);
  CHECK(p.find("truck") != std::string::npos);
  CHECK(p.find("bicycle") == std::string::npos);
  CHECK(p.find("dog") == std::string::npos);
}

TEST(prompt_carries_risk_level_and_ends_with_output_marker) {
  const auto p = build_prompt(obs_with(2, 2, "danger"));
  CHECK(p.find("风险=danger") != std::string::npos);
  CHECK(p.size() >= 9);
  CHECK(p.compare(p.size() - 9, 9, "输出：") == 0);
}

TEST(prompt_truncates_rules_to_three) {
  const auto p = build_prompt(obs_with(1, 5));
  CHECK(p.find("rule2") != std::string::npos);
  CHECK(p.find("rule3") == std::string::npos);
}

TEST(prompt_handles_empty_scene) {
  const auto p = build_prompt(obs_with(0, 0, "normal"));
  CHECK(p.find("无目标") != std::string::npos);
  CHECK(p.find("规则：无") != std::string::npos);
}

TEST(prompt_accepts_a_custom_question) {
  const auto p = build_prompt(obs_with(1, 0), "请给出四十字以内的处置建议");
  CHECK(p.find("请给出四十字以内的处置建议") != std::string::npos);
  CHECK(p.find("用一句中文预警当前风险") == std::string::npos);
}

TEST(prompt_stays_short_for_ttft) {
  // prefill 在 NPU 上跑，prompt 越长首 token 越慢；这条测试防止有人把描述堆回去
  const auto p = build_prompt(obs_with(6, 5, "danger"));
  CHECK(p.size() < 600);
}

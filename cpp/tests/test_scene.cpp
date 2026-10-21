#include "scene.h"

#include <string>
#include <vector>

#include "test_framework.h"

using namespace mm;

namespace {

Detection det(float x1, float y1, float x2, float y2, const std::string& label,
              float score = 0.9f) {
  Detection d;
  d.x1 = x1;
  d.y1 = y1;
  d.x2 = x2;
  d.y2 = y2;
  d.label = label;
  d.score = score;
  return d;
}

bool has_rule(const Observation& o, const std::string& text) {
  for (const auto& r : o.rules) {
    if (r == text) return true;
  }
  return false;
}

}  // namespace

TEST(scene_flags_person_in_danger_zone) {
  const auto o = analyze({det(450, 700, 550, 950, "person")}, 1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("danger"));
  CHECK(has_rule(o, "人员进入危险区域"));
}

TEST(scene_zone_is_inclusive_at_the_corner) {
  // 中心正好落在 (0.2w, 0.55h)
  const auto o = analyze({det(190, 530, 210, 570, "person")}, 1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("danger"));
}

TEST(scene_flags_person_and_vehicle_overlap) {
  // 人车都在画面上方，均不在危险区
  const auto o = analyze({det(100, 100, 200, 300, "person"),
                          det(120, 150, 260, 320, "car")},
                         1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("warning"));
  CHECK(has_rule(o, "人车距离过近"));
}

TEST(scene_flags_crowd) {
  const auto o = analyze({det(50, 50, 120, 200, "person"),
                          det(150, 50, 220, 200, "person"),
                          det(250, 50, 320, 200, "person"),
                          det(350, 50, 420, 200, "person")},
                         1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("warning"));
  CHECK(has_rule(o, "区域人员聚集"));
}

TEST(scene_is_normal_when_empty) {
  const auto o = analyze({}, 1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("normal"));
  CHECK_EQ(o.location_text, std::string("无目标"));
  CHECK(o.rules.empty());
}

TEST(scene_notes_vehicle_without_people) {
  const auto o = analyze({det(50, 50, 200, 200, "car")}, 1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("normal"));
  CHECK(has_rule(o, "区域内有车辆活动但无人"));
}

TEST(scene_position_text_uses_left_and_near) {
  const auto o = analyze({det(0, 500, 100, 950, "person")}, 1000, 1000, Zone{});
  CHECK_EQ(o.positions.size(), 1u);
  CHECK_EQ(o.positions[0], std::string("左侧近处"));
  CHECK_EQ(o.location_text, std::string("person(0.90)@左侧近处"));
}

TEST(scene_danger_beats_warning) {
  const auto o = analyze({det(450, 700, 550, 950, "person"),
                          det(120, 150, 260, 320, "car")},
                         1000, 1000, Zone{});
  CHECK_EQ(o.risk_level, std::string("danger"));
}

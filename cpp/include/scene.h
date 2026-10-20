// 空间规则 + 结构化 Prompt：几何判断留在这里，LLM 只负责把事实讲成人话。
#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace mm {

// 归一化危险区（画面下方中央）
struct Zone {
  float x1 = 0.20f, y1 = 0.55f, x2 = 0.80f, y2 = 1.0f;
};

Observation analyze(const std::vector<Detection>& dets, int w, int h,
                    const Zone& zone = Zone{});

std::string build_prompt(const Observation& obs, const std::string& question = "");

}  // namespace mm

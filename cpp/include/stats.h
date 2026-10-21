// 延迟统计：报告和 bench 共用同一套分位数口径。
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mm {

// 最近秩百分位：排序后取 floor(p/100 * (n-1))，和旧报告保持一致。
inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double q = std::min(1.0, std::max(0.0, p / 100.0));
  const size_t i = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
  return v[i];
}

inline double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

}  // namespace mm

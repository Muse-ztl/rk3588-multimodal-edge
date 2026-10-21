#include "stats.h"

#include <vector>

#include "test_framework.h"

using namespace mm;

TEST(percentile_matches_nearest_rank) {
  const std::vector<double> v{1, 2, 3, 4};
  CHECK_NEAR(percentile(v, 0), 1.0, 1e-9);
  CHECK_NEAR(percentile(v, 50), 2.0, 1e-9);
  CHECK_NEAR(percentile(v, 95), 3.0, 1e-9);
  CHECK_NEAR(percentile(v, 100), 4.0, 1e-9);
}

TEST(percentile_handles_empty_and_single_sample) {
  CHECK_NEAR(percentile({}, 95), 0.0, 1e-9);
  CHECK_NEAR(percentile({42.0}, 95), 42.0, 1e-9);
}

TEST(percentile_tail_is_not_max) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  CHECK_NEAR(percentile(v, 50), 50.0, 1e-9);
  CHECK_NEAR(percentile(v, 95), 95.0, 1e-9);
  CHECK_NEAR(percentile(v, 99), 99.0, 1e-9);
}

TEST(mean_ignores_empty_input) {
  CHECK_NEAR(mean({}), 0.0, 1e-9);
  CHECK_NEAR(mean({1.0, 2.0, 6.0}), 3.0, 1e-9);
}

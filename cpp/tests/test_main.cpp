#include <cstdio>
#include <exception>

#include "test_framework.h"

int main() {
  int failed = 0;
  for (const auto& c : test::all()) {
    try {
      c.fn();
      std::printf("[ ok ] %s\n", c.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[fail] %s\n       %s\n", c.name, e.what());
    }
  }
  std::printf("\n%zu tests, %d failed\n", test::all().size(), failed);
  return failed == 0 ? 0 : 1;
}

// 极简测试框架：不引第三方依赖，CI 里一条命令就能跑。
#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& all() {
  static std::vector<Case> cases;
  return cases;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { all().push_back({name, fn}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void fail(const char* file, int line, const std::string& msg) {
  std::ostringstream os;
  os << file << ":" << line << "  " << msg;
  throw Failure(os.str());
}

template <typename T>
std::string show(const T& v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

}  // namespace test

#define TEST(name)                                        \
  static void name();                                     \
  static ::test::Registrar reg_##name(#name, name);       \
  static void name()

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) ::test::fail(__FILE__, __LINE__, "CHECK failed: " #cond);       \
  } while (0)

#define CHECK_EQ(a, b)                                                           \
  do {                                                                           \
    const auto va = (a);                                                          \
    const auto vb = (b);                                                          \
    if (!(va == vb))                                                             \
      ::test::fail(__FILE__, __LINE__,                                           \
                   std::string("CHECK_EQ failed: " #a " == " #b " (") +          \
                       ::test::show(va) + " vs " + ::test::show(vb) + ")");      \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                    \
  do {                                                                           \
    const double va = static_cast<double>(a);                                    \
    const double vb = static_cast<double>(b);                                    \
    if (!(std::abs(va - vb) <= (eps)))                                           \
      ::test::fail(__FILE__, __LINE__,                                           \
                   std::string("CHECK_NEAR failed: " #a " ~ " #b " (") +         \
                       ::test::show(va) + " vs " + ::test::show(vb) + ")");      \
  } while (0)

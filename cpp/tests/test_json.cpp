#include "json.h"

#include <string>

#include "test_framework.h"

using namespace mm;

TEST(json_escapes_quotes_and_backslashes) {
  CHECK_EQ(json_escape("a\"b\\c"), std::string("a\\\"b\\\\c"));
}

TEST(json_escapes_control_characters) {
  CHECK_EQ(json_escape("line1\nline2"), std::string("line1\\nline2"));
  CHECK_EQ(json_escape("\x01"), std::string("\\u0001"));
}

TEST(json_keeps_utf8_text_untouched) {
  const std::string zh = "人员密集，请注意安全";
  CHECK_EQ(json_escape(zh), zh);
}

TEST(json_escaped_line_stays_one_line) {
  const std::string text = "高危\"警报\"\n第二行";
  const std::string line = "{\"text\":\"" + json_escape(text) + "\"}";
  CHECK(line.find('\n') == std::string::npos);
  CHECK_EQ(line, std::string("{\"text\":\"高危\\\"警报\\\"\\n第二行\"}"));
}

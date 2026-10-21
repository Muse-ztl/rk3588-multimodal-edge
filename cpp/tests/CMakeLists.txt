# 规则和 prompt 都是纯逻辑，不需要 OpenCV / RKNN，x86 上直接编译运行。
add_executable(mm_tests
  test_main.cpp
  test_scene.cpp
  test_prompt.cpp
  test_stats.cpp
  test_json.cpp
  ${CMAKE_SOURCE_DIR}/src/scene.cpp
)

target_include_directories(mm_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME mm_core COMMAND mm_tests)

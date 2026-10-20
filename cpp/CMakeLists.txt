cmake_minimum_required(VERSION 3.16)
project(mm_edge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ---------------------------------------------------------------- OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)

# ---------------------------------------------------------------- RKNN
# rknn_api.h 随仓库提供（Rockchip 官方头文件），运行库用板上的 librknnrt.so（需 2.3.2）
set(RKNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/rknn/include" CACHE PATH "rknn headers")
find_library(RKNN_LIB NAMES rknnrt PATHS /usr/lib /usr/local/lib REQUIRED)

# ---------------------------------------------------------------- ONNX Runtime
# 从官方 release 解压，例如：
#   wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz
#   tar xzf onnxruntime-linux-aarch64-1.23.2.tgz -C ~/deps
set(ORT_ROOT "$ENV{ONNXRUNTIME_DIR}" CACHE PATH "onnxruntime C++ sdk root")
if(NOT ORT_ROOT)
  set(ORT_ROOT "$ENV{HOME}/deps/onnxruntime-linux-aarch64-1.23.2")
endif()
message(STATUS "ONNXRUNTIME_DIR = ${ORT_ROOT}")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ORT_ROOT}/include" REQUIRED)
find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" REQUIRED)

# ---------------------------------------------------------------- RKLLM
# 板上 /usr/include/rkllm.h 是旧版（1.0），必须优先用 rknn-llm 自带的 1.2.3 头文件
set(RKLLM_RUNTIME_INC "$ENV{HOME}/rknn-llm/rkllm-runtime/Linux/librkllm_api/include")
if(EXISTS "${RKLLM_RUNTIME_INC}/rkllm.h")
  set(RKLLM_INCLUDE_DIR "${RKLLM_RUNTIME_INC}")
else()
  find_path(RKLLM_INCLUDE_DIR rkllm.h PATHS /usr/include REQUIRED)
endif()
message(STATUS "RKLLM include = ${RKLLM_INCLUDE_DIR}")
# 注意：/usr/local/lib 下的 librkllmrt.so 才是 1.2.3（/usr/lib 那份是旧的）
find_library(RKLLM_LIB NAMES rkllmrt PATHS /usr/local/lib /usr/lib REQUIRED)

add_executable(mm_edge
  src/main.cpp
  src/rfdetr_split.cpp
  src/scene.cpp
  src/llm.cpp
  src/pipeline.cpp
)

target_include_directories(mm_edge PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${RKNN_INCLUDE_DIR}
  ${ORT_INCLUDE_DIR}
  ${RKLLM_INCLUDE_DIR}
)

target_link_libraries(mm_edge PRIVATE
  ${OpenCV_LIBS}
  ${RKNN_LIB}
  ${ORT_LIB}
  ${RKLLM_LIB}
  pthread
)

set_target_properties(mm_edge PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib;/usr/local/lib"
  INSTALL_RPATH "${ORT_ROOT}/lib;/usr/local/lib"
)

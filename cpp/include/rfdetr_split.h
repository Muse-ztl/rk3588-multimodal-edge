// RF-DETR split inference in C++: DINOv2 backbone on RKNN/NPU, transformer head on
// ONNX Runtime/CPU. This is the deployed path; scripts/*.py only produce the models.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common.h"

namespace mm {

struct RknnCtx;
struct OrtHead;

class SplitRFDETR {
 public:
  SplitRFDETR(const std::string& backbone_rknn, const std::string& head_onnx,
              float threshold = 0.45f, int num_select = 300, int ort_threads = 4,
              int npu_core_mask = 0, int input_size = 0, bool zero_copy = true);
  ~SplitRFDETR();

  SplitRFDETR(const SplitRFDETR&) = delete;
  SplitRFDETR& operator=(const SplitRFDETR&) = delete;

  std::vector<Detection> infer(const cv::Mat& bgr);
  const Timing& timing() const { return timing_; }
  int input_size() const { return input_hw_; }

 private:
  std::vector<float> preprocess(const cv::Mat& bgr) const;
  void run_backbone(const std::vector<float>& input, std::vector<std::vector<float>>& feats,
                    std::vector<std::array<int, 4>>& shapes);
  void run_head(const std::vector<std::vector<float>>& feats,
                const std::vector<std::array<int, 4>>& shapes,
                std::vector<float>& boxes, std::vector<float>& logits);
  std::vector<Detection> postprocess(const std::vector<float>& boxes,
                                     const std::vector<float>& logits, int w, int h) const;

  std::unique_ptr<RknnCtx> rknn_;
  std::unique_ptr<OrtHead> head_;
  float threshold_;
  int num_select_;
  int input_hw_;
  Timing timing_;
};

}  // namespace mm

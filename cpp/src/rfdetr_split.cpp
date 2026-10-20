#include "rfdetr_split.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <rknn_api.h>

namespace mm {
namespace {

constexpr int kMaxDets = 100;
constexpr float kNmsIou = 0.65f;
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

const char* kCoco[] = {
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

}  // namespace

const char* coco_name(int idx) {
  const int n = static_cast<int>(sizeof(kCoco) / sizeof(kCoco[0]));
  return (idx >= 0 && idx < n) ? kCoco[idx] : "class";
}

// ---------------------------------------------------------------- RKNN 部分

struct RknnCtx {
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  rknn_tensor_attr in_attr{};
  rknn_tensor_attr out_attr[8]{};
  bool alive = false;
};

// ---------------------------------------------------------------- ORT 部分

struct OrtHead {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mm_edge"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> input_shapes;
};

SplitRFDETR::SplitRFDETR(const std::string& backbone_rknn, const std::string& head_onnx,
                         float threshold, int num_select, int ort_threads,
                         int npu_core_mask, int input_size)
    : threshold_(threshold), num_select_(num_select) {
  // --- backbone
  rknn_ = std::make_unique<RknnCtx>();
  FILE* fp = std::fopen(backbone_rknn.c_str(), "rb");
  if (fp == nullptr) throw std::runtime_error("cannot open backbone rknn: " + backbone_rknn);
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> model(sz);
  if (std::fread(model.data(), 1, sz, fp) != static_cast<size_t>(sz)) {
    std::fclose(fp);
    throw std::runtime_error("short read on rknn model");
  }
  std::fclose(fp);

  rknn_init_extend extend{};
  int ret = rknn_init(&rknn_->ctx, model.data(), static_cast<uint32_t>(model.size()),
                      RKNN_FLAG_PRIOR_MEDIUM, &extend);
  if (ret != RKNN_SUCC) throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
  rknn_->alive = true;

  if (npu_core_mask > 0) {
    rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
    switch (npu_core_mask) {
      case 1: mask = RKNN_NPU_CORE_0; break;
      case 2: mask = RKNN_NPU_CORE_1; break;
      case 4: mask = RKNN_NPU_CORE_2; break;
      case 3: mask = RKNN_NPU_CORE_0_1; break;
      case 7: mask = RKNN_NPU_CORE_0_1_2; break;
      default: break;
    }
    rknn_set_core_mask(rknn_->ctx, mask);
  }

  rknn_query(rknn_->ctx, RKNN_QUERY_IN_OUT_NUM, &rknn_->io_num, sizeof(rknn_->io_num));
  rknn_->in_attr.index = 0;
  rknn_query(rknn_->ctx, RKNN_QUERY_INPUT_ATTR, &rknn_->in_attr, sizeof(rknn_tensor_attr));
  input_hw_ = input_size > 0 ? input_size : static_cast<int>(rknn_->in_attr.dims[2]);
  for (uint32_t i = 0; i < rknn_->io_num.n_output && i < 8; ++i) {
    rknn_->out_attr[i].index = i;
    rknn_query(rknn_->ctx, RKNN_QUERY_OUTPUT_ATTR, &rknn_->out_attr[i],
               sizeof(rknn_tensor_attr));
  }

  // --- head
  head_ = std::make_unique<OrtHead>();
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(ort_threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  head_->session = std::make_unique<Ort::Session>(head_->env, head_onnx.c_str(), opts);

  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < head_->session->GetInputCount(); ++i) {
    head_->input_names.emplace_back(head_->session->GetInputNameAllocated(i, alloc).get());
    head_->input_shapes.push_back(
        head_->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < head_->session->GetOutputCount(); ++i) {
    head_->output_names.emplace_back(head_->session->GetOutputNameAllocated(i, alloc).get());
  }
}

SplitRFDETR::~SplitRFDETR() {
  if (rknn_ && rknn_->alive) rknn_destroy(rknn_->ctx);
}

std::vector<float> SplitRFDETR::preprocess(const cv::Mat& bgr) const {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(input_hw_, input_hw_), 0, 0, cv::INTER_LINEAR);
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // RKNN 的 normalize 只接受 NHWC 源布局（NCHW 会直接报错），
  // 且模型配置里 mean=0/std=1，这里只做归一化+排布，不做重复缩放。
  std::vector<float> out(static_cast<size_t>(input_hw_) * input_hw_ * 3);
  for (int y = 0; y < input_hw_; ++y) {
    const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
    for (int x = 0; x < input_hw_; ++x) {
      for (int c = 0; c < 3; ++c) {
        out[(static_cast<size_t>(y) * input_hw_ + x) * 3 + c] = (row[x][c] - kMean[c]) / kStd[c];
      }
    }
  }
  return out;
}

void SplitRFDETR::run_backbone(const std::vector<float>& input,
                               std::vector<std::vector<float>>& feats,
                               std::vector<std::array<int, 4>>& shapes) {
  rknn_input in{};
  in.index = 0;
  in.buf = const_cast<float*>(input.data());
  in.size = static_cast<uint32_t>(input.size() * sizeof(float));
  in.type = RKNN_TENSOR_FLOAT32;
  in.fmt = RKNN_TENSOR_NHWC;
  in.pass_through = 0;
  int ret = rknn_inputs_set(rknn_->ctx, 1, &in);
  if (ret != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));

  ret = rknn_run(rknn_->ctx, nullptr);
  if (ret != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(ret));

  std::vector<rknn_output> outs(rknn_->io_num.n_output);
  for (uint32_t i = 0; i < rknn_->io_num.n_output; ++i) {
    outs[i].index = i;
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  ret = rknn_outputs_get(rknn_->ctx, rknn_->io_num.n_output, outs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));
  }

  feats.clear();
  shapes.clear();
  for (uint32_t i = 0; i < rknn_->io_num.n_output; ++i) {
    const auto& attr = rknn_->out_attr[i];
    const size_t n = outs[i].size / sizeof(float);
    std::vector<float> f(n);
    std::memcpy(f.data(), outs[i].buf, outs[i].size);
    feats.push_back(std::move(f));
    shapes.push_back({static_cast<int>(attr.dims[0]), static_cast<int>(attr.dims[1]),
                      static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[3])});
  }
  rknn_outputs_release(rknn_->ctx, rknn_->io_num.n_output, outs.data());
}

void SplitRFDETR::run_head(const std::vector<std::vector<float>>& feats,
                           const std::vector<std::array<int, 4>>& shapes,
                           std::vector<float>& boxes, std::vector<float>& logits) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // 先把所有输入张量的数据准备好（需要转置的在这里转），保证内存活到 Run 返回。
  std::vector<std::vector<float>> storage(head_->input_names.size());
  for (size_t i = 0; i < head_->input_names.size(); ++i) {
    const auto& src = feats[i];
    const auto& s4 = shapes[i];
    const auto shape = head_->input_shapes[i];
    // RKNN 输出 NHWC，检测头要 NCHW：用 head 声明的形状反推是否需要转置。
    const bool need_nchw = shape.size() == 4 && shape[1] > 0 && shape[1] == s4[3] &&
                           shape[2] == s4[1] && shape[3] == s4[2];
    if (need_nchw) {
      const int H = s4[1], W = s4[2], C = s4[3];
      storage[i].resize(src.size());
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          for (int c = 0; c < C; ++c)
            storage[i][static_cast<size_t>(c) * H * W + h * W + w] =
                src[static_cast<size_t>(h) * W * C + w * C + c];
    } else {
      storage[i] = src;
    }
  }

  std::vector<Ort::Value> inputs;
  inputs.reserve(storage.size());
  for (size_t i = 0; i < storage.size(); ++i) {
    auto shape = head_->input_shapes[i];
    inputs.push_back(Ort::Value::CreateTensor<float>(mem, storage[i].data(), storage[i].size(),
                                                     shape.data(), shape.size()));
  }

  std::vector<const char*> in_names, out_names;
  for (auto& s : head_->input_names) in_names.push_back(s.c_str());
  for (auto& s : head_->output_names) out_names.push_back(s.c_str());

  auto outputs = head_->session->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                     inputs.size(), out_names.data(), out_names.size());
  const auto* b = outputs[0].GetTensorData<float>();
  const auto bshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto* l = outputs[1].GetTensorData<float>();
  const auto lshape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  boxes.assign(b, b + bshape[0] * bshape[1] * bshape[2]);
  logits.assign(l, l + lshape[0] * lshape[1] * lshape[2]);
}

std::vector<Detection> SplitRFDETR::postprocess(const std::vector<float>& boxes,
                                                const std::vector<float>& logits, int w,
                                                int h) const {
  const int num_q = static_cast<int>(boxes.size() / 4);
  const int num_c = static_cast<int>(logits.size() / std::max(num_q, 1));

  struct Cand { float score; int q; int c; };
  std::vector<Cand> cands;
  cands.reserve(static_cast<size_t>(num_q) * num_c);
  for (int q = 0; q < num_q; ++q) {
    for (int c = 0; c < num_c; ++c) {
      const float p = sigmoid(logits[static_cast<size_t>(q) * num_c + c]);
      if (p > threshold_) cands.push_back({p, q, c});
    }
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    return a.score > b.score;
  });
  if (static_cast<int>(cands.size()) > num_select_) cands.resize(num_select_);

  std::vector<cv::Rect> rects;
  std::vector<float> scores;
  std::vector<int> classes;
  rects.reserve(cands.size());
  for (const auto& cd : cands) {
    const float cx = boxes[static_cast<size_t>(cd.q) * 4 + 0];
    const float cy = boxes[static_cast<size_t>(cd.q) * 4 + 1];
    const float bw = boxes[static_cast<size_t>(cd.q) * 4 + 2];
    const float bh = boxes[static_cast<size_t>(cd.q) * 4 + 3];
    const float x1 = std::max(0.f, (cx - bw / 2) * w);
    const float y1 = std::max(0.f, (cy - bh / 2) * h);
    const float x2 = std::min(static_cast<float>(w), (cx + bw / 2) * w);
    const float y2 = std::min(static_cast<float>(h), (cy + bh / 2) * h);
    rects.emplace_back(cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                cv::Point(static_cast<int>(x2), static_cast<int>(y2))));
    scores.push_back(cd.score);
    classes.push_back(cd.c);
  }

  std::vector<Detection> out;
  if (rects.empty()) return out;
  std::vector<int> keep;
  cv::dnn::NMSBoxes(rects, scores, threshold_, kNmsIou, keep);
  for (int idx : keep) {
    if (static_cast<int>(out.size()) >= kMaxDets) break;
    Detection d;
    d.x1 = static_cast<float>(rects[idx].x);
    d.y1 = static_cast<float>(rects[idx].y);
    d.x2 = static_cast<float>(rects[idx].x + rects[idx].width);
    d.y2 = static_cast<float>(rects[idx].y + rects[idx].height);
    d.score = scores[idx];
    d.class_id = classes[idx];
    d.label = coco_name(classes[idx]);
    out.push_back(d);
  }
  return out;
}

std::vector<Detection> SplitRFDETR::infer(const cv::Mat& bgr) {
  const double t0 = now_ms();
  auto input = preprocess(bgr);
  const double t1 = now_ms();

  std::vector<std::vector<float>> feats;
  std::vector<std::array<int, 4>> shapes;
  run_backbone(input, feats, shapes);
  const double t2 = now_ms();

  std::vector<float> boxes, logits;
  run_head(feats, shapes, boxes, logits);
  const double t3 = now_ms();

  auto dets = postprocess(boxes, logits, bgr.cols, bgr.rows);
  const double t4 = now_ms();

  timing_.preprocess_ms = t1 - t0;
  timing_.backbone_ms = t2 - t1;
  timing_.head_ms = t3 - t2;
  timing_.postprocess_ms = t4 - t3;
  timing_.total_ms = t4 - t0;
  return dets;
}

}  // namespace mm

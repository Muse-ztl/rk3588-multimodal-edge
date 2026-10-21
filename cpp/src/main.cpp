// 命令行入口：mm_edge --video ... --backbone ... --head ... --llm ...
//
// 子命令式参数：
//   --bench N        只跑检测基准（配合 --image）
//   --image PATH     单图推理
//   --duration SEC   限制运行时长（0 = 跑完整段视频）
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "common.h"
#include "llm.h"
#include "pipeline.h"
#include "rfdetr_split.h"
#include "stats.h"

namespace {

struct Args {
  std::string image = "data/sample.jpg";
  std::string video = "data/demo_scene.mp4";
  std::string backbone = "models/backbone.rknn";
  std::string head = "models/detection.onnx";
  std::string llm_model;
  std::string out = "output";
  std::string question;
  float threshold = 0.45f;
  int stride = 1;
  int bench = 0;
  int core_mask = 0;
  std::vector<int> cv_cores = {4, 5, 6, 7};
  double duration = 0.0;
  bool no_loop = false;
  bool single_image = false;
};

std::vector<int> parse_cores(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (char c : s + ",") {
    if (c == ',') {
      if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  return out;
}

void usage() {
  printf(
      "usage: mm_edge [options]\n"
      "  --image PATH       single image for --bench/--image mode\n"
      "  --video PATH       video source (default data/demo_scene.mp4)\n"
      "  --backbone PATH    backbone.rknn (default models/backbone.rknn)\n"
      "  --head PATH        detection.onnx (default models/detection.onnx)\n"
      "  --llm PATH         Qwen2.5-1.5B.rkllm (empty = vision only)\n"
      "  --out DIR          output dir (default output)\n"
      "  --stride N         inference stride (default 1)\n"
      "  --threshold F      detection score threshold (default 0.45)\n"
      "  --core-mask N      RKNN core mask: 0 auto / 1 core0 / 3 core0+1 / 7 all\n"
      "  --cv-cores a,b,c   CPU affinity of the CV thread (default 4,5,6,7)\n"
      "  --duration SEC     stop after SEC seconds (0 = whole video)\n"
      "  --no-loop          do not loop the video\n"
      "  --bench N          run N detections on --image and print stage latency\n");
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (k == "--image") a.image = next();
    else if (k == "--video") a.video = next();
    else if (k == "--backbone") a.backbone = next();
    else if (k == "--head") a.head = next();
    else if (k == "--llm") a.llm_model = next();
    else if (k == "--out") a.out = next();
    else if (k == "--question") a.question = next();
    else if (k == "--threshold") a.threshold = std::atof(next().c_str());
    else if (k == "--stride") a.stride = std::atoi(next().c_str());
    else if (k == "--bench") a.bench = std::atoi(next().c_str());
    else if (k == "--core-mask") a.core_mask = std::atoi(next().c_str());
    else if (k == "--cv-cores") a.cv_cores = parse_cores(next());
    else if (k == "--duration") a.duration = std::atof(next().c_str());
    else if (k == "--no-loop") a.no_loop = true;
    else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
  }
  return a;
}

int run_bench(const Args& a) {
  cv::Mat img = cv::imread(a.image);
  if (img.empty()) {
    fprintf(stderr, "cannot read %s\n", a.image.c_str());
    return 1;
  }
  mm::SplitRFDETR det(a.backbone, a.head, a.threshold, 300, 4, a.core_mask);
  std::vector<double> total, npu, head, pre, post;
  std::vector<mm::Detection> dets;
  for (int i = 0; i < a.bench + 3; ++i) {
    dets = det.infer(img);
    if (i < 2) continue;  // 前两次是预热
    const auto& t = det.timing();
    total.push_back(t.total_ms);
    npu.push_back(t.backbone_ms);
    head.push_back(t.head_ms);
    pre.push_back(t.preprocess_ms);
    post.push_back(t.postprocess_ms);
  }
  printf("[bench] %d runs  backbone_npu=%.1f/%.1fms  head_cpu=%.1f/%.1fms  "
         "pre=%.1fms  post=%.1fms  total=%.1f/%.1fms (p50/p95)\n",
         a.bench, mm::percentile(npu, 50), mm::percentile(npu, 95),
         mm::percentile(head, 50), mm::percentile(head, 95), mm::percentile(pre, 50),
         mm::percentile(post, 50), mm::percentile(total, 50), mm::percentile(total, 95));
  printf("[bench] %.1f fps, %zu detections:", 1000.0 / std::max(1e-6, mm::percentile(total, 50)),
         dets.size());
  for (size_t i = 0; i < dets.size() && i < 6; ++i) {
    printf(" %s(%.2f)", dets[i].label.c_str(), dets[i].score);
  }
  printf("\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (a.bench > 0) return run_bench(a);

  try {
    std::filesystem::create_directories(a.out);
    mm::SplitRFDETR det(a.backbone, a.head, a.threshold, 300, 4, a.core_mask);
    printf("[mm_edge] detector ready, input=%d\n", det.input_size());

    std::unique_ptr<mm::LLMEngine> llm;
    if (!a.llm_model.empty()) {
      llm = std::make_unique<mm::LLMEngine>(a.llm_model, 48, 2048);
      const double t0 = mm::now_ms();
      llm->run("你好");  // 预热：把 NPU 权重换入的开销摊掉，不污染 TTFT 统计
      printf("[mm_edge] llm ready: %s (warmup %.1f ms)\n", a.llm_model.c_str(),
             mm::now_ms() - t0);
    }

    mm::Pipeline pipe(det, llm.get(), a.video, a.out, a.stride, 15, !a.no_loop, a.cv_cores);
    auto stats = pipe.run(a.duration);

    printf("\n=== report ===\n");
    printf("frames=%ld infer=%ld affinity=%s\n", stats.frames_read, stats.infer_frames,
           stats.affinity.c_str());
    printf("llm_calls=%ld published=%ld consumed=%ld frames_while_busy=%ld\n", stats.llm_calls,
           stats.observations_published, stats.observations_consumed,
           stats.frames_while_llm_busy);
    printf("report.json / llm_traces.jsonl written to %s/\n", a.out.c_str());
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}

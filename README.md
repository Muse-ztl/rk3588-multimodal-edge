# Edge Multimodal Inference on RK3588 (C++ / RKNN / ONNX Runtime / RKLLM)

在 Orange Pi 5 Pro（RK3588S，16GB）上跑通的「视觉感知 → 大模型决策」多模态推理，**核心链路
全部 C++**：

- **RF-DETR-Nano**：DINOv2 backbone 走 RKNN/NPU，Transformer + 检测头走 ONNX Runtime/CPU（分段推理）
- **Qwen2.5-1.5B**：RKLLM 端侧量化模型，流式生成场景描述与安全预警
- **异构调度**：CV 线程 NPU 高频采样 + LLM 线程异步推理，单槽位背压，端到端秒级

Python 只负责 x86 上的模型导出/转换（`scripts/`），板端运行的是 CMake 构建的单体可执行文件 `mm_edge`。

![demo frame](assets/demo_frame.jpg)

> 上图为 `mm_edge` 实际输出帧（检测框、危险区、risk 标记由 C++ 绘制）；
> 底部那句中文是 LLM 输出，因 OpenCV 内置字体不含 CJK，由 `tools/render_zh.py` 叠加，
> 该脚本只用于生成配图，不参与推理。

## 实测（Orange Pi 5 Pro / RK3588S，单核 NPU 运行 backbone）

| 阶段 | 位置 | 延迟 |
|---|---|---|
| Backbone（单尺度 256×24×24 输出） | RKNN / NPU (core0) | **222 ms** |
| Transformer + 检测头（INT8 动态量化） | ONNX Runtime / CPU 4 线程 | **42 ms** |
| 预处理 + 后处理 | CPU | 4 ms |
| 单帧检测端到端 | — | **278 ms**（p50） |
| Qwen2.5-1.5B 首 token（流水线并行时） | RKLLM / NPU | **1.21–1.44 s** |
| Qwen2.5-1.5B 单次生成 | RKLLM / NPU | 3.8–5.9 s |
| 端到端（观测采集 → 预警输出） | — | **4.22 s**（p50） |

`assets/report_demo.json` 原始数据：45 帧推理 / 3 次 LLM / 观测丢弃 42（背压生效）。

```
frame=1    ttft=1443ms total=5936ms  **高危警报！人员密集区存在安全隐患。**
frame=81   ttft=1219ms total=4217ms  人员密集且部分区域存在安全风险。
frame=137  ttft=1409ms total=3786ms  人员密集区域，人车距离过近，请注意安全。
```

## 架构

```
                ┌──────────────── CV 线程（绑 A55 小核） ────────────────┐
 camera/video → │ preprocess → [RKNN: DINOv2 backbone, core0]           │
                │      → NHWC→NCHW → [ORT: transformer+head, CPU]       │
                │      → postprocess(NMS-free) → 规则判定 → Observation │
                └───────────────────────┬───────────────────────────────┘
                                        │ 单槽位（只保留最新观测）
                ┌───────────────────────▼───────────────────────────────┐
                │ LLM 线程：build_prompt → [RKLLM: Qwen2.5-1.5B]        │
                │           → 流式回调 → 一句中文预警                    │
                └───────────────────────────────────────────────────────┘
```

## 目录

```
cpp/
  CMakeLists.txt
  include/  common.h scene.h rfdetr_split.h llm.h pipeline.h
  src/      rfdetr_split.cpp   RKNN backbone + ORT head + 后处理
            scene.cpp          空间规则 + 结构化 Prompt
            llm.cpp            RKLLM 封装（流式回调 + TTFT）
            pipeline.cpp       CV/LLM 双线程调度、背压、统计
            main.cpp           命令行入口 + --bench
  third_party/rknn/include/rknn_api.h   Rockchip 官方头文件
scripts/                    x86 侧模型导出（Python）
  export_split_onnx.py      RF-DETR checkpoint → backbone.onnx + detection.onnx
  convert_backbone_rknn.py  backbone.onnx → backbone.rknn
  make_demo_clip.py         合成演示素材
tools/render_zh.py          README 配图用的中文叠加（不参与推理）
docs/quantize_w4a16.md      w4a16 量化说明
```

## 复现

### 1) x86：导出模型

```bash
python3 -m venv ~/venv_rfdetr && source ~/venv_rfdetr/bin/activate
pip install rfdetr onnx onnxruntime supervision
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("Jordan12322/rfdetr_nano", "rfdetr_nano.pth", local_dir="/tmp"))
PY

python scripts/export_split_onnx.py --weights /tmp/rfdetr_nano.pth --out output --skip-full

python3 -m venv ~/rknn_env && source ~/rknn_env/bin/activate
pip install rknn-toolkit2==2.3.2 "setuptools<81" "onnx==1.16.1" "numpy<2"
python scripts/convert_backbone_rknn.py --input output/backbone.onnx --output output/backbone.rknn
```

### 2) 板端：编译运行

```bash
# RKNN 运行时必须是 2.3.2（系统自带 1.4.0 加载不了 2.x 模型）
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

# ONNX Runtime C++ SDK (aarch64)
mkdir -p ~/deps && cd ~/deps
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz
tar xzf onnxruntime-linux-aarch64-1.23.2.tgz

cd ~/mm_edge/cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_DIR=$HOME/deps/onnxruntime-linux-aarch64-1.23.2
make -j4

# 单图基准
./mm_edge --bench 10 --image ../data/sample.jpg \
  --backbone ../models/backbone.rknn --head ../models/detection.onnx --core-mask 1

# 完整流水线
./mm_edge --video ../data/demo_scene.mp4 \
  --backbone ../models/backbone.rknn --head ../models/detection.onnx \
  --llm ~/rkllm_work/model/Qwen2.5-1.5B.rkllm \
  --out ../output --stride 4 --core-mask 1 --cv-cores 0,1,2,3 --no-loop
```

## 已知限制

- 检测头是 ONNX Runtime 动态量化（MatMul/Gemm INT8），没做 QAT；`detection_fp32.onnx` 可对比精度。
- backbone 目前 FP16；`convert_backbone_rknn.py --quantize --dataset dataset.txt` 支持 INT8，
  需自备校准集。
- 演示素材为合成片段（`data/sample.jpg` 缓慢推近），不含真实监控画面。
- LLM 是 Qwen2.5-1.5B **w8a8**（运行时日志 `model_dtype: W8A8`）；w4a16 重导流程见
  `docs/quantize_w4a16.md`。

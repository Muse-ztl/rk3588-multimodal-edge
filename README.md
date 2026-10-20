# Edge Multimodal Inference on RK3588 (RF-DETR + Qwen2.5 + RKLLM)

在 Orange Pi 5 Pro（RK3588S，16GB）上跑通「视觉感知 → 大模型决策」的多模态推理：
**RF-DETR 骨干走 NPU、Transformer 头走 CPU、Qwen2.5-1.5B 走 RKLLM**，CV 线程与 LLM 线程
异步并行，把检测结果结构化后注入 Prompt，由端侧 LLM 输出场景描述与安全预警。

> 这个仓库解决的是一个具体问题：RF-DETR 是 Transformer 检测器，整模型转 RKNN
> 既慢又掉精度；但它的 DINOv2 骨干是纯卷积、在 NPU 上跑得飞快。于是只把骨干
> 交给 NPU，Transformer/检测头留在 CPU 上用 ONNX Runtime 跑，端到端反而更快。

## 实测结果（Orange Pi 5 Pro / RK3588S，RF-DETR-Nano 384×384 + Qwen2.5-1.5B）

| 阶段 | 位置 | 延迟 |
|---|---|---|
| Backbone（单尺度 256×24×24 输出） | RKNN / NPU | **203 ms** |
| Transformer + 检测头（INT8 动态量化） | ONNX Runtime / CPU（4 线程） | **38 ms** |
| 预处理 + 后处理 | CPU | 6 ms |
| 单帧检测端到端 | — | **250 ms**（p50） |
| Qwen2.5-1.5B 首 token（`bench_stages`，预热后） | RKLLM / NPU | **112 ms** |
| Qwen2.5-1.5B 单次生成（约 10–20 token） | RKLLM / NPU | **1.0–1.2 s** |
| 模型加载（1.5B, w8a8） | — | 8.0 s |
| 流水线并行时 LLM 首 token | — | 1.06 s（p50，CV 同时以 3.8 fps 采样） |
| 流水线端到端（观测采集 → 预警输出） | — | 4.5 s（p50） |

实测产物见 `assets/report_demo.json` 与 `assets/llm_traces_demo.jsonl`：

```
frame=1   ttft=1325ms total=4710ms risk=danger 人员密集区存在潜在安全风险，请注意安全。
frame=65  ttft=945ms  total=4462ms risk=danger 人员密集区存在潜在安全威胁，请注意安全。
frame=125 ttft=1061ms total=3884ms risk=danger 人员密集区存在潜在安全威胁，请注意安全。
```

![demo frame](assets/demo_frame.jpg)

> 关于并行的真实代价：视觉与 LLM 共用一颗 NPU（3 core），CV 每跑一帧
> backbone 就会占住 NPU 200 ms，LLM 的 prefill/decode 会被推迟。实测把
> CV 绑到 A55 小核、NPU core mask 限制为单核后，LLM TTFT 从 8.8 s 降到 1.06 s；
> 想再快只能降采样率或给 backbone 上 INT8。这一点在 README 里写清楚，
> 比"端到端毫秒级"更可信。

复现命令见文末。

## 架构

```
                    ┌──────────────── CV 线程（绑 A76 大核） ───────────────┐
 camera / video ──▶ │ 预处理 → [RKNN: DINOv2 backbone] → 3×feature map      │
                    │        → 布局适配 → [ORT: transformer+head] → boxes   │
                    │        → 规则判定（危险区/人车距离）→ Observation      │
                    └───────────────────────┬──────────────────────────────┘
                                            │ 单槽位（只保留最新观测）
                    ┌───────────────────────▼──────────────────────────────┐
                    │ LLM 线程：结构化 Prompt → [RKLLM: Qwen2.5-1.5B]       │
                    │            → 流式输出 → 场景描述 / 预警               │
                    └──────────────────────────────────────────────────────┘
```

关键设计：

- **分段推理**：`backbone.rknn` 只吃 `[1,3,384,384]`，输出 3 个尺度的特征图；检测头以
  这些特征图为输入，位置编码（positional embedding）在图内烘焙成常量，避免每次推理
  重复计算正弦编码。中间那层「RKNN 输出 NHWC / ORT 期望 NCHW」的转换由
  `rfdetr_split._to_nchw()` 按 head 的输入 shape 自动判定，不依赖固定版本行为。
- **规则与语义分离**：几何判定（是否越界、是否人车过近）放在 `zones.py`，LLM 只负责
  把结构化事实讲成人话。1.5B 模型做不了几何推理，这样切职责既准又省 token。
- **背压而非排队**：观测槽位只有 1 个。LLM 一次生成几百毫秒，期间 CV 已产出多帧，
  旧观测对"当前场景"无意义，直接丢弃（统计数据里能看到 `observations_dropped`）。
- **异步并行**：LLM 推理不阻塞采样，`report.json` 里的 `frames_while_llm_busy`
  就是并行度的直接证据。

## 目录

```
src/mm_edge/
  rfdetr_split.py    分段推理（RKNN backbone + ORT head）
  llm_bridge.cpp     RKLLM C API 的极简 C ABI 封装（在板端编译成 .so）
  llm_bridge.py      ctypes 绑定 + 流式回调 + TTFT 统计
  zones.py           空间规则与风险判定
  prompt.py          结构化 Prompt 构造
  scheduler.py       CV/LLM 双线程调度、单槽位背压、指标统计
  viz.py             标注（PIL 渲染中文，cv2 只画框）
scripts/
  export_split_onnx.py     x86：RF-DETR checkpoint → backbone.onnx + detection.onnx
  convert_backbone_rknn.py x86：backbone.onnx → backbone.rknn
  build_llm_bridge.sh      板端：编译 librkllm_bridge.so
  smoke_llm.py             板端：LLM 单测
  bench_stages.py          板端：分阶段延迟基准
  run_demo.py              板端：完整流水线
```

## 复现

### 1) x86 主机（Ubuntu 22.04 / WSL）

```bash
python3 -m venv ~/venv_rfdetr && source ~/venv_rfdetr/bin/activate
pip install rfdetr onnx onnxruntime onnxsim supervision
# RKNN 转换环境和推理环境分开，避免 numpy/onnx 版本互相打架
python3 -m venv ~/rknn_env && source ~/rknn_env/bin/activate
pip install rknn-toolkit2==2.3.2 "setuptools<81" "onnx==1.16.1" "numpy<2"

# 官方 COCO nano 权重（也可从 Roboflow 官方渠道获取）
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("Jordan12322/rfdetr_nano", "rfdetr_nano.pth", local_dir="/tmp"))
PY

python scripts/export_split_onnx.py \
    --weights /tmp/rfdetr_nano.pth --out output --skip-full
python scripts/convert_backbone_rknn.py \
    --input output/backbone.onnx --output output/backbone.rknn
```

### 2) 板端

```bash
# RKNN 运行时必须是 2.3.2（系统自带 1.4.0 加载不了 2.x 模型）
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak
sudo cp /path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

python3 -m venv --system-site-packages ~/mm_env
~/mm_env/bin/pip install rknn-toolkit-lite2==2.3.2 onnxruntime opencv-python-headless pillow

bash scripts/build_llm_bridge.sh build
~/mm_env/bin/python scripts/smoke_llm.py /path/to/Qwen2.5-1.5B.rkllm
~/mm_env/bin/python scripts/bench_stages.py --backbone models/backbone.rknn \
    --head models/detection.onnx --llm /path/to/Qwen2.5-1.5B.rkllm
~/mm_env/bin/python scripts/run_demo.py --video data/demo_scene.mp4 \
    --backbone models/backbone.rknn --head models/detection.onnx \
    --llm /path/to/Qwen2.5-1.5B.rkllm --out output --duration 20
```

## 已知限制

- 检测头使用 ONNX Runtime 动态量化（MatMul/Gemm INT8），没有做 QAT；
  换用 `detection_fp32.onnx` 可对比精度差。
- 演示素材为合成片段（`data/sample.jpg` 三段不同构图），不含真实监控画面。
- backbone 目前是 FP16；`convert_backbone_rknn.py --quantize --dataset dataset.txt`
  支持 INT8，需要自备校准集。
- LLM 为 Qwen2.5-1.5B **w8a8** 模型（`model_dtype: W8A8`，由 rkllm-toolkit 1.2.3 导出）。
  若需要 w4a16，用 `rkllm-toolkit` 重新量化脚本见 `docs/quantize_w4a16.md`。

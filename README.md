# RK3588 边缘多模态推理

视频进来，先检测人和车，再让端侧的 Qwen2.5-1.5B 说一句中文预警。检测和生成都在 Orange Pi 5 Pro 上跑，不依赖云端。

![demo frame](assets/demo_frame.jpg)

上图是 `mm_edge` 实际输出的帧：框、危险区和 risk 标记由 C++ 画；底部那句中文是 LLM 输出，因为 OpenCV 自带字体没有中文字形，配图时用 `tools/render_zh.py` 叠了一行，不参与推理。

![tests](https://github.com/Muse-ztl/rk3588-multimodal-edge/actions/workflows/ci.yml/badge.svg)

## 能做什么

- RF-DETR-Nano 拆成两段：DINOv2 backbone 走 RKNN / NPU，Transformer 和检测头走 CPU 上的 ONNX Runtime
- 规则层判断"人员进入危险区 / 人车过近 / 人员聚集"，LLM 只负责把确定的事实讲成人话
- CV 线程和 LLM 线程异步，观测只保留最新一条，预警不会越积越旧

## 跑起来

x86 上导出模型：

```bash
python3 -m venv ~/venv_rfdetr && source ~/venv_rfdetr/bin/activate
pip install rfdetr onnx onnxruntime supervision
export HF_ENDPOINT=https://hf-mirror.com
python scripts/export_split_onnx.py --weights rfdetr_nano.pth --out output --skip-full

python3 -m venv ~/rknn_env && source ~/rknn_env/bin/activate
pip install rknn-toolkit2==2.3.2 "setuptools<81" "onnx==1.16.1" "numpy<2"
python scripts/convert_backbone_rknn.py --input output/backbone.onnx --output output/backbone.rknn
```

板端需要 RKNN runtime 2.3.2、ONNX Runtime C++ SDK 和 RKLLM 1.2.3：

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_DIR=$HOME/deps/onnxruntime-linux-aarch64-1.23.2
cmake --build build -j4

./build/mm_edge \
  --video ../data/demo_scene.mp4 \
  --backbone ../models/backbone.rknn \
  --head ../models/detection.onnx \
  --llm ~/rkllm_work/model/Qwen2.5-1.5B.rkllm \
  --out ../output --stride 4 --core-mask 1 --cv-cores 0,1,2,3 --no-loop
```

单图基准：`./build/mm_edge --bench 10 --image ../data/sample.jpg ...`

## 实测

Orange Pi 5 Pro（RK3588S，16GB），CV 绑 A55、backbone 绑 NPU core0，stride=4：

| 阶段 | p50 | p95 |
|---|---|---|
| Backbone（NPU） | 222 ms | 238 ms |
| Transformer + 检测头（CPU 4 线程） | 42 ms | 47 ms |
| 单帧检测 | 279 ms | 297 ms |
| LLM 首 token | 1.36 s | 1.38 s |
| 单次生成 | 4.39 s | 4.42 s |
| 端到端 | 4.43 s | 4.44 s |

180 帧视频、45 次推理、3 次 LLM 调用；期间丢掉 42 条观测，说明单槽背压确实在工作。
原始数据在 `assets/report_demo.json`，三句预警在 `assets/llm_traces_demo.jsonl`。

## 目录

```
cpp/             板端 C++ 实现
  src/           rfdetr_split / scene / llm / pipeline / main
  tests/         x86 单测（规则、Prompt、统计、JSON）
scripts/         x86 侧导出和转换
assets/          实测 report、LLM 输出、演示帧
docs/NOTES.md    实现细节和测量口径
```

## 限制

- 演示素材是合成的，不含真实监控画面
- 检测头是 ORT 动态量化，backbone 是 FP16，都没做 QAT
- LLM 是 Qwen2.5-1.5B 的 w8a8；w4a16 转换流程在 `docs/quantize_w4a16.md`，没有在板上跑
- 只处理一路视频，不接摄像头流，也不做多轮对话

## 更多

拆分推理的原因、线程绑核、Prompt 压缩这些内容在 `docs/NOTES.md`。

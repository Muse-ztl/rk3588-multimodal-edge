# 推理优化记录

板子：Orange Pi 5 Pro（RK3588S，16GB），RKNN runtime 2.3.2，ONNX Runtime 1.23.2，
RKLLM 1.2.3。backbone 绑 NPU core0，CV 线程绑 A55（`0,1,2,3`）。原始 JSON 在 `assets/`。

## 测量协议

- `--bench 200`：同一张 `data/sample.jpg`，前两次预热，记录 200 次；p50/p95 最近秩。
- 记录 RSS、进程 CPU 时间、温度、NPU/CPU 频率。板子没有 root，锁不了 governor，p50 有几 ms 波动。
- 单帧延迟拆成：预处理、`rknn_inputs_set`、`rknn_run`、`rknn_outputs_get`、输出拷贝、
  NHWC→NCHW 转置、ORT、后处理。

## Baseline

| 阶段 | p50 |
|---|---|
| 预处理（384×384 + ImageNet 归一化） | 3.8 ms |
| `rknn_inputs_set`（float32 → FP16） | 4.1 ms |
| `rknn_run`（DINOv2 backbone） | 188.4 ms |
| `rknn_outputs_get` | 1.4 ms |
| 输出 memcpy | 0.2 ms |
| NHWC→NCHW 转置 | 0.1 ms |
| ORT Transformer + head（4 线程） | 39.7 ms |
| 后处理 | 0.3 ms |
| **合计** | **242.1 ms** |

backbone 占 78%，head 占 16%。I/O 一共 5.8ms，能省但改变不了量级。

## 零拷贝

做法和自动剪辑项目一致：

- 输入 `rknn_create_mem` + `rknn_set_io_mem`，FP16 NHWC，`pass_through=1`；
  ImageNet 归一化和 `/255` 留在 CPU，写进输入内存后再 `rknn_mem_sync(TO_DEVICE)`。
- 唯一输出（256×24×24 FP16 NCHW）也绑 io_mem，读之前 `sync(FROM_DEVICE)`，转 float 后直接给 ORT。
- 失败时自动回退到 `rknn_inputs_set` + `rknn_outputs_get`，用 `--no-zero-copy` 可以强制走旧路径。

| 阶段 | 原版 | 零拷贝 |
|---|---|---|
| 输入 | 4.1 ms | **0.2 ms** |
| `rknn_run` | 188.4 ms | 188.6 ms |
| 输出 | 1.4 + 0.2 ms | **0.4 ms** |
| ORT | 39.7 ms | 38.7 ms |
| **p50 / p95** | **242.1 / 333.2 ms** | **235.0 / 314.4 ms** |

检测结果不变（5 个目标），p95 降了 5.6%。I/O 从 5.8ms 降到 0.7ms 以后，瓶颈很明确：
**NPU 上的 backbone 占 188ms**，继续抠 CPU 拷贝没有意义。

## 数值对齐（ONNX vs RKNN）

`scripts/check_backbone_equiv.py` 在板端用同一张图分别跑 ONNX Runtime（CPU）和 RKNN（NPU）：

```
shape=(1,256,24,24)
max_abs=1.039397  mean_abs=0.003060
cosine=0.999821   rel_l2=0.018804
```

max_abs 有 1.04 的离群点，但平均绝对误差 0.003、余弦 0.9998、相对 L2 1.9%，
说明 FP16 转换没有把 backbone 打崩。这个检查只能说明"特征没变"，不能替代检测精度评估。

## 完整流水线

零拷贝后在 `demo_scene.mp4` 上：CV 单帧 p50 263ms，backbone 205ms，head 43ms，
CV 有效 3.7–3.8 fps；LLM 首 token 1.5s 左右，端到端 5.5s 左右。
LLM 的波动比 CV 大（prompt 长度、KV cache、和 CV 抢核都会影响），报告里保留三次原始值。
完整流水线的 263ms 和之前 279ms 的差异里混着运行波动，零拷贝的真实收益看 bench 的 242→235ms。

## 没做 / 下一步

- **backbone INT8**：DINOv2 对量化更敏感，而且要先解决 ImageNet 归一化和校准分布的问题；
  本轮只做了 FP16 数值对齐。
- **NPU/CPU 流水线重叠**：head 跑在 CPU 时 NPU 是空闲的，理论上限约 `max(188, 39)`，
  可以把 CV 吞吐再提 ~15%，但会和 LLM 抢 CPU，需要一起测 TTFT。
- **ORT 线程数/自旋**：目前 4 线程、默认自旋；LLM 并行时是否该降到 2 线程，还没做 A/B。
- **多路视频**：单路都还没跑满，多路只会更早撞到 NPU 瓶颈。

## 复现

```bash
# 板端：A/B 零拷贝
./build/mm_edge --bench 200 --image ../data/sample.jpg \
  --backbone ../models/backbone.rknn --head ../models/detection.onnx \
  --core-mask 1 --no-zero-copy --bench-json ../assets/bench_baseline.json
./build/mm_edge --bench 200 --image ../data/sample.jpg \
  --backbone ../models/backbone.rknn --head ../models/detection.onnx \
  --core-mask 1 --bench-json ../assets/bench_zerocopy.json

# 数值对齐
python scripts/check_backbone_equiv.py --rknn models/backbone.rknn \
  --onnx models/backbone.onnx --image data/sample.jpg
```

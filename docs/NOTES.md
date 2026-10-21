# 实现笔记

## 数据流

```
视频帧
 └─ CV 线程（绑 A55 0-3）
      ├─ 预处理：letterbox 到 384×384，NHWC
      ├─ RKNN：DINOv2 backbone，单尺度 256×24×24
      ├─ 输出 NHWC → NCHW，喂给 ONNX Runtime
      ├─ ORT：Transformer + 检测头（INT8 动态量化），4 线程
      ├─ 后处理：sigmoid + top-k，无 NMS
      └─ 规则判断 → Observation
            │
            │ 单槽位（只保留最新观测）
            ▼
      LLM 线程（独占 A76 4-7）
         build_prompt → RKLLM Qwen2.5-1.5B → 流式回调 → 一句中文预警
```

## 为什么拆成两段

RF-DETR 是 Transformer 检测器。整模型转 RKNN 时，注意力里的 softmax/exp 在 NPU 上效率不高，精度也掉；但 DINOv2 backbone 是纯卷积，NPU 很合适。

所以只把 backbone 转 RKNN，Transformer 和检测头留在 CPU 上跑 ONNX Runtime。中间特征图做一次 NHWC→NCHW 转置，换来 222ms + 42ms 的单帧延迟，比全 CPU 快一个量级。

Nano 只有 1 个尺度（256×24×24），写 wrapper 时不能假设有 3 个尺度。

## 调度

一开始 CV 和 LLM 抢 CPU 大核，LLM 首 token 从 0.3s 涨到 8.8s。定位过程：

1. 先把 RKNN 限制到单核，TTFT 只从 8.8s 降到 7.5s，没用；
2. 再把 CV 线程绑到 A55 小核（`sched_setaffinity`），TTFT 降到 1.2s，有效；
3. 结论是瓶颈在 CPU 大核，不在 NPU 算力——rkllm 默认用 CPU4-7，CV 把大核占满后，LLM 的采样和 embedding 被饿死。

现在 CV 绑 `0,1,2,3`，RKNN 用 NPU core0，LLM 独占 A76 和其余 NPU 核。

## Prompt

Prompt 长度直接决定 prefill 时间。Python 版做过对比：

- 6 条完整检测描述 + few-shot ≈ 200 token → TTFT 3.9s
- 压成 4 条摘要、去掉 few-shot ≈ 90 token → TTFT 1.06s

C++ 版目前是 4 条检测 + 3 条规则，首 token 1.2～1.5s。Prompt 里只放结构化事实，不让 LLM 自己算"谁越界"——1.5B 做不了几何推理，容易编。

## 规则

- 危险区：归一化坐标 `[0.2, 0.55] - [0.8, 1.0]`，人体框中心落在里面 → `danger`
- 人车过近：人框和车框 IoU > 0.05 → `warning`
- 人员聚集：画面里人 ≥ 4 → `warning`
- 以上都没有 → `normal`

规则层输出确定性的 `rules` 和 `risk_level`，LLM 只负责措辞。

## 背压

LLM 一次生成 4s 左右，期间 CV 能产出十几帧。排队没有意义，预警会永远慢半拍。

所以槽位只保留最新一条 Observation，新的直接覆盖旧的。`observations_dropped=42` 就是背压生效的证据；`frames_while_llm_busy=44` 表示 LLM 忙的时候 CV 仍在跑。

## 零拷贝与数值对齐

- 输入按 FP16 绑 `rknn_set_io_mem`，`pass_through=1`；ImageNet 归一化和 `/255` 在 CPU 做完再写进去，
  避免 runtime 把 normalize 插进 NPU。
- 输出（256×24×24 FP16 NCHW）也绑 io_mem，`rknn_mem_sync(FROM_DEVICE)` 后转 float 给 ORT。
- 失败会自动回退到 `rknn_inputs_set` + `rknn_outputs_get`；`--no-zero-copy` 强制走旧路径。
- 单帧 I/O 从 5.7ms 降到 0.7ms，p95 从 333ms 降到 314ms；backbone 的 188ms 没变。
- FP16 backbone 和 ONNX 参考对比：余弦 0.9998、相对 L2 1.9%、平均绝对误差 0.003。
  脚本是 `scripts/check_backbone_equiv.py`，只能说明特征没被转换打崩，不能替代检测精度评估。

## 测量

- LLM 启动时先跑一次"你好"预热，把权重换入和首次内存分配的成本摊掉，否则第一次 TTFT 会被算成 1.8s 以上。
- `--bench N` 跑单图检测，前两次预热，输出各阶段 p50 / p95，以及 RSS、CPU 占用、温度、NPU/CPU 频率。
- 完整流水线记录每帧 pre / backbone / head / post，以及每次 LLM 的 TTFT、总时长、观测从采集到出字的 e2e。
- 分位数是最近秩，和历史 report 口径一致。LLM 只有 3 次调用，p95 字段没有统计意义，看 p50 和三次原始值更靠谱。

## 没做 / 做得很糙

- 检测头只有 ORT 动态量化，没有 QAT；backbone FP16，INT8 还没跑。
- LLM 是 w8a8；`docs/quantize_w4a16.md` 里有 w4a16 的转换流程，但没有在板上跑过。
- 只处理一路视频，不接 RTSP/摄像头，也不做多轮对话。
- NPU 跑 backbone 和 CPU 跑 head 还是串行的，没做两级流水；做了之后 CV 吞吐理论上还能再提 ~15%。
- 演示素材是合成的，模型在真实监控画面上的泛化没测。
- 后处理没有 NMS，靠 RF-DETR 本身的无 NMS 设计；如果换模型需要补。

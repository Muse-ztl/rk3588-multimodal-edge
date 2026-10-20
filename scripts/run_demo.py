"""端侧多模态推理 Demo：视觉感知 -> 结构化观测 -> 端侧 LLM 语义预警。

板端运行示例：
    LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH \
    python scripts/run_demo.py \
        --video data/demo_scene.mp4 \
        --backbone models/backbone.rknn \
        --head models/detection.onnx \
        --llm /home/orangepi/rkllm_work/model/Qwen2.5-1.5B.rkllm \
        --lib build/librkllm_bridge.so \
        --out output --duration 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mm_edge.llm_bridge import RKLLMEngine  # noqa: E402
from mm_edge.rfdetr_split import SplitRFDETR  # noqa: E402
from mm_edge.scheduler import Pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/demo_scene.mp4")
    ap.add_argument("--backbone", default="models/backbone.rknn")
    ap.add_argument("--head", default="models/detection.onnx")
    ap.add_argument("--llm", default="")
    ap.add_argument("--lib", default="build/librkllm_bridge.so")
    ap.add_argument("--out", default="output")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = whole video")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--core-mask", type=int, default=0,
                    help="RKNN NPU core mask: 0=auto, 1=core0, 3=core0+1, 7=all")
    ap.add_argument("--cv-cores", default="4,5,6,7", help="CPU affinity for the CV thread")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    detector = SplitRFDETR(
        args.backbone, args.head, threshold=args.threshold,
        npu_core_mask=args.core_mask or None,
    )
    print(f"[demo] detector ready, input={detector.input_hw}")

    engine = None
    if args.llm:
        engine = RKLLMEngine(args.lib, args.llm, max_new_tokens=args.max_new_tokens)
        # 首次推理要付 NPU 权重换入/图初始化的钱（约 1.5~2s），
        # 先跑一次空转把它摊掉，否则统计里的 TTFT 会把冷启动算进交互延迟。
        t0 = time.perf_counter()
        engine.run("你好")
        print(f"[demo] llm ready: {args.llm} (warmup {time.perf_counter() - t0:.1f}s)")

    pipe = Pipeline(
        detector, engine, args.video, out_dir=args.out,
        cv_stride=args.stride, loop_video=not args.no_loop,
        cv_cores=[int(c) for c in args.cv_cores.split(",") if c.strip()],
    )
    report = pipe.run(duration_s=args.duration or None)
    print("\n=== report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    detector.release()
    if engine:
        engine.close()


if __name__ == "__main__":
    main()

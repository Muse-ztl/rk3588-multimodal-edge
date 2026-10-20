"""分阶段性能基准：把端到端延迟拆到每一段，便于定位瓶颈。

    python scripts/bench_stages.py --backbone ... --head ... --llm ... --lib ... --runs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mm_edge.llm_bridge import RKLLMEngine  # noqa: E402
from mm_edge.prompt import build_prompt  # noqa: E402
from mm_edge.rfdetr_split import SplitRFDETR  # noqa: E402
from mm_edge.zones import Observation, analyze  # noqa: E402


def stats(vals):
    a = np.array(vals, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/sample.jpg")
    ap.add_argument("--backbone", default="models/backbone.rknn")
    ap.add_argument("--head", default="models/detection.onnx")
    ap.add_argument("--llm", default="")
    ap.add_argument("--lib", default="build/librkllm_bridge.so")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="output/bench_stages.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")

    det = SplitRFDETR(args.backbone, args.head)
    pre, npu, head, post, total = [], [], [], [], []

    for i in range(args.warmup + args.runs):
        d = det.infer(img)
        if i < args.warmup:
            continue
        t = det.last_timing
        pre.append(t["preprocess_ms"])
        npu.append(t["backbone_npu_ms"])
        head.append(t["head_cpu_ms"])
        post.append(t["postprocess_ms"])
        total.append(t["total_ms"])
        dets = d
        loc, rules, level = analyze(dets, img.shape[1], img.shape[0])

    result = {
        "detector": {
            "preprocess_ms": stats(pre),
            "backbone_npu_ms": stats(npu),
            "head_cpu_ms": stats(head),
            "postprocess_ms": stats(post),
            "total_ms": stats(total),
            "num_detections": len(dets),
        }
    }
    print("[bench] detector:", json.dumps(result["detector"]["total_ms"]))
    print(f"[bench] npu={result['detector']['backbone_npu_ms']['p50']:.1f}ms "
          f"head={result['detector']['head_cpu_ms']['p50']:.1f}ms "
          f"total={result['detector']['total_ms']['p50']:.1f}ms")

    if args.llm:
        obs = Observation(frame_id=1, ts=time.time(), width=img.shape[1],
                          height=img.shape[0], dets=dets, location_text=loc,
                          rules=rules or ["无"], risk_level=level)
        prompt = build_prompt(obs)
        eng = RKLLMEngine(args.lib, args.llm, max_new_tokens=48)
        ttfts, totals = [], []
        for i in range(3):
            r = eng.run(prompt)
            ttfts.append(r.ttft_ms)
            totals.append(r.total_ms)
            print(f"[bench] llm sample {i}: ttft={r.ttft_ms:.0f}ms total={r.total_ms:.0f}ms | {r.text[:40]}")
        result["llm"] = {"ttft_ms": stats(ttfts), "total_ms": stats(totals)}
        eng.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[bench] wrote {args.out}")
    det.release()


if __name__ == "__main__":
    main()

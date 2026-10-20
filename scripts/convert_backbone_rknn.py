"""Convert the RF-DETR backbone ONNX to RKNN (x86 host, rknn-toolkit2 2.3.2).

    python scripts/convert_backbone_rknn.py \
        --input output/backbone.onnx --output output/backbone.rknn \
        --calib-dir data/calib --quantize

注意：
  * 输入已是 [0,1] 归一化的 float32 NCHW，这里 mean=0 / std=1 避免重复归一化。
  * 默认不做量化（FP16）。INT8 需要 --quantize + 一批校准图（dataset.txt），
    在 RK3588 上大约还能再省 30~40% 带宽，但要注意 transformer 前段对量化敏感。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/backbone.onnx")
    ap.add_argument("--output", default="output/backbone.rknn")
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--dataset", default="", help="dataset.txt for INT8 calibration")
    args = ap.parse_args()

    from rknn.api import RKNN

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    if not src.exists():
        raise SystemExit(f"missing {src}")

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            target_platform=args.target,
        )
        print(f"[convert] config ret={ret}")
        ret = rknn.load_onnx(model=str(src))
        print(f"[convert] load ret={ret}")
        kwargs = {"do_quantization": bool(args.quantize)}
        if args.quantize:
            if not args.dataset or not os.path.exists(args.dataset):
                raise SystemExit("--quantize requires an existing --dataset")
            kwargs["dataset"] = args.dataset
        ret = rknn.build(**kwargs)
        print(f"[convert] build ret={ret}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(dst))
        print(f"[convert] export ret={ret} -> {dst}")
    finally:
        rknn.release()

    # 顺手做一次模拟器精度对比，确认转换没有把 backbone 打崩
    try:
        import numpy as np
        import onnxruntime as ort

        rknn2 = RKNN(verbose=False)
        rknn2.load_rknn(str(dst))
        rknn2.init_runtime()
        x = np.random.rand(1, 3, 384, 384).astype(np.float32)
        outs = rknn2.inference(inputs=[x], data_format=["nchw"])
        sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        ref = sess.run(None, {sess.get_inputs()[0].name: x})
        for i, (a, b) in enumerate(zip(outs, ref)):
            a = np.asarray(a)
            if a.shape != b.shape and a.ndim == 4 and b.ndim == 4:
                a = a.transpose(0, 3, 1, 2)
            diff = float(np.abs(a.reshape(b.shape) - b).max())
            print(f"[verify] feat{i} shape={b.shape} max_abs_diff={diff:.5f}")
        rknn2.release()
    except Exception as exc:
        print(f"[verify] skipped: {exc}")


if __name__ == "__main__":
    main()

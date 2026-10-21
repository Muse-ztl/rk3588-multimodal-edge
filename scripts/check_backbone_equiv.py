"""Compare the RKNN backbone against the ONNX reference on the board.

    python scripts/check_backbone_equiv.py \
        --rknn models/backbone.rknn --onnx models/backbone.onnx --image data/sample.jpg

Both run on the same machine: ONNX Runtime on CPU, RKNN on NPU. This is a
conversion sanity check (FP16), not a quantization accuracy report: it answers
"did the NPU conversion change the features", not "is the detector still good".
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    nhwc = rgb[None, ...]
    nchw = nhwc.transpose(0, 3, 1, 2)
    return nhwc, nchw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default="models/backbone.rknn")
    ap.add_argument("--onnx", default="models/backbone.onnx")
    ap.add_argument("--image", default="data/sample.jpg")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    nhwc, nchw = preprocess(args.image, args.size)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: nchw})

    rk = RKNNLite()
    if rk.load_rknn(args.rknn) != 0:
        raise SystemExit("load_rknn failed")
    if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        raise SystemExit("init_runtime failed")
    outs = rk.inference(inputs=[nhwc], data_format=["nhwc"])
    rk.release()

    a = np.asarray(ref[0], dtype=np.float32)
    b = np.asarray(outs[0], dtype=np.float32)
    if a.shape != b.shape:
        if a.ndim == 4 and b.ndim == 4 and b.shape[1] == a.shape[3]:
            b = b.transpose(0, 3, 1, 2)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: onnx {a.shape} vs rknn {b.shape}")

    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1))) + 1e-12
    cosine = float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)
    rel = float(np.linalg.norm(diff.reshape(-1)) / (np.linalg.norm(a.reshape(-1)) + 1e-12))
    print(f"[equiv] shape={a.shape} max_abs={diff.max():.6f} mean_abs={diff.mean():.6f}")
    print(f"[equiv] cosine={cosine:.6f} rel_l2={rel:.6f}")


if __name__ == "__main__":
    main()

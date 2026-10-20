"""RF-DETR split inference: backbone on RK3588 NPU, transformer head on CPU (ONNX Runtime).

RF-DETR's ONNX graph is transformer-heavy; end-to-end RKNN conversion loses both
accuracy and speed. This module implements the split deployment:

    frame -> preprocess -> [RKNN: DINOv2 backbone] -> 3 feature maps
          -> layout adapter -> [ORT: transformer + detector head] -> boxes

The layout adapter is required because RKNNLite returns feature maps in the
layout of the RKNN runtime (NHWC for 4-D tensors), while the exported head
expects NCHW. Converting by shape inference keeps the code version-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .coco import class_name


@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    label: str
    class_id: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_nchw(out: np.ndarray, expected: Sequence) -> np.ndarray:
    """Align an RKNN output to the head's expected NCHW shape."""
    exp = [d for d in expected if isinstance(d, int) and d > 0]
    if len(exp) == out.ndim and list(out.shape[: len(exp)]) == exp:
        return out
    if out.ndim == 4 and out.shape[0] == 1:
        cand = out.transpose(0, 3, 1, 2)
        if len(exp) == 4 and list(cand.shape) == exp:
            return np.ascontiguousarray(cand)
        # 3-D expected shapes (some exports drop the batch dim)
        if len(exp) == 3 and list(cand.shape[1:]) == exp:
            return np.ascontiguousarray(cand[0])
    if out.ndim == 4 and len(exp) == 3 and list(out.shape[1:]) == exp:
        return out[0]
    return out


class SplitRFDETR:
    """RF-DETR nano/small deployment with a split RKNN + ORT pipeline."""

    def __init__(
        self,
        backbone_rknn: str,
        detector_onnx: str,
        threshold: float = 0.45,
        num_select: int = 300,
        ort_threads: int = 4,
        npu_core_mask: int | None = None,
        input_size: int = 0,
    ) -> None:
        self.threshold = threshold
        self.num_select = num_select
        self._npu_core_mask = npu_core_mask

        from rknnlite.api import RKNNLite  # imported lazily so x86 tooling can reuse this file

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(backbone_rknn)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn({backbone_rknn}) failed: {ret}")
        kwargs = {}
        if npu_core_mask is not None:
            kwargs["core_mask"] = npu_core_mask
        ret = self.rknn.init_runtime(**kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed: {ret}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ort_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.head = ort.InferenceSession(
            detector_onnx, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._head_inputs = self.head.get_inputs()
        self._head_shapes = [i.shape for i in self._head_inputs]

        # head 的输入是特征图（如 1×256×24×24），不能拿来当图像分辨率。
        # 优先读导出时写入的 meta.json，保证 x86/板端一致。
        meta = {}
        meta_path = os.path.join(os.path.dirname(os.path.abspath(detector_onnx)), "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        if input_size:
            self.input_hw = (input_size, input_size)
        elif meta.get("resolution"):
            res = int(meta["resolution"])
            self.input_hw = (res, res)
        else:
            self.input_hw = (384, 384)
        if num_select == 300 and meta.get("num_select"):
            self.num_select = int(meta["num_select"])

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ utils
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def _backbone(self, tensor: np.ndarray) -> List[np.ndarray]:
        outs = self.rknn.inference(inputs=[tensor], data_format=["nchw"])
        if outs is None:
            raise RuntimeError("RKNNLite.inference returned None")
        aligned = []
        for i, out in enumerate(outs):
            if i < len(self._head_shapes):
                aligned.append(_to_nchw(np.asarray(out), self._head_shapes[i]))
            else:
                aligned.append(np.asarray(out))
        return aligned

    def _head(self, feats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feed = {}
        for i, inp in enumerate(self._head_inputs):
            feed[inp.name] = feats[i]
        outs = self.head.run(None, feed)
        return outs[0], outs[1]

    def postprocess(
        self, dets: np.ndarray, logits: np.ndarray, orig_hw: Tuple[int, int]
    ) -> List[Detection]:
        """NMS-free decode: sigmoid + top-k over (query, class) then scale to pixels."""
        h, w = orig_hw
        prob = _sigmoid(logits)[0]                    # (Q, C)
        flat = prob.reshape(-1)
        k = min(self.num_select, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(-flat[idx])]

        num_classes = prob.shape[1]
        q_idx = idx // num_classes
        c_idx = idx % num_classes
        scores = flat[idx]

        keep = scores > self.threshold
        q_idx, c_idx, scores = q_idx[keep], c_idx[keep], scores[keep]
        if scores.size == 0:
            return []

        boxes = dets[0][q_idx]                        # cx, cy, w, h in [0,1]
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - bw / 2) * w, 0, w)
        y1 = np.clip((cy - bh / 2) * h, 0, h)
        x2 = np.clip((cx + bw / 2) * w, 0, w)
        y2 = np.clip((cy + bh / 2) * h, 0, h)

        # RF-DETR is NMS-free; a cheap class-wise NMS removes the rare duplicate
        # produced by quantising the head (keeps downstream prompting clean).
        xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep_ids = cv2.dnn.NMSBoxes(
            xywh, scores.tolist(), self.threshold, 0.65
        )
        if len(keep_ids) > 0:
            keep_ids = np.asarray(keep_ids).reshape(-1)
        else:
            keep_ids = np.arange(len(xywh))

        out: List[Detection] = []
        for i in keep_ids:
            out.append(
                Detection(
                    box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                    score=float(scores[i]),
                    label=class_name(int(c_idx[i])),
                    class_id=int(c_idx[i]),
                )
            )
        return out

    # ------------------------------------------------------------------- api
    def infer(self, bgr: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        tensor = self.preprocess(bgr)
        t1 = time.perf_counter()
        feats = self._backbone(tensor)
        t2 = time.perf_counter()
        dets, logits = self._head(feats)
        t3 = time.perf_counter()
        result = self.postprocess(dets, logits, (bgr.shape[0], bgr.shape[1]))
        t4 = time.perf_counter()
        self.last_timing = {
            "preprocess_ms": (t1 - t0) * 1e3,
            "backbone_npu_ms": (t2 - t1) * 1e3,
            "head_cpu_ms": (t3 - t2) * 1e3,
            "postprocess_ms": (t4 - t3) * 1e3,
            "total_ms": (t4 - t0) * 1e3,
        }
        return result

    def release(self) -> None:
        try:
            self.rknn.release()
        except Exception:
            pass

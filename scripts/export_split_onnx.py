"""Export RF-DETR as split ONNX: backbone + transformer/detector head.

Run on x86 (WSL/Ubuntu), not on the board:

    python scripts/export_split_onnx.py \
        --weights ~/rfdetr_ckpt/checkpoint_best_regular.pth \
        --out output

Outputs (in --out):
    backbone.onnx            DINOv2 backbone, NCHW input -> 3 feature maps
    detection_fp32.onnx      transformer + class/box heads
    detection.onnx           dynamic-quantized (INT8 weight) head, for CPU
    meta.json                resolution / num_select / tensor shapes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class BackboneWrapper(torch.nn.Module):
    """导出专用：lwdetr.export() 后 Joiner.forward 变为 forward_export(plain tensor)。"""

    def __init__(self, lwdetr: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = lwdetr.backbone

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        # forward_export 返回 (feats, masks, poss, cross_attn)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        return tuple(feats)


class DetectionWrapper(torch.nn.Module):
    """Transformer decoder + class/box heads，位置编码作为常量烘焙进图。"""

    def __init__(self, lwdetr, poss, image_hw):
        super().__init__()
        self.transformer = lwdetr.transformer
        self.refpoint_embed = lwdetr.refpoint_embed
        self.query_feat = lwdetr.query_feat
        self.class_embed = lwdetr.class_embed
        self.bbox_embed = lwdetr.bbox_embed
        self.bbox_reparam = bool(getattr(lwdetr, "bbox_reparam", False))
        self.num_queries = int(lwdetr.num_queries)
        self.num_pos = len(poss)
        self.image_h = int(image_hw[0])
        self.image_w = int(image_hw[1])
        for i, p in enumerate(poss):
            self.register_buffer(f"pos_{i}", p.detach().clone(), persistent=True)

    def forward(self, *srcs):
        if len(srcs) != self.num_pos:
            raise ValueError(f"expected {self.num_pos} feature maps, got {len(srcs)}")
        poss = [getattr(self, f"pos_{i}") for i in range(self.num_pos)]
        srcs_list = list(srcs)
        ref_w = self.refpoint_embed.weight[: self.num_queries]
        qfeat_w = self.query_feat.weight[: self.num_queries]

        outs = self.transformer(srcs_list, None, poss, ref_w, qfeat_w)
        hs, ref_unsigmoid = outs[0], outs[1]

        if self.bbox_reparam:
            delta = self.bbox_embed(hs)
            cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            boxes = torch.concat([cxcy, wh], dim=-1)
        else:
            boxes = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        logits = self.class_embed(hs)
        return boxes, logits


def resolve_lwdetr(model):
    """兼容不同 rfdetr 版本的嵌套层级，拿到真正的 LWDETR nn.Module。"""
    node = model
    for _ in range(4):
        nxt = getattr(node, "model", None)
        if nxt is None:
            break
        node = nxt
        if hasattr(node, "backbone") and hasattr(node, "transformer"):
            return node
    raise RuntimeError("cannot locate LWDETR module on RF-DETR instance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RFDETRNano")
    ap.add_argument("--weights", required=True, help="local RF-DETR checkpoint (.pth)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quant", action="store_true", help="keep head in fp32")
    ap.add_argument("--skip-full", action="store_true", help="skip the official full ONNX export")
    args = ap.parse_args()

    import rfdetr

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctor = getattr(rfdetr, args.model)
    try:
        # 官方 checkpoint 里带了非 tensor 对象，新版本 torch 默认 weights_only=True
        # 会拒绝加载，必须显式信任本地权重文件。
        model = ctor(pretrain_weights=args.weights, trust_checkpoint=True)
    except TypeError:
        model = ctor(pretrain_weights=args.weights)
    lwdetr = resolve_lwdetr(model).cpu().eval()

    resolution = int(getattr(getattr(model, "model", model), "resolution", 0) or 0)
    if resolution <= 0:
        resolution = int(getattr(lwdetr, "resolution", 384))
    print(f"[export] model={args.model} resolution={resolution}")

    if not args.skip_full:
        try:
            model.export(output_dir=str(out_dir))
            print("[export] full ONNX (reference) done")
        except Exception as exc:
            print(f"[export] full ONNX export skipped: {exc}")

    dummy = torch.randn(1, 3, resolution, resolution, dtype=torch.float32)
    lwdetr.export()  # Joiner 切换到 forward_export：接受纯 Tensor，返回 (feats, masks, poss, cross)
    with torch.no_grad():
        out = lwdetr.backbone(dummy)
        srcs = out[0]
        poss = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
        if poss is None:
            raise RuntimeError("backbone did not return positional embeddings; unsupported version")
        print("[export] backbone outputs:", [tuple(s.shape) for s in srcs])
        print("[export] pos embeddings  :", [tuple(p.shape) for p in poss])

    backbone_path = out_dir / "backbone.onnx"
    head_fp32_path = out_dir / "detection_fp32.onnx"
    head_path = out_dir / "detection.onnx"

    torch.onnx.export(
        BackboneWrapper(lwdetr).eval(), dummy, str(backbone_path),
        opset_version=args.opset, dynamo=False,
        input_names=["input"], output_names=[f"feat{i}" for i in range(len(srcs))],
    )
    torch.onnx.export(
        DetectionWrapper(lwdetr, poss, (resolution, resolution)).eval(), tuple(srcs),
        str(head_fp32_path), opset_version=args.opset, dynamo=False,
        input_names=[f"feat{i}" for i in range(len(srcs))],
        output_names=["boxes", "logits"],
    )

    if args.no_quant:
        head_path.write_bytes(head_fp32_path.read_bytes())
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prep = out_dir / "detection_prep.onnx"
        quant_pre_process(str(head_fp32_path), str(prep))
        quantize_dynamic(str(prep), str(head_path), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        prep.unlink(missing_ok=True)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    meta = {
        "model": args.model,
        "resolution": resolution,
        "num_select": int(getattr(lwdetr, "num_queries", 300)),
        "num_queries": int(getattr(lwdetr, "num_queries", 300)),
        "backbone_outputs": [tuple(int(d) for d in s.shape) for s in srcs],
        "head_inputs": [i.name for i in sess.get_inputs()],
        "head_input_shapes": [list(i.shape) for i in sess.get_inputs()],
        "head_outputs": [o.name for o in sess.get_outputs()],
        "head_output_shapes": [list(o.shape) for o in sess.get_outputs()],
        "head_quantized": not args.no_quant,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[export] meta:", json.dumps(meta, indent=2))
    print(f"[export] files: {[p.name for p in out_dir.glob('*.onnx')]}")


if __name__ == "__main__":
    main()

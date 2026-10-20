"""给 C++ 产出的标注帧加上中文 LLM 输出（仅用于生成 README 配图）。

推理链路在 cpp/ 里跑，OpenCV 自带字体不支持中文，所以文档配图这一步用 PIL
补一行中文。**不参与推理**，也不进入板上运行路径。

    python tools/render_zh.py --frame frame.jpg --trace llm_traces.jsonl --out demo.jpg
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bottom", type=int, default=44)
    args = ap.parse_args()

    last = None
    with open(args.trace, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise SystemExit("empty trace file")

    img = Image.open(args.frame).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + args.bottom), (24, 24, 24))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    text = f"LLM（端侧 Qwen2.5-1.5B）: {last['text']}"
    draw.text((10, h + 12), text, font=font, fill=(240, 240, 240))
    canvas.save(args.out, quality=92)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""生成多模态演示素材：把静图做成缓慢推近的"监控画面"。

    python scripts/make_demo_clip.py --image data/sample.jpg --out data/demo_scene.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/sample.jpg")
    ap.add_argument("--out", default="data/demo_scene.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", default="960x540")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # zoompan 做缓慢推近，模拟云台/变焦摄像头
    vf = (
        f"scale={w * 2}:{h * 2},"
        f"zoompan=z='min(zoom+0.0008,1.25)':d={int(args.seconds * args.fps)}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={args.fps},"
        f"format=yuv420p"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-loop", "1", "-t", f"{args.seconds}", "-i", args.image,
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
           "-t", f"{args.seconds}", args.out]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode("utf-8", "ignore")[-500:])
    print(f"wrote {args.out} ({args.seconds}s, {w}x{h}@{args.fps})")


if __name__ == "__main__":
    main()

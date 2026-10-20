"""帧标注：检测框 + 危险区 + LLM 语义输出。用于产物留档与 README 配图。

注意：cv2.putText 只支持 ASCII，板端要显示中文必须走 PIL + CJK 字体，
否则输出全是方框。这里做了字体探测 + ASCII 降级。
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from .zones import DEFAULT_DANGER_ZONE, Observation

_GREEN = (80, 200, 80)
_RED = (60, 60, 230)
_AMBER = (40, 170, 240)
_ZONE = (0, 200, 255)

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _find_cjk_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    hits = glob.glob("/usr/share/fonts/**/*CJK*", recursive=True) + glob.glob(
        "/usr/share/fonts/**/*wqy*", recursive=True
    )
    return hits[0] if hits else None


_FONT_PATH = _find_cjk_font()
_pil_font = None
if _FONT_PATH:
    try:
        from PIL import ImageFont

        _pil_font = ImageFont.truetype(_FONT_PATH, 18)
    except Exception:
        _pil_font = None


def _put_text(img: np.ndarray, text: str, xy, color, scale=0.52) -> np.ndarray:
    """有 CJK 字体时用 PIL 画，否则退回 cv2（只能画 ASCII）。"""
    if _pil_font is not None and any(ord(c) > 127 for c in text):
        from PIL import Image, ImageDraw

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text(xy, text, font=_pil_font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    safe = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(img, safe, (xy[0], xy[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)
    return img


def draw(frame: np.ndarray, obs: Observation, llm_text: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    zx1, zy1, zx2, zy2 = DEFAULT_DANGER_ZONE
    overlay = img.copy()
    cv2.rectangle(overlay, (int(zx1 * w), int(zy1 * h)), (int(zx2 * w), int(zy2 * h)),
                  _ZONE, -1)
    img = cv2.addWeighted(overlay, 0.10, img, 0.90, 0)
    cv2.rectangle(img, (int(zx1 * w), int(zy1 * h)), (int(zx2 * w), int(zy2 * h)),
                  _ZONE, 2)
    cv2.putText(img, "DANGER ZONE", (int(zx1 * w) + 6, int(zy1 * h) + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _ZONE, 2, cv2.LINE_AA)

    color = _RED if obs.risk_level == "danger" else (
        _AMBER if obs.risk_level == "warning" else _GREEN)
    for d in obs.dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.label} {d.score:.2f}"
        cv2.putText(img, label, (x1 + 2, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    bar_h = 60
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, bar_h), (20, 20, 20), -1)
    img = cv2.addWeighted(bar, 0.65, img, 0.35, 0)

    head = f"frame {obs.frame_id} | risk={obs.risk_level}"
    if obs.rules:
        head += " | " + "/".join(obs.rules)
    img = _put_text(img, head, (8, 4), color)
    if llm_text:
        img = _put_text(img, llm_text[:40], (8, 30), (245, 245, 245))
    return img

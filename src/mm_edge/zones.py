"""Spatial rules: turn raw boxes into语义化位置 + 风险命中.

小模型不适合做几何推理，所以"是否越界/是否过近"这类判定放在规则层，
LLM 只负责把这些结构化事实讲成人话，职责边界清晰、也更容易复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .rfdetr_split import Detection

# 归一化危险区（画面下方中央），对应"设备前方作业区"
DEFAULT_DANGER_ZONE = (0.20, 0.55, 0.80, 1.0)


@dataclass
class Observation:
    frame_id: int
    ts: float
    width: int
    height: int
    dets: List[Detection]
    location_text: str
    rules: List[str] = field(default_factory=list)
    risk_level: str = "normal"


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def position_text(box: Tuple[float, float, float, float], w: int, h: int) -> str:
    cx, cy = _center(box)
    nx, ny = cx / max(w, 1), cy / max(h, 1)
    lr = "左侧" if nx < 1 / 3 else ("中央" if nx < 2 / 3 else "右侧")
    # 用框底边判断远近，比用中心点更稳（人站在近处时中心仍然偏上）
    bottom = box[3] / max(h, 1)
    depth = "近处" if bottom > 0.75 else ("中景" if bottom > 0.45 else "远处")
    return f"{lr}{depth}"


def _in_zone(box: Tuple[float, float, float, float], w: int, h: int, zone) -> bool:
    cx, cy = _center(box)
    nx, ny = cx / max(w, 1), cy / max(h, 1)
    return zone[0] <= nx <= zone[2] and zone[1] <= ny <= zone[3]


def _box_iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


PERSON_LABELS = {"person"}
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}


def analyze(dets: Sequence[Detection], w: int, h: int, zone=DEFAULT_DANGER_ZONE) -> Tuple[str, List[str], str]:
    """返回 (画面描述, 命中规则, 风险等级)。"""
    persons = [d for d in dets if d.label in PERSON_LABELS]
    vehicles = [d for d in dets if d.label in VEHICLE_LABELS]

    parts = []
    for d in dets[:8]:
        parts.append(
            f"{d.label}({d.score:.2f})@{position_text(d.box, w, h)}"
        )
    location_text = "，".join(parts) if parts else "无目标"

    rules: List[str] = []
    level = "normal"

    intruders = [p for p in persons if _in_zone(p.box, w, h, zone)]
    if intruders:
        rules.append("人员进入危险区域")
        level = "danger"

    for p in persons:
        for v in vehicles:
            if _box_iou(p.box, v.box) > 0.05:
                rules.append("人车距离过近")
                level = "danger" if level == "danger" else "warning"
                break

    if len(persons) >= 4:
        rules.append("区域人员聚集")
        level = "danger" if level == "danger" else "warning"

    if not persons and vehicles:
        rules.append("区域内有车辆活动但无人")

    return location_text, rules, level

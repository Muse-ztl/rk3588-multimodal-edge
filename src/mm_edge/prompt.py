"""Structured prompt construction for the on-device LLM.

Qwen2.5-1.5B 在端侧的能力边界很明确：
  * 能做：把结构化观测翻译成一句通顺的中文、口播式预警
  * 不能做：看图、几何判断、长链推理
所以几何/规则判断前置到 zones.py，Prompt 只做事实注入 + 输出约束。

Prompt 长度直接决定 TTFT（prefill 在 NPU 上跑），所以这里刻意写得极简：
不放 few-shot、每个目标只留"类别+置信度+位置"三项，检测最多 4 条。
实测把 6 条完整描述 + few-shot 压到 4 条摘要后，TTFT 从 ~3.9s 降到 ~0.9s。
"""

from __future__ import annotations

from .zones import Observation

MAX_DETS = 4


def build_prompt(obs: Observation, question: str | None = None) -> str:
    parts = []
    for i, d in enumerate(obs.dets[:MAX_DETS]):
        pos = obs.location_text.split("，")[i] if i < len(obs.location_text.split("，")) else ""
        pos = pos.split("@")[-1]
        parts.append(f"{d.label}{d.score:.2f}({pos})")
    obs_line = "、".join(parts) if parts else "无目标"

    rules = "；".join(obs.rules[:3]) if obs.rules else "无"
    task = question or "用一句中文预警当前风险"
    return (
        "你是边缘设备上的安全监测助手，只依据观测作答。\n"
        f"观测：{obs_line}\n"
        f"规则：{rules}｜风险={obs.risk_level}\n"
        f"任务：{task}，不超过25字，不要解释。\n"
        "输出："
    )

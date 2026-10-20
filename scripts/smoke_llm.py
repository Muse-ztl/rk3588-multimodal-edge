"""Smoke test for the RKLLM bridge (runs on the board).

    python3 scripts/smoke_llm.py <model.rkllm> [lib_path]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mm_edge.llm_bridge import RKLLMEngine  # noqa: E402


def main() -> None:
    model = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "build/librkllm_bridge.so"

    t0 = time.perf_counter()
    engine = RKLLMEngine(lib, model, max_new_tokens=64, max_context_len=2048)
    print(f"[smoke] model loaded in {time.perf_counter() - t0:.1f}s")

    prompts = [
        "你是一个边缘设备上的视觉助手。请用一句话说明你的作用。",
        "[观测] 人(0.91)@中央近处；[规则] 人员进入危险区域（风险等级 danger）\n"
        "要求：只输出一句话，不超过40个汉字。\n输出：",
    ]
    for p in prompts:
        r = engine.run(p)
        print(f"[smoke] ttft={r.ttft_ms:.0f}ms total={r.total_ms:.0f}ms segs={r.segments}")
        print(f"[smoke] text: {r.text}")
    engine.close()


if __name__ == "__main__":
    main()

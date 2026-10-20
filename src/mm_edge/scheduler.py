"""CV / LLM 异构调度：NPU 高频采样 + LLM 异步推理。

设计要点（都是为了在 6 TOPS 的 RK3588S 上让两类任务真正并行）：
  * 观测槽位只有 1 个。LLM 一次生成要几百毫秒，期间 CV 可能跑出十来帧，
    旧观测对"当前场景"已经没有意义，所以主动丢弃而不是排队（背压策略）。
  * CV 线程绑大核（A76 4-7），OS/LLM 侧留在小核，避免 CPU 侧预处理与
    ONNX head 抢占 NPU 喂数据的线程。
  * 统计口径分开记：CV 纯推理耗时、LLM 首 token、LLM 总耗时、端到端
    （观测采集时刻 -> LLM 输出完毕）。
"""

from __future__ import annotations

import json
import os
import queue
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from . import viz, zones
from .llm_bridge import LLMReply, RKLLMEngine
from .prompt import build_prompt
from .rfdetr_split import Detection, SplitRFDETR
from .zones import Observation


def pin_to_big_cores(cores: Optional[List[int]] = None) -> str:
    """把当前线程绑到指定 CPU 核（默认 RK3588 的 4 个 A76 大核）。"""
    try:
        big = set(cores) if cores else {4, 5, 6, 7}
        cur = os.sched_getaffinity(0)
        target = cur & big
        if target:
            os.sched_setaffinity(0, target)
            return f"cpu_affinity={sorted(target)}"
        return "cpu_affinity=unchanged(no big core)"
    except Exception as exc:  # pragma: no cover - 非 Linux 平台
        return f"cpu_affinity=unsupported({exc})"


@dataclass
class ObservationSlot:
    """单槽位 + 条件变量：只保留最新观测。"""

    _obs: Optional[Observation] = None
    _seq: int = 0
    published: int = 0
    consumed: int = 0
    _cv: threading.Condition = field(default_factory=threading.Condition)

    def put(self, obs: Observation) -> None:
        with self._cv:
            self._obs = obs
            self._seq += 1
            self.published += 1
            self._cv.notify()

    def wait_newer(self, last_seq: int, timeout: float = 0.2):
        with self._cv:
            if self._seq <= last_seq:
                self._cv.wait(timeout)
            if self._seq <= last_seq or self._obs is None:
                return None, last_seq
            self.consumed += 1
            return self._obs, self._seq


@dataclass
class LLMTrace:
    frame_id: int
    obs_ts: float
    text: str
    ttft_ms: float
    total_ms: float
    e2e_ms: float
    risk_level: str
    rules: List[str]


class Pipeline:
    def __init__(
        self,
        detector: SplitRFDETR,
        engine: Optional[RKLLMEngine],
        source: str,
        out_dir: str = "output",
        cv_stride: int = 1,
        viz_every: int = 15,
        loop_video: bool = True,
        cv_cores: Optional[List[int]] = None,
    ) -> None:
        self.detector = detector
        self.engine = engine
        self.source = source
        self.out_dir = out_dir
        self.cv_stride = max(1, cv_stride)
        self.viz_every = max(1, viz_every)
        self.loop_video = loop_video
        self.cv_cores = cv_cores

        self.slot = ObservationSlot()
        self.traces: List[LLMTrace] = []
        self._stop = threading.Event()
        self.stats = {
            "frames": 0,
            "infer_frames": 0,
            "cv_ms": [],
            "npu_ms": [],
            "head_ms": [],
            "frames_while_llm_busy": 0,
        }
        self._llm_busy = threading.Event()
        self._affinity_note = ""

        os.makedirs(out_dir, exist_ok=True)
        self._jsonl = open(os.path.join(out_dir, "llm_traces.jsonl"), "w", encoding="utf-8")

    # ------------------------------------------------------------------ cv
    def _cv_loop(self, duration_s: Optional[float]) -> None:
        self._affinity_note = pin_to_big_cores(self.cv_cores)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source: {self.source}")
        t_begin = time.time()
        frame_id = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if not self.loop_video:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_id += 1
            self.stats["frames"] += 1
            if duration_s and time.time() - t_begin > duration_s:
                break
            if (frame_id - 1) % self.cv_stride != 0:
                continue

            if self._llm_busy.is_set():
                self.stats["frames_while_llm_busy"] += 1

            dets = self.detector.infer(frame)
            loc, rules, level = zones.analyze(dets, frame.shape[1], frame.shape[0])
            obs = Observation(
                frame_id=frame_id,
                ts=time.time(),
                width=frame.shape[1],
                height=frame.shape[0],
                dets=dets,
                location_text=loc,
                rules=rules,
                risk_level=level,
            )
            self.slot.put(obs)

            t = self.detector.last_timing
            self.stats["infer_frames"] += 1
            self.stats["cv_ms"].append(t["total_ms"])
            self.stats["npu_ms"].append(t["backbone_npu_ms"])
            self.stats["head_ms"].append(t["head_cpu_ms"])

            if self.stats["infer_frames"] % self.viz_every == 0:
                self._dump_viz(frame, obs, f"frame_{frame_id:05d}.jpg")
        cap.release()

    def _dump_viz(self, frame: np.ndarray, obs: Observation, name: str) -> None:
        last = self.traces[-1] if self.traces else None
        canvas = viz.draw(frame, obs, last.text if last else "")
        cv2.imwrite(os.path.join(self.out_dir, name), canvas)

    # ----------------------------------------------------------------- llm
    def _llm_loop(self) -> None:
        seq = 0
        while not self._stop.is_set():
            obs, seq = self.slot.wait_newer(seq, timeout=0.2)
            if obs is None:
                continue
            if self.engine is None:
                continue
            prompt = build_prompt(obs)
            self._llm_busy.set()
            try:
                reply: LLMReply = self.engine.run(prompt)
            except Exception as exc:
                print(f"[llm] run failed: {exc}")
                self._llm_busy.clear()
                continue
            finally:
                self._llm_busy.clear()

            trace = LLMTrace(
                frame_id=obs.frame_id,
                obs_ts=obs.ts,
                text=reply.text,
                ttft_ms=reply.ttft_ms,
                total_ms=reply.total_ms,
                e2e_ms=(time.time() - obs.ts) * 1e3,
                risk_level=obs.risk_level,
                rules=obs.rules,
            )
            self.traces.append(trace)
            self._jsonl.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
            self._jsonl.flush()
            print(
                f"[llm] frame={trace.frame_id} risk={trace.risk_level} "
                f"ttft={trace.ttft_ms:.0f}ms total={trace.total_ms:.0f}ms | {trace.text}"
            )

    # ----------------------------------------------------------------- run
    def run(self, duration_s: Optional[float] = None) -> dict:
        cv_t = threading.Thread(target=self._cv_loop, args=(duration_s,), name="cv", daemon=True)
        llm_t = threading.Thread(target=self._llm_loop, name="llm", daemon=True)
        llm_t.start()
        cv_t.start()
        try:
            cv_t.join()
        except KeyboardInterrupt:
            pass
        self._stop.set()
        llm_t.join(timeout=5.0)
        self._jsonl.close()
        return self.report()

    def report(self) -> dict:
        def q(vals, p):
            return float(np.percentile(vals, p)) if vals else 0.0

        cv, npu, head = self.stats["cv_ms"], self.stats["npu_ms"], self.stats["head_ms"]
        ttfts = [t.ttft_ms for t in self.traces]
        totals = [t.total_ms for t in self.traces]
        e2es = [t.e2e_ms for t in self.traces]
        report = {
            "affinity": self._affinity_note,
            "frames_read": self.stats["frames"],
            "infer_frames": self.stats["infer_frames"],
            "cv_total_p50_ms": q(cv, 50),
            "cv_total_p90_ms": q(cv, 90),
            "backbone_npu_p50_ms": q(npu, 50),
            "head_cpu_p50_ms": q(head, 50),
            "cv_fps_effective": (1000.0 / statistics.mean(cv)) if cv else 0.0,
            "llm_calls": len(self.traces),
            "llm_ttft_p50_ms": q(ttfts, 50),
            "llm_total_p50_ms": q(totals, 50),
            "llm_total_p90_ms": q(totals, 90),
            "e2e_p50_ms": q(e2es, 50),
            "e2e_p90_ms": q(e2es, 90),
            "observations_published": self.slot.published,
            "observations_dropped": max(0, self.slot.published - self.slot.consumed),
            "frames_while_llm_busy": self.stats["frames_while_llm_busy"],
        }
        with open(os.path.join(self.out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

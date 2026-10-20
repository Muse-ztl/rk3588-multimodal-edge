"""ctypes binding for the C++ RKLLM bridge.

The bridge exposes a tiny C ABI, so Python never has to mirror RKLLM's C
structs (which change between runtime releases). Streaming tokens are pushed
back through a ctypes callback, which gives us real TTFT measurement.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass

TEXT_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


@dataclass
class LLMReply:
    text: str
    ttft_ms: float
    total_ms: float
    segments: int


class RKLLMEngine:
    def __init__(
        self,
        lib_path: str,
        model_path: str,
        max_new_tokens: int = 64,
        max_context_len: int = 2048,
    ) -> None:
        self._lib = ctypes.CDLL(lib_path)
        self._lib.mmllm_init.restype = ctypes.c_int
        self._lib.mmllm_init.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            TEXT_CB, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.mmllm_run.restype = ctypes.c_int
        self._lib.mmllm_run.argtypes = [ctypes.c_char_p]
        self._lib.mmllm_destroy.restype = None
        self._lib.mmllm_last_error.restype = ctypes.c_char_p

        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._t_first = 0.0
        self._segments = 0

        self._cb = TEXT_CB(self._on_chunk)  # 必须保活，否则回调指针会被 GC
        err = ctypes.create_string_buffer(256)
        ret = self._lib.mmllm_init(
            model_path.encode("utf-8"), max_new_tokens, max_context_len,
            self._cb, None, err, 256,
        )
        if ret != 0:
            msg = err.value.decode("utf-8", "replace") or self.last_error()
            raise RuntimeError(f"rkllm init failed (ret={ret}): {msg}")
        self.model_path = model_path

    # ------------------------------------------------------------------ cb
    def _on_chunk(self, chunk: bytes, _user) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._t_first == 0.0:
                self._t_first = now
            self._segments += 1
            if chunk:
                self._chunks.append(chunk.decode("utf-8", "ignore"))

    # ------------------------------------------------------------------ api
    def last_error(self) -> str:
        p = self._lib.mmllm_last_error()
        return p.decode("utf-8", "replace") if p else "unknown"

    def run(self, prompt: str) -> LLMReply:
        with self._lock:
            self._chunks = []
            self._segments = 0
            self._t_first = 0.0
            self._t0 = time.perf_counter()

        ret = self._lib.mmllm_run(prompt.encode("utf-8"))
        t_end = time.perf_counter()
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed (ret={ret}): {self.last_error()}")

        with self._lock:
            text = "".join(self._chunks).strip()
            ttft = (self._t_first - self._t0) * 1e3 if self._t_first else 0.0
            segs = self._segments
        return LLMReply(text=text, ttft_ms=ttft, total_ms=(t_end - self._t0) * 1e3, segments=segs)

    def close(self) -> None:
        try:
            self._lib.mmllm_destroy()
        except Exception:
            pass

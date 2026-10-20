#!/usr/bin/env bash
# Build the RKLLM -> C ABI bridge on the board.
#
#   bash scripts/build_llm_bridge.sh [out_dir]
#
# Uses the runtime that ships with rknn-llm at /usr/local/lib (v1.2.3).
# /usr/lib/librkllmrt.so is an older build and will fail rkllm_init.
set -euo pipefail

RKLLM_ROOT="${RKLLM_ROOT:-$HOME/rknn-llm/rkllm-runtime/Linux/librkllm_api}"
OUT_DIR="${1:-build}"
SRC="$(dirname "$0")/../src/mm_edge/llm_bridge.cpp"

mkdir -p "$OUT_DIR"
echo "[build] header : $RKLLM_ROOT/include/rkllm.h"
echo "[build] runtime: /usr/local/lib/librkllmrt.so"

g++ -O2 -std=c++17 -fPIC -shared \
  -I"$RKLLM_ROOT/include" \
  "$SRC" \
  -o "$OUT_DIR/librkllm_bridge.so" \
  -L/usr/local/lib -lrkllmrt -Wl,-rpath,/usr/local/lib

ls -lh "$OUT_DIR/librkllm_bridge.so"
nm -D "$OUT_DIR/librkllm_bridge.so" | grep mmllm_ || true

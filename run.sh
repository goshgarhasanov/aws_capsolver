#!/usr/bin/env bash
# Start the API on 127.0.0.1:8790 (override with HOST / PORT).
set -euo pipefail
cd "$(dirname "$0")"
# Fast grid model (2026-11-20): ViT-B-32 alone + its own head - same held-out accuracy as
# B-32+B-16 (319/319) at ~125 ms instead of ~600 ms per grid. Both vars MUST change together
# (a backbone/head mismatch falls back to zero-shot). Override either to use the old pair.
export AWS_CAPSOLVER_CLIP_MODELS="${AWS_CAPSOLVER_CLIP_MODELS:-ViT-B-32:laion2b_s34b_b79k}"
export AWS_CAPSOLVER_HEAD="${AWS_CAPSOLVER_HEAD:-$PWD/weights/aws_capsolver_head_b32.pt}"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m aws_capsolver serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8790}" "$@"

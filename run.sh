#!/usr/bin/env bash
# Start the API on 127.0.0.1:8790 (override with HOST / PORT).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m aws_capsolver serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8790}" "$@"

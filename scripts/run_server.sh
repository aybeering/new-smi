#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the HNSW search service.
# Usage: METADATA_PATH=data/rules_metadata.json MODEL_PATH=dir PORT=8000 ./scripts/run_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export METADATA_PATH="${METADATA_PATH:-$ROOT_DIR/data/rules_metadata.json}"
export MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/dir}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

cd "$ROOT_DIR"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "[run_server] uvicorn not found; please install dependencies (e.g., pip install -r requirements.txt)" >&2
  exit 1
fi

exec uvicorn src.server:app --host "$HOST" --port "$PORT"


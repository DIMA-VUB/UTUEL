#!/usr/bin/env bash
# run_pipeline.sh — activate venv, run pipeline in background, stream log
#
# Usage:
#   bash run_pipeline.sh                        # default config
#   bash run_pipeline.sh pipeline.num_runs=3    # any Hydra override
#
# While running:
#   tail -f logs/pipeline_<timestamp>.log

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-venv}"          # override: VENV_DIR=.venv bash run_pipeline.sh
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# ── activate virtual environment ──────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found at '$VENV_DIR/bin/activate'" >&2
    echo "        Create one first:  python3 -m venv $VENV_DIR && pip install -r requirements.txt" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

# ── prepare log directory ─────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── launch in background ──────────────────────────────────────────────────────
echo "[INFO] Starting pipeline — logging to $LOG_FILE"
echo "[INFO] Follow with:  tail -f $LOG_FILE"

nohup python3 -m prompts_pipeline "$@" > "$LOG_FILE" 2>&1 &
PID=$!

echo "[INFO] PID $PID"
echo "$PID" > "$LOG_DIR/pipeline_${TIMESTAMP}.pid"

# ── tail the log ──────────────────────────────────────────────────────────────
# Ctrl-C exits the tail without killing the pipeline process.
tail -f "$LOG_FILE"

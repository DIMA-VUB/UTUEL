#!/usr/bin/env bash
# run_train.sh — Train TableEmbedJePA on a single GPU with configurable embedder.
#
# Hydra overrides are built from the arguments; output and logs are organised
# under a slug derived from the model name.
#
# USAGE:
#   bash run_train.sh [OPTIONS]
#
# OPTIONS:
#   -t  MODEL_TYPE      huggingface | openai | ollama       (default: huggingface)
#   -n  MODEL_NAME      full model identifier               (default: sentence-transformers/all-MiniLM-L6-v2)
#   -d  EMBED_DIM       output embedding dimension          (default: 384)
#   -m  TRAINING_MODE   global | per_table                  (default: global)
#   -g  GPU_ID          CUDA device index                   (default: 0)
#   -x  EXTRA           extra Hydra overrides (space-separated key=value pairs)
#   -s                  sweep mode — run Optuna HPO via sweep_optuna.yaml (--multirun)
#   -h                  show this help and exit
#
# EXAMPLES:
#   # Default — all-MiniLM-L6-v2, GPU 0
#   bash run_train.sh
#
#   # Nomic 768-d on GPU 1, per-table mode
#   bash run_train.sh -n "nomic-ai/nomic-embed-text-v1.5" -d 768 -g 1 -m per_table
#
#   # Ollama backend with extra Hydra overrides
#   bash run_train.sh -t ollama -n "nomic-embed-text" -d 768 \
#       -x "training.epochs=30 training.batch_size=32"
#
#   # Optuna HPO sweep (50 trials, config from sweep_optuna.yaml)
#   bash run_train.sh -s
#   bash run_train.sh -s -x "hydra.sweeper.n_trials=20"

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_TYPE="huggingface"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM=384
TRAINING_MODE="global"
GPU_ID=0
EXTRA=""
SWEEP=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while getopts ":t:n:d:m:g:x:sh" opt; do
    case $opt in
        t) MODEL_TYPE="$OPTARG" ;;
        n) MODEL_NAME="$OPTARG" ;;
        d) EMBED_DIM="$OPTARG" ;;
        m) TRAINING_MODE="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        x) EXTRA="$OPTARG" ;;
        s) SWEEP=true ;;
        h)
            sed -n '/^# USAGE/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        :) echo "ERROR: -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

# ── Filesystem slug: replace / \ : and whitespace with - ─────────────────────
SLUG="${MODEL_NAME//[\/\\: ]/-}"
TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="TRL_Adhesive/checkpoints/${SLUG}"
LOG_DIR="TRL_Adhesive/logs"
LOG_FILE="${LOG_DIR}/train_${SLUG}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────
SEP="$(printf '─%.0s' {1..60})"
echo -e "\033[36m${SEP}\033[0m"
echo -e "\033[36m  TableEmbedJePA — training run\033[0m"
echo -e "\033[36m${SEP}\033[0m"
echo "  GPU              : ${GPU_ID}  (CUDA_VISIBLE_DEVICES=${GPU_ID})"
echo "  Model type       : ${MODEL_TYPE}"
echo "  Model name       : ${MODEL_NAME}"
echo "  Embed dim        : ${EMBED_DIM}"
echo "  Training mode    : ${TRAINING_MODE}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  Log file         : ${LOG_FILE}"
$SWEEP && echo "  Sweep mode       : ON  (sweep_optuna.yaml, --multirun)"
[[ -n "$EXTRA" ]] && echo "  Extra overrides  : ${EXTRA}"
echo -e "\033[36m${SEP}\033[0m"
echo ""

# ── Build Hydra override array ────────────────────────────────────────────────
OVERRIDES=(
    "embedder.model_type=${MODEL_TYPE}"
    "embedder.model_name=${MODEL_NAME}"
    "embedder.embed_dim=${EMBED_DIM}"
	"model.hidden_size=${EMBED_DIM}"
    "training.mode=${TRAINING_MODE}"
)

# Append extra overrides (split on whitespace)
if [[ -n "$EXTRA" ]]; then
    read -ra EXTRA_ARR <<< "$EXTRA"
    OVERRIDES+=("${EXTRA_ARR[@]}")
fi

# ── Write run header to log ───────────────────────────────────────────────────
{
    echo "════════════════════════════════════════════════════════════"
    echo "  TableEmbedJePA — training run"
    echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  GPU        : ${GPU_ID}"
    echo "  Model      : ${MODEL_NAME}"
    echo "  Embed dim  : ${EMBED_DIM}"
    echo "  Mode       : ${TRAINING_MODE}"
    echo "  Output dir : ${OUTPUT_DIR}"
    [[ -n "$EXTRA" ]] && echo "  Extra      : ${EXTRA}"
    echo "════════════════════════════════════════════════════════════"
    echo ""
} > "$LOG_FILE"

# ── Launch in background ──────────────────────────────────────────────────────
# All stdout + stderr append to $LOG_FILE so `tail -f` works from any terminal.
# A footer with elapsed time and exit code is written when the job finishes.
PID_FILE="${LOG_FILE%.log}.pid"

(
    _start=$(date +%s)
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    if $SWEEP; then
        python TRL_Adhesive/train.py --config-name sweep_optuna --multirun "${OVERRIDES[@]}" && _ec=0 || _ec=$?
    else
        python TRL_Adhesive/train.py "${OVERRIDES[@]}" && _ec=0 || _ec=$?
    fi
    _elapsed=$(( $(date +%s) - _start ))
    _fmt=$(printf '%02d:%02d:%02d' $(( _elapsed/3600 )) $(( (_elapsed%3600)/60 )) $(( _elapsed%60 )))
    echo ""
    echo "════════════════════════════════════════════════════════════"
    if [[ $_ec -eq 0 ]]; then
        echo "  Done  — elapsed ${_fmt}"
    else
        echo "  FAILED (exit ${_ec}) — elapsed ${_fmt}"
    fi
    echo "  Log  → ${LOG_FILE}"
    echo "  Ckpt → ${OUTPUT_DIR}"
    echo "════════════════════════════════════════════════════════════"
    rm -f "$PID_FILE"
    exit $_ec
) >> "$LOG_FILE" 2>&1 &

BG_PID=$!
echo "$BG_PID" > "$PID_FILE"

# ── Print monitor commands and exit ──────────────────────────────────────────
echo -e "\033[32m[train] Launched in background — PID ${BG_PID}\033[0m"
echo ""
echo -e "\033[36m  tail -f ${LOG_FILE}\033[0m"
echo ""
echo "  Stop  : kill ${BG_PID}"
echo "  PID   : ${PID_FILE}"

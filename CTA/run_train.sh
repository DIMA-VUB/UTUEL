#!/usr/bin/env bash
# CTA/run_train.sh — Train the CTA column type annotation model.
#
# Supports two sequential stages (pretrain → finetune) or either alone.
# Hydra overrides are passed through to the underlying Python scripts.
#
# USAGE:
#   bash CTA/run_train.sh [OPTIONS]
#
# OPTIONS:
#   -t  MODEL_TYPE    huggingface | openai | ollama  (default: huggingface)
#   -n  MODEL_NAME    full model identifier          (default: sentence-transformers/all-MiniLM-L6-v2)
#   -d  EMBED_DIM     embedding dimension            (default: 384)
#   -g  GPU_ID        CUDA device index              (default: 0)
#   -x  EXTRA         extra Hydra overrides          (space-separated key=value)
#   -s  STAGE         pretrain | finetune | both     (default: both)
#   --sweep           HPO sweep via sweep_optuna.yaml (--multirun)
#   -h                show this help and exit
#
# EXAMPLES:
#   # Full pipeline: pretrain then finetune (default embedder)
#   bash CTA/run_train.sh
#
#   # Fine-tune only from an existing pretrained checkpoint
#   bash CTA/run_train.sh -s finetune \
#       -x "finetuning.pretrained_ckpt=CTA/checkpoints/pretrain/last.ckpt"
#
#   # Optuna HPO sweep on the fine-tuning stage
#   bash CTA/run_train.sh --sweep -s finetune
#
#   # Use a custom Ollama model, GPU 1
#   bash CTA/run_train.sh -t ollama -n "nomic-embed-text" -d 768 -g 1

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_TYPE="huggingface"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM=384
GPU_ID=0
EXTRA=""
STAGE="both"
SWEEP=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) MODEL_TYPE="$2";  shift 2 ;;
        -n) MODEL_NAME="$2";  shift 2 ;;
        -d) EMBED_DIM="$2";   shift 2 ;;
        -g) GPU_ID="$2";      shift 2 ;;
        -x) EXTRA="$2";       shift 2 ;;
        -s) STAGE="$2";       shift 2 ;;
        --sweep) SWEEP=true;  shift ;;
        -h|--help)
            sed -n '/^# USAGE/,/^[^#]/p' "$0" | head -n -1
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Build common Hydra overrides ──────────────────────────────────────────────
SLUG="${MODEL_NAME//\//_}"
# NOTE: do NOT include the slug in these base dirs.
# pretrain.py and finetune.py append  <slug>/<timestamp>_<cfghash>  automatically.
OVERRIDES=(
    "embedder.model_type=${MODEL_TYPE}"
    "embedder.model_name=${MODEL_NAME}"
    "embedder.embed_dim=${EMBED_DIM}"
    "pretraining.output_dir=CTA/checkpoints/pretrain"
    "finetuning.output_dir=CTA/checkpoints/finetune"
    "eval.output_dir=CTA/outputs"
)
if [[ -n "${EXTRA}" ]]; then
    read -ra EXTRA_ARRAY <<< "${EXTRA}"
    OVERRIDES+=("${EXTRA_ARRAY[@]}")
fi

OVERRIDES_STR="${OVERRIDES[*]}"

# ── Set CUDA device ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ── Helper: run python script ─────────────────────────────────────────────────
run_python() {
    local script="$1"
    local multirun="${2:-false}"
    if [[ "${multirun}" == "true" ]]; then
        python "${script}" --config-name sweep_optuna --multirun ${OVERRIDES_STR}
    else
        python "${script}" ${OVERRIDES_STR}
    fi
}

# ── Execute requested stage(s) ────────────────────────────────────────────────
case "${STAGE}" in
    pretrain)
        echo "=== [CTA] Stage: pretrain ==="
        run_python "CTA/pretrain.py" "${SWEEP}"
        ;;
    finetune)
        echo "=== [CTA] Stage: finetune ==="
        run_python "CTA/finetune.py" "${SWEEP}"
        ;;
    both)
        echo "=== [CTA] Stage 1/2: pretrain ==="
        run_python "CTA/pretrain.py" false
        # Discover the most recent checkpoint written by pretrain.py.
        # Python writes to:  CTA/checkpoints/pretrain/<slug>/<ts>_<hash>/last.ckpt
        PT_BASE="CTA/checkpoints/pretrain/${SLUG}"
        PRETRAIN_CKPT=$(ls -t "${PT_BASE}"/*/last.ckpt 2>/dev/null | head -1)
        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            echo "[ERROR] No pretrain checkpoint found under ${PT_BASE}/"
            exit 1
        fi
        echo "=== [CTA] Stage 2/2: finetune (from ${PRETRAIN_CKPT}) ==="
        OVERRIDES+=("finetuning.pretrained_ckpt=${PRETRAIN_CKPT}")
        OVERRIDES_STR="${OVERRIDES[*]}"
        run_python "CTA/finetune.py" "${SWEEP}"
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use pretrain | finetune | both."
        exit 1
        ;;
esac

echo "=== [CTA] Done ==="

#!/usr/bin/env bash
# CTA/run_experiments.sh - Submit pretrain or finetune jobs to SLURM.
#
# USAGE:
#   bash CTA/run_experiments.sh pretrain
#   bash CTA/run_experiments.sh finetune

set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "pretrain" && "$1" != "finetune" ) ]]; then
    echo "Usage: $0 pretrain|finetune"
    exit 1
fi
STAGE="$1"

# -- Cluster settings ----------------------------------------------------------
PARTITION=ampere_gpu
TIME=0-8:00:00
MEM=64G
SBATCH_SCRIPT="sbatch_job_CTA.sh"

# -- Embedder models -----------------------------------------------------------
# Format: "model_type|model_name|embed_dim"
MODELS=(
    "ollama|all-minilm|384"
    "ollama|nomic-embed-text|768"
    "ollama|qwen3-embedding:0.6b|768"
    "huggingface|sentence-transformers/all-MiniLM-L6-v2|384"
    "huggingface|sentence-transformers/all-mpnet-base-v2|768"
    "ollama|embeddinggemma|768"
)

# -- Pretrain checkpoints (required when STAGE=finetune) ----------------------
# Key   = model slug  (last part of model_name, / -> _, : -> #)
# Value = exact path to last.ckpt from a previous pretrain run.
#         Leave "" to let sbatch_job.sh auto-discover via PT_BASE scan.
#
# Fill in after pretrain jobs complete:
#   ls CTA/checkpoints/pretrain/<slug>/*/last.ckpt
declare -A PT_CKPTS
PT_CKPTS["all-minilm"]=""
PT_CKPTS["nomic-embed-text"]=""
PT_CKPTS["qwen3-embedding#0.6b"]=""
PT_CKPTS["all-MiniLM-L6-v2"]=""
PT_CKPTS["all-mpnet-base-v2"]=""
PT_CKPTS["embeddinggemma"]=""

# -- Finetune combos -----------------------------------------------------------
# "freeze_encoder|embed_mode|label"
FT_ENCODER_COMBOS=(
    "false|column|joint-col"
    "false|cell|joint-cell"
    "true|column|frozen-col"
    "true|cell|frozen-cell"
)
# "embed_mode|label"
FT_NOENC_COMBOS=(
    "column|noenc-col"
    "cell|noenc-cell"
)

# -- Submit loop ---------------------------------------------------------------
for MODEL_CFG in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_TYPE MODEL_NAME EMBED_DIM <<< "${MODEL_CFG}"

    SLUG="${MODEL_NAME##*/}"
    SLUG="${SLUG// /_}"; SLUG="${SLUG//:/#}"

    echo ""
    echo "==== ${MODEL_NAME} (${MODEL_TYPE}, dim=${EMBED_DIM}, slug=${SLUG}) ===="

    # -- Case 1: Pretrain ------------------------------------------------------
    if [[ "${STAGE}" == "pretrain" ]]; then
        JID=$(STAGE=pretrain \
            MODEL_TYPE="${MODEL_TYPE}" MODEL_NAME="${MODEL_NAME}" EMBED_DIM="${EMBED_DIM}" \
            sbatch --parsable \
                --job-name="CTA_PT_${SLUG}" \
                --partition="${PARTITION}" --time="${TIME}" --mem="${MEM}" \
                --export=ALL \
                "${SBATCH_SCRIPT}" )
        echo "  pretrain    JID=${JID}"

    # -- Case 2: Finetune ------------------------------------------------------
    elif [[ "${STAGE}" == "finetune" ]]; then
        KNOWN_CKPT="${PT_CKPTS[${SLUG}]:-}"
        PT_CKPT_ARG="${KNOWN_CKPT}"
        PT_BASE_ARG="${KNOWN_CKPT:+}"
        [[ -z "${KNOWN_CKPT}" ]] && PT_BASE_ARG="CTA/checkpoints/pretrain/${SLUG}"

        if [[ -n "${KNOWN_CKPT}" ]]; then
            echo "  checkpoint  ${KNOWN_CKPT}"
        else
            echo "  checkpoint  (auto-discover from ${PT_BASE_ARG})"
        fi

        # Finetune with encoder
        for FT_CFG in "${FT_ENCODER_COMBOS[@]}"; do
            IFS='|' read -r FREEZE EM LABEL <<< "${FT_CFG}"
            JID=$(STAGE=finetune USE_ENCODER=true \
                MODEL_TYPE="${MODEL_TYPE}" MODEL_NAME="${MODEL_NAME}" EMBED_DIM="${EMBED_DIM}" \
                FREEZE_ENCODER="${FREEZE}" EMBED_MODE="${EM}" \
                PT_CKPT="${PT_CKPT_ARG}" PT_BASE="${PT_BASE_ARG}" \
                sbatch --parsable \
                    --job-name="CTA_FT_${SLUG}_${LABEL}" \
                    --partition="${PARTITION}" --time="${TIME}" --mem="${MEM}" \
                    --export=ALL \
                    "${SBATCH_SCRIPT}")
            echo "  finetune/${LABEL}    JID=${JID}"
        done

        # No-encoder baselines
        for FT_CFG in "${FT_NOENC_COMBOS[@]}"; do
            IFS='|' read -r EM LABEL <<< "${FT_CFG}"
            JID=$(STAGE=finetune USE_ENCODER=false \
                MODEL_TYPE="${MODEL_TYPE}" MODEL_NAME="${MODEL_NAME}" EMBED_DIM="${EMBED_DIM}" \
                EMBED_MODE="${EM}" \
                sbatch --parsable \
                    --job-name="CTA_FT_${SLUG}_${LABEL}" \
                    --partition="${PARTITION}" --time="${TIME}" --mem="${MEM}" \
                    --export=ALL \
                    "${SBATCH_SCRIPT}")
            echo "  finetune/${LABEL}    JID=${JID}"
        done

    else
        echo "[ERROR] Unknown STAGE='${STAGE}'. Set to 'pretrain' or 'finetune'."
        exit 1
    fi
done

echo ""
echo "All ${STAGE} jobs submitted. Monitor: squeue -u \$USER"
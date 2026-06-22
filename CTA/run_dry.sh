#!/bin/bash
#SBATCH --output=logs/utuel_%x_%j.txt
#SBATCH --job-name=UTUEL_CTA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ampere_gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Job parameters (exported by run_experiments.sh via env) ───────────────────
#
#  STAGE         pretrain | finetune                    (required)
#  MODEL_TYPE    huggingface | ollama | openai
#  MODEL_NAME    full model identifier
#  EMBED_DIM     embedding dimension (int)
#  FREEZE_ENCODER  true | false                         (finetune only)
#  EMBED_MODE    column | cell                          (finetune only)
#  USE_ENCODER   true | false                           (finetune only)
#                  false → no encoder (raw LLM embs → head)
#  PT_CKPT       explicit pretrain checkpoint path      (finetune only, optional)
#                  if set, skips auto-discovery of the checkpoint
#  PT_BASE       base dir to search for pretrain ckpt   (finetune only, optional)
#                  default: CTA/checkpoints/pretrain/<slug>
#                  ignored when PT_CKPT is set directly
#  EXTRA_ARGS    extra space-separated Hydra overrides  (optional)

STAGE="${STAGE:-pretrain}"
MODEL_TYPE="${MODEL_TYPE:-huggingface}"
MODEL_NAME="${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
EMBED_DIM="${EMBED_DIM:-384}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
EMBED_MODE="${EMBED_MODE:-column}"
USE_ENCODER="${USE_ENCODER:-true}"
PT_CKPT="${PT_CKPT:-}"            # explicit checkpoint path (overrides auto-discovery)
PT_BASE="${PT_BASE:-}"            # base search dir (falls back to default when empty)
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[INFO] ================================================================="
echo "[INFO] STAGE=${STAGE}"
echo "[INFO] MODEL_TYPE=${MODEL_TYPE}  MODEL_NAME=${MODEL_NAME}  EMBED_DIM=${EMBED_DIM}"
if [[ "${STAGE}" == "finetune" ]]; then
    echo "[INFO] USE_ENCODER=${USE_ENCODER}  FREEZE_ENCODER=${FREEZE_ENCODER}  EMBED_MODE=${EMBED_MODE}"
fi
echo "[INFO] ================================================================="

# ── Model slug (matches Python logic in pretrain.py / finetune.py) ─────────────
# cfg.embedder.model_name.split("/")[-1].replace(" ","_").replace(":","#")
SLUG="${MODEL_NAME##*/}"
SLUG="${SLUG// /_}"
SLUG="${SLUG//:/#}"

# ── Build common Hydra overrides ──────────────────────────────────────────────
# OVERRIDES=(
#     "embedder.model_type=${MODEL_TYPE}"
#     "embedder.model_name=${MODEL_NAME}"
#     "embedder.embed_dim=${EMBED_DIM}"
#     "pretraining.output_dir=CTA/checkpoints/pretrain"
#     "finetuning.output_dir=CTA/checkpoints/finetune"
#     "eval.output_dir=CTA/outputs"
# )

OVERRIDES=(
    "data.max_records=null"
    "data.max_rows_train=10"
    "data.max_rows_dev=1"
    "data.max_rows_test=1"
    "data.folder=/scratch/brussel/vo/000/bvo00018/vsc10413/CTA_Dataset/"
    "pretraining.epochs=50"
    "pretraining.batch_size=64" 
    "pretraining.dataloader_num_workers=4"
    "pretraining.output_dir=/scratch/brussel/vo/000/bvo00018/vsc10413/UTUEL_OUTPUT_CTA/checkpoints/pretrain"
    "finetuning.epochs=50"
    "finetuning.batch_size=64"
    "finetuning.dataloader_num_workers=4"
    #"finetuning.pretrained_ckpt=/scratch/brussel/vo/000/bvo00018/vsc10413/UTUEL_OUTPUT_CTA/checkpoints/pretrain/last-v1.ckpt"
    "finetuning.output_dir=/scratch/brussel/vo/000/bvo00018/vsc10413/UTUEL_OUTPUT_CTA/checkpoints/pretrain_finetune"
    "eval.output_dir=/scratch/brussel/vo/000/bvo00018/vsc10413/UTUEL_OUTPUT_CTA/checkpoints/pretrain_finetune_no_enc"
    "embedder.model_type=${MODEL_TYPE}"
    "embedder.model_name=${MODEL_NAME}"
    "embedder.embed_dim=${EMBED_DIM}"
    "embedder.cache_embeddings=true"
    "embedder.embed_cache_dir=/scratch/brussel/vo/000/bvo00018/vsc10413/CTA_Dataset/.cache"
    "model.num_layers=1"
)


# ── Finetune-specific overrides ───────────────────────────────────────────────
if [[ "${STAGE}" == "finetune" ]]; then
    OVERRIDES+=("classifier.embed_mode=${EMBED_MODE}")
    
    if [[ "${USE_ENCODER}" == "true" ]]; then
        # Resolve the pretrain checkpoint:
        #   1. Use PT_CKPT directly if the caller set it (most precise).
        #   2. Search PT_BASE if the caller set a base dir.
        #   3. Fall back to the default slug-based dir.
        if [[ -z "${PT_CKPT}" ]]; then
            _SEARCH_BASE="${PT_BASE:-CTA/checkpoints/pretrain/${SLUG}}"
            PT_CKPT=$(ls -t "${_SEARCH_BASE}"/*/last.ckpt 2>/dev/null | head -1)
        fi
        if [[ -z "${PT_CKPT}" ]]; then
            echo "[ERROR] No pretrain checkpoint found."
            echo "[ERROR] Set PT_CKPT=<path> or PT_BASE=<dir> or run pretrain first."
            exit 1
        fi
        echo "[INFO] Using pretrain checkpoint: ${PT_CKPT}"
        OVERRIDES+=("finetuning.pretrained_ckpt=${PT_CKPT}")
        OVERRIDES+=("classifier.freeze_encoder=${FREEZE_ENCODER}")
    else
        # No encoder: freeze_encoder=true + no pretrained_ckpt → raw LLM → head
        OVERRIDES+=("classifier.freeze_encoder=true")
        OVERRIDES+=("finetuning.pretrained_ckpt=null")
    fi
fi

# ── Extra overrides (appended last so they win) ───────────────────────────────
# Accepts overrides from two sources (both can be used together):
#   1. EXTRA_ARGS env var — space-separated string, e.g. from run_experiments.sh
#   2. Positional args    — array, e.g. bash sbatch_job.sh "key=v1" "key=v2"
#                           or: sbatch sbatch_job.sh key=v1 key=v2
# Positional args are appended last and therefore have highest priority.
if [[ -n "${EXTRA_ARGS}" ]]; then
    read -ra EXTRA_ARR <<< "${EXTRA_ARGS}"
    OVERRIDES+=("${EXTRA_ARR[@]}")
fi
if [[ $# -gt 0 ]]; then
    OVERRIDES+=("$@")
fi

OVERRIDES_STR="${OVERRIDES[*]}"

# ── Environment setup ─────────────────────────────────────────────────────────
module purge
ml load ollama/0.15.6-GCCcore-14.2.0-CUDA-12.8.0
ml load Python/3.13.1-GCCcore-14.2.0

source "$VSC_DATA_VO/.venv_utuel_pascal_gpu/bin/activate"

# ── Ollama (only needed for ollama model_type) ────────────────────────────────
OLLAMA_PID=""
if [[ "${MODEL_TYPE}" == "ollama" ]]; then
    export OLLAMA_MAX_LOADED_MODELS=4
    export OLLAMA_MAX_QUEUE=2048
    export OLLAMA_NUM_PARALLEL=4
    export OLLAMA_MODELS=/data/brussel/vo/000/bvo00018/.ollama/models

    if ! pgrep -x "ollama" > /dev/null; then
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        echo "[INFO] Ollama started (PID ${OLLAMA_PID})"
    else
        echo "[INFO] Ollama already running — skipping serve"
        OLLAMA_PID=$(pgrep -x "ollama" | head -1)
    fi
    echo "[INFO] OLLAMA_MODELS=${OLLAMA_MODELS}"
    echo "[INFO] OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
    echo "[INFO] OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
    echo "[INFO] OLLAMA_MAX_QUEUE=${OLLAMA_MAX_QUEUE}"
    sleep 5
fi

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p logs

CMD="python CTA/${STAGE}.py ${OVERRIDES_STR}"
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') — executing:"
echo "[INFO] ${CMD}"
$CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] ${STAGE} failed with exit code ${EXIT_CODE}"
else
    echo "[OK] ${STAGE} completed successfully"
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [[ -n "${OLLAMA_PID}" ]]; then
    kill "${OLLAMA_PID}" 2>/dev/null && echo "[INFO] Ollama (PID ${OLLAMA_PID}) stopped"
fi

exit $EXIT_CODE

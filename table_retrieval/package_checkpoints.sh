#!/usr/bin/env bash
set -euo pipefail

# Package checkpoints listed in a YAML config.
# For each checkpoint_dir entry, this script copies:
#   1) the checkpoint directory itself
#   2) run_config.yaml from the parent of that checkpoint directory
# It preserves relative structure under the provided MAIN_DIR and creates a zip.
#
# Usage:
#   bash table_retrieval/package_checkpoints.sh <MAIN_DIR> [CONFIG_FILE] [OUT_DIR]
#
# Example:
#   bash table_retrieval/package_checkpoints.sh \
#     "D:/UTUEL_OUTPUT/collect_results_revision/checkpoints" \
#     "table_retrieval/config_embed_asses.yaml" \
#     "table_retrieval/checkpoint_bundle"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAIN_DIR="${1:-}"
CONFIG_FILE="${2:-${SCRIPT_DIR}/config_embed_asses.yaml}"
OUT_DIR="${3:-${SCRIPT_DIR}/checkpoint_bundle}"

if [[ -z "${MAIN_DIR}" ]]; then
    echo "[ERROR] MAIN_DIR is required."
    echo "Usage: bash table_retrieval/package_checkpoints.sh <MAIN_DIR> [CONFIG_FILE] [OUT_DIR]"
    exit 1
fi

if [[ ! -d "${MAIN_DIR}" ]]; then
    echo "[ERROR] MAIN_DIR not found: ${MAIN_DIR}"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] CONFIG_FILE not found: ${CONFIG_FILE}"
    exit 1
fi

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE_DIR="${OUT_DIR}/checkpoint_package_${STAMP}"
ZIP_PATH="${OUT_DIR}/checkpoint_package_${STAMP}.zip"
mkdir -p "${STAGE_DIR}"

# Extract checkpoint_dir values from YAML.
# Works with lines like:
#   checkpoint_dir: model\\run_id\\final
mapfile -t CHECKPOINT_DIRS < <(
    awk -F: '/^[[:space:]]*checkpoint_dir[[:space:]]*:/ {
    val = substr($0, index($0, ":") + 1)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
    gsub(/^"|"$/, "", val)
    print val
    }' "${CONFIG_FILE}" | sort -u
)

if [[ "${#CHECKPOINT_DIRS[@]}" -eq 0 ]]; then
    echo "[ERROR] No checkpoint_dir entries found in ${CONFIG_FILE}"
    exit 1
fi

echo "[INFO] Found ${#CHECKPOINT_DIRS[@]} checkpoint_dir entries"

copied_ckpt=0
copied_cfg=0
missing_ckpt=0
missing_cfg=0

for rel_raw in "${CHECKPOINT_DIRS[@]}"; do
    # Normalize Windows-style separators for shell path handling.
    rel="${rel_raw//\\//}"
    
    ckpt_abs="${MAIN_DIR}/${rel}"
    parent_rel="$(dirname "${rel}")"
    parent_abs="${MAIN_DIR}/${parent_rel}"
    run_cfg_abs="${parent_abs}/run_config.yaml"
    
    dest_parent="${STAGE_DIR}/${parent_rel}"
    mkdir -p "${dest_parent}"
    
    if [[ -d "${ckpt_abs}" ]]; then
        cp -a "${ckpt_abs}" "${dest_parent}/"
        copied_ckpt=$((copied_ckpt + 1))
        echo "[OK] checkpoint copied: ${rel}"
    else
        missing_ckpt=$((missing_ckpt + 1))
        echo "[WARN] checkpoint missing: ${ckpt_abs}"
    fi
    
    if [[ -f "${run_cfg_abs}" ]]; then
        cp -a "${run_cfg_abs}" "${dest_parent}/run_config.yaml"
        copied_cfg=$((copied_cfg + 1))
        echo "[OK] run_config copied: ${parent_rel}/run_config.yaml"
    else
        missing_cfg=$((missing_cfg + 1))
        echo "[WARN] run_config missing: ${run_cfg_abs}"
    fi
    
done

# Create zip archive using Python stdlib (avoids dependency on zip CLI).
python - "${STAGE_DIR}" "${ZIP_PATH}" <<'PY'
import pathlib
import sys
import zipfile

stage = pathlib.Path(sys.argv[1])
zip_path = pathlib.Path(sys.argv[2])

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in stage.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(stage))

print(f"[OK] zip created: {zip_path}")
PY

echo
echo "[SUMMARY]"
echo "  checkpoints copied : ${copied_ckpt}"
echo "  run_config copied  : ${copied_cfg}"
echo "  missing checkpoints: ${missing_ckpt}"
echo "  missing run_config : ${missing_cfg}"
echo "  stage folder       : ${STAGE_DIR}"
echo "  zip file           : ${ZIP_PATH}"

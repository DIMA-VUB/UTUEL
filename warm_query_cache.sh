#!/usr/bin/env bash
# warm_query_cache.sh
# Embeds 10 hardcoded queries with each Ollama model and saves embeddings/<model>.json
# Usage: ./warm_query_cache.sh  |  OLLAMA_URL=http://host:11434 ./warm_query_cache.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/query_embeddings"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

mkdir -p "$OUT_DIR"
QUERIES_FILE="${OUT_DIR}/queries.json"

# ── Hardcoded queries ─────────────────────────────────────────────────────────
python3 - "$QUERIES_FILE" <<'PYEOF'
import sys, json
queries = [
    "What is terrence ross' nationality",
    "What clu was in toronto 1995-96",
    "which club was in toronto 2003-06",
    "Where was Assen held?",
    "What was the date of the race in Misano?",
    "What are the nationalities of the player picked from Thunder Bay Flyers (ushl)",
    "What's Dorain Anneck's pick number?",
    "What is the nationality of the player from Vancouver Canucks?",
    "What's the pick number of the player from Springfield Olympics (Nejhl)?",
    "When were the ships launched that were laid down on september 1, 1964?",
]
open(sys.argv[1], "w").write(json.dumps(queries, ensure_ascii=False, indent=2))
print(f"queries : {len(queries)}")
PYEOF

# ── Hardcoded model list ──────────────────────────────────────────────────────
MODELS=(
    "qwen3-embedding:0.6b"
    "nomic-embed-text"
    "all-minilm"
    "embeddinggemma"
)

echo "models  : ${#MODELS[@]}"
echo ""

# ── One curl batch per model ──────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    SAFE="${MODEL//:/_}"
    OUT_FILE="${OUT_DIR}/${SAFE}.json"
    PAYLOAD_FILE=$(mktemp)
    
    python3 -c "
import json
q = json.load(open('${QUERIES_FILE}'))
print(json.dumps({'model': '${MODEL}', 'input': q}))
    " > "$PAYLOAD_FILE"
    
    echo "► ${MODEL}"
    curl -sf "${OLLAMA_URL}/api/embed" \
    -H "Content-Type: application/json" \
    -d "@${PAYLOAD_FILE}" \
    -o "${OUT_FILE}"
    rm -f "$PAYLOAD_FILE"
    
    python3 -c "
import json
d = json.load(open('${OUT_FILE}'))
e = d.get('embeddings', [])
print(f'  saved {len(e)} embeddings  dim={len(e[0]) if e else \"n/a\"}  →  ${SAFE}.json')
    "
    echo ""
done

echo "Output files:"
ls -lh "${OUT_DIR}/"

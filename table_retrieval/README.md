# Table Retrieval Benchmark

Evaluates table embedding models on **table retrieval**: given a natural-language question, rank all corpus tables and measure how highly the correct table is ranked.

## Directory layout

```
table_retrieval/
├── config.yaml          # evaluation config (datasets, embedder entries, paths)
├── evaluate.py          # main evaluation script
├── report.py            # compiles per-run JSONs → CSV / Markdown / HTML report
├── embedder.py          # HuggingFace and Ollama embedder backends + factory
├── utuel_embedder.py    # UTUEL / TableEmbedJePA custom embedder
├── abstract.py          # TableRetrieverBase ABC and embedder registry
├── retrieve_eval.ipynb  # interactive notebook (mirrors evaluate.py)
└── results/             # output — one JSON per run + report.{csv,md,html}
```

---

## Quick start

All commands are run from the **repository root**.

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Configure `config.yaml`

Edit `table_retrieval/config.yaml`:

| Key | Purpose |
|-----|---------|
| `output_dir` | where per-run JSONs and reports are written |
| `checkpoints_dir` | base directory for `checkpoint_dir` values in embedder entries |
| `cache_dir` | base directory for `cache_file` values in embedder entries |
| `retrieval.k_values` | Hit@k cut-offs (default `[1, 3, 5, 10, 20]`) |
| `retrieval.tsr_top_k` | node vectors scanned per query in TSR evaluation |
| `retrieval.tsr_mrr_depth` | unique tables used for MRR in TSR evaluation |

Each entry under `datasets` defines one evaluation run:

```yaml
datasets:
  - name: test_lookup_WikiSQL          # label used in the report
    path: datasets/test_lookup_WikiSQL.jsonl
    embedder:
      type: huggingface
      model_name: sentence-transformers/all-MiniLM-L6-v2
      batch_size: 64
```

### 3 — Run the evaluation

```bash
python table_retrieval/evaluate.py
# or point at a different config:
python table_retrieval/evaluate.py --config table_retrieval/config.yaml
```

Results are written to `output_dir`:
- `<label>.json` — full per-run metrics
- `<label>_top5.json` — top-5 retrieved tables per query (for inspection)
- `manifest.json` — combined summary of all runs

### 4 — Compile the report

```bash
python table_retrieval/report.py
```

Produces `results/report.csv`, `results/report.md`, and `results/report.html`.

---

## Supported embedder types

### `huggingface`

Loads a [sentence-transformers](https://www.sbert.net/) model locally. No server needed.

```yaml
embedder:
  type: huggingface
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 64
```

### `ollama`

Uses [langchain-ollama](https://github.com/langchain-ai/langchain) to call a local Ollama server. Supports asymmetric embedding (separate document and query paths).

```yaml
embedder:
  type: ollama
  base_url: http://localhost:11434
  model_name: nomic-embed-text
  batch_size: 1
  # max_chars: auto   # omit or "auto" to detect context window from /api/show
```

`max_chars` caps text length before embedding. When omitted or set to `"auto"`, the embedder queries `/api/show` to read the model's context window and derives a safe character limit automatically:

$$\text{max\_chars} = \lfloor \text{ctx\_tokens} \times 3.5 \times 0.9 \rfloor$$

If the server is unreachable it falls back to 4096 chars.

### `custom`

Any class registered with `@register_embedder` (see [Adding a new embedder](#adding-a-new-embedder-method)).

```yaml
embedder:
  type: custom
  class: UTUELTableEmbedder
  model_name: UTUEL/all-MiniLM-L6-v2
  checkpoint_dir: all-MiniLM-L6-v2/run_id/final   # relative to checkpoints_dir
  cache_file: my_cache.embed_cache.pt              # relative to cache_dir
  batch_size: 128
  pool: mean    # mean | max
```

`checkpoint_dir` and `cache_file` values are resolved against `checkpoints_dir` / `cache_dir` unless they are already absolute paths.

---

## Metrics

### Pooled-embedding metrics (standard)

Computed over the `[T, D]` corpus matrix returned by `encode_table_corpus_variants()`.

| Metric | Description |
|--------|-------------|
| `MRR` | Mean Reciprocal Rank over all queries |
| `Hit@k` | Fraction of queries where the correct table appears in the top *k* |

Three pooling variants are reported for the UTUEL embedder: `node_a`, `node_b`, `both`.

### TSR metrics (UTUEL only)

Top-Score-Rank evaluation over the **unpooled** global node index — mirrors the post-training evaluation in `TRL-model/train.py`.

Instead of pooling nodes into one table vector, all `node_a + node_b` embeddings from every table are concatenated into a global index. For each query:

1. The top-`tsr_top_k` highest-cosine-scoring node vectors are retrieved.
2. Their `table_id` labels are de-duplicated in descending-score order → unique table ranking.
3. Hit@k and MRR (up to depth `tsr_mrr_depth`) are computed on that ranking.

Four search spaces are reported:

| TSR space | Index contents |
|-----------|---------------|
| `tsr` | all `node_a + node_b` vectors |
| `col_tsr` | column-level mean of `node_a` |
| `row_tsr` | row-level mean of `node_a` |
| `tbl_tsr` | table-level mean of `node_a` (one vector per table) |

---

## Adding a new embedder method

### Option A — YAML only (HuggingFace / Ollama models)

Add an entry to `config.yaml` under `datasets`. No Python changes needed.

### Option B — Custom Python class

1. **Create** your embedder class in a new file (e.g. `table_retrieval/my_embedder.py`) or add it to an existing module:

   ```python
   from abstract import TableRetrieverBase, register_embedder
   import numpy as np

   @register_embedder
   class MyEmbedder(TableRetrieverBase):
       label = "my-model"

       def __init__(self, embedder_cfg: dict, project_root):
           # initialise your model here
           ...

       def encode_table_corpus(self, records: list[dict]) -> np.ndarray:
           # return [T, D] float32, L2-normalised
           ...

       def encode_queries(self, queries: list[str]) -> np.ndarray:
           # return [Q, D] float32, L2-normalised
           ...
   ```

2. **Register auto-import** — add your module name to the list in `build_embedder()` inside `embedder.py`:

   ```python
   for _mod in ("utuel_embedder", "my_embedder"):
   ```

3. **Add a YAML entry**:

   ```yaml
   - name: test_lookup_WikiSQL
     path: datasets/test_lookup_WikiSQL.jsonl
     embedder:
       type: custom
       class: MyEmbedder
       model_name: my-model-label
       # any extra keys your __init__ reads from embedder_cfg
   ```

4. Run `python table_retrieval/evaluate.py`.

### Optional: TSR metrics for a custom embedder

Implement `encode_table_corpus_variants()` returning `{"node_a": ..., "node_b": ..., "both": ...}` and store the global node index as `self._global_node_embs` / `self._global_node_tids` etc. (see `utuel_embedder.py` for the full pattern). `evaluate.py` will automatically detect and call `compute_tsr_metrics()` if it exists.

---

## Dataset format

Each `.jsonl` file contains one JSON object per line:

```json
{
  "id": "query-001",
  "table_id": "table-abc",
  "question": "How many goals did the top scorer score?",
  "header": ["Player", "Goals", "Assists"],
  "rows": [["Alice", "12", "5"], ["Bob", "9", "8"]]
}
```

Multiple queries may share the same `table_id`. The evaluation deduplicates tables automatically.

"""
evaluate.py
Table retrieval evaluation pipeline.

For every dataset entry in config.yaml:
  1. Load the JSONL dataset.
  2. Extract unique tables (by table_id) and serialise each as a document.
  3. Embed all table documents.
  4. Embed all queries (questions).
  5. Rank tables per query by cosine similarity.
  6. Compute retrieval metrics: MRR, Hit@1/3/5/10/20.
  7. Save per-run results to `output_dir/<run_id>.json`.
  8. Save top-k retrieved table_ids per query for inspection.

Run from the repo root:
    python table_retrieval/evaluate.py
    python table_retrieval/evaluate.py --config table_retrieval/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent        # table_retrieval/
PROJECT_ROOT = _HERE.parent                            # repo root
sys.path.insert(0, str(_HERE))                        # for embedder.py
sys.path.insert(0, str(PROJECT_ROOT))                 # for TRL-model imports

from embedder import build_embedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path.name}:{lineno} — skipping bad JSON: {exc}")
    return records


def extract_unique_tables(records: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Return (table_ids, unique_records) keeping the first occurrence per table_id.
    unique_records[i] is the representative dict for table_ids[i].
    """
    seen: dict[str, dict] = {}
    for rec in records:
        tid = rec["table_id"]
        if tid not in seen:
            seen[tid] = rec
    table_ids      = list(seen.keys())
    unique_records = [seen[t] for t in table_ids]
    return table_ids, unique_records


def compute_retrieval_metrics(
    ranked_indices: np.ndarray,
    gold_ids: list[str],
    table_ids: list[str],
    table_id_to_idx: dict[str, int],
    k_values: list[int],
) -> dict:
    """
    Compute MRR and Hit@k.

    Parameters
    ----------
    ranked_indices   : [Q, T] — each row is table indices sorted best→worst.
    gold_ids         : correct table_id per query.
    table_ids        : ordered list of all unique table_ids.
    table_id_to_idx  : table_id → column index in ranked_indices.
    k_values         : cut-offs for Hit@k.

    Returns
    -------
    dict with keys "MRR" and "Hit@k" for each k.
    """
    reciprocal_ranks: list[float] = []
    hits = {k: 0 for k in k_values}

    for q_idx, gold_tid in enumerate(gold_ids):
        gold_col = table_id_to_idx.get(gold_tid)
        if gold_col is None:
            reciprocal_ranks.append(0.0)
            continue
        positions = np.where(ranked_indices[q_idx] == gold_col)[0]
        rank      = int(positions[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in k_values:
            if rank <= k:
                hits[k] += 1

    n = len(gold_ids)
    return {
        "MRR":    float(np.mean(reciprocal_ranks)),
        **{f"Hit@{k}": hits[k] / n for k in k_values},
    }


def build_top_k_export(
    records: list[dict],
    ranked_indices: np.ndarray,
    sim_matrix: np.ndarray,
    gold_ids: list[str],
    table_ids: list[str],
    table_id_to_idx: dict[str, int],
    top_k: int,
) -> list[dict]:
    """Build the per-query top-k retrieval list for inspection."""
    export = []
    for q_idx, rec in enumerate(records):
        gold_tid   = gold_ids[q_idx]
        ranked_row = ranked_indices[q_idx]
        sims_row   = sim_matrix[q_idx]

        top_k_list = [
            {
                "rank":       rank + 1,
                "table_id":   table_ids[ranked_row[rank]],
                "similarity": float(sims_row[ranked_row[rank]]),
                "is_gold":    table_ids[ranked_row[rank]] == gold_tid,
            }
            for rank in range(min(top_k, len(ranked_row)))
        ]

        gold_col  = table_id_to_idx.get(gold_tid)
        gold_rank = (
            int(np.where(ranked_row == gold_col)[0][0]) + 1
            if gold_col is not None else None
        )

        export.append(
            {
                "query_id":       rec.get("id", str(q_idx)),
                "question":       rec["question"],
                "gold_table_id":  gold_tid,
                "gold_rank":      gold_rank,
                f"top{top_k}":    top_k_list,
            }
        )
    return export


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_entry(
    entry_idx: int,
    dataset_cfg: dict,
    retrieval_cfg: dict,
    output_dir: Path,
    project_root: Path,
) -> dict:
    """
    Evaluate one dataset/model entry.  Returns the metrics dict (also saved to disk).
    """
    ds_name    = dataset_cfg["name"]
    ds_path    = project_root / dataset_cfg["path"]
    emb_cfg    = dataset_cfg["embedder"]
    k_values      = retrieval_cfg.get("k_values", [1, 3, 5, 10, 20])
    top_k_exp     = retrieval_cfg.get("top_k_export", 5)
    tsr_top_k     = retrieval_cfg.get("tsr_top_k",     2000)
    tsr_mrr_depth = retrieval_cfg.get("tsr_mrr_depth", 2000)

    # ── Build embedder ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Entry {entry_idx}  |  dataset: {ds_name}")
    print(f"Embedder type: {emb_cfg['type']}")
    embedder = build_embedder(emb_cfg, project_root=project_root)
    run_label = embedder.label
    print(f"Label: {run_label}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {ds_path} …")
    records  = load_jsonl(ds_path)
    gold_ids = [rec["table_id"] for rec in records]
    queries  = [rec["question"] for rec in records]
    print(f"  {len(records):,} records loaded")

    # ── Unique tables ─────────────────────────────────────────────────────────
    table_ids, unique_records = extract_unique_tables(records)
    table_id_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    print(f"  {len(table_ids):,} unique tables")

    # ── Embed tables (document side) ─────────────────────────────────────────
    print(f"Embedding {len(unique_records):,} table documents …")
    t0 = time.perf_counter()
    has_variants = hasattr(embedder, "encode_table_corpus_variants")
    if has_variants:
        table_variants = embedder.encode_table_corpus_variants(unique_records)
        table_embeddings = table_variants["both"]   # used for top-k export
    else:
        table_variants   = None
        table_embeddings = embedder.encode_table_corpus(unique_records)    # [T, D]
    t_tables = time.perf_counter() - t0
    print(f"  done in {t_tables:.1f}s  shape={table_embeddings.shape}")

    # ── Embed queries (query side) ────────────────────────────────────────────
    print(f"Embedding {len(queries):,} queries …")
    t0 = time.perf_counter()
    query_embeddings = embedder.encode_queries(queries)          # [Q, D]
    t_queries = time.perf_counter() - t0
    print(f"  done in {t_queries:.1f}s  shape={query_embeddings.shape}")

    # ── Rank-and-metrics helper ───────────────────────────────────────────────
    def _rank_and_metrics(tbl_embs: np.ndarray):
        """Return (metrics_dict, sim_matrix, ranked_indices) for one table embedding."""
        try:
            import torch as _torch
            _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            if _dev.type == "cuda":
                _q   = _torch.from_numpy(query_embeddings).to(_dev)
                _t   = _torch.from_numpy(tbl_embs).to(_dev)
                _sim = (_q @ _t.T).cpu().numpy()
            else:
                _sim = query_embeddings @ tbl_embs.T
        except ImportError:
            _sim = query_embeddings @ tbl_embs.T
        _ranked = np.argsort(-_sim, axis=1)
        _m = compute_retrieval_metrics(
            ranked_indices  = _ranked,
            gold_ids        = gold_ids,
            table_ids       = table_ids,
            table_id_to_idx = table_id_to_idx,
            k_values        = k_values,
        )
        return _m, _sim, _ranked

    # ── Rank tables per query ─────────────────────────────────────────────────
    if has_variants:
        variant_metrics: dict = {}
        sim_matrix = ranked_indices = None
        for vname, vembs in table_variants.items():
            vm, vsim, vranked = _rank_and_metrics(vembs)
            variant_metrics[vname] = vm
            if vname == "both":
                sim_matrix, ranked_indices = vsim, vranked
        if sim_matrix is None:  # "both" absent — fall back to first variant
            sim_matrix, ranked_indices = vsim, vranked

        metrics = variant_metrics["both"]
        print(f"\n  {'Variant':<10}  {'MRR':>8}" +
              "".join(f"  Hit@{k}" for k in k_values))
        print(f"  {'-'*10}  {'-'*8}" + "".join(f"  {'-----'}" for _ in k_values))
        for vname, vm in variant_metrics.items():
            row = f"  {vname:<10}  {vm['MRR']:>8.4f}"
            for k in k_values:
                row += f"  {vm[f'Hit@{k}']:.4f}"
            print(row)
    else:
        variant_metrics = None
        metrics, sim_matrix, ranked_indices = _rank_and_metrics(table_embeddings)
        print(f"\n  MRR    : {metrics['MRR']:.4f}")
        for k in k_values:
            print(f"  Hit@{k:<3}: {metrics[f'Hit@{k}']:.4f}")

    # ── TSR metrics (UTUEL node-level global index) ───────────────────────────
    tsr_variant_metrics: dict | None = None
    if hasattr(embedder, "compute_tsr_metrics"):
        print("\n  Computing TSR (Top-Score-Rank) metrics …")
        tsr_variant_metrics = embedder.compute_tsr_metrics(
            query_embeddings = query_embeddings,
            gold_ids         = gold_ids,
            k_values         = k_values,
            top_k_table      = tsr_top_k,
            mrr_depth        = tsr_mrr_depth,
        )
        _tsr_search_space = {
            "tsr":     "node_a+node_b (unique by max cosine score)",
            "col_tsr": "col_mean(node_a) (unique by max cosine score)",
            "row_tsr": "row_mean(node_a) (unique by max cosine score)",
            "tbl_tsr": "table_mean(node_a) (unique by max cosine score)",
        }
        print(f"\n  {'TSR space':<12}  {'MRR':>8}  " + "  ".join(f"Hit@{k}" for k in k_values))
        print(f"  {'-'*12}  {'-'*8}  " + "  ".join("-"*6 for _ in k_values))
        for sp, vm in tsr_variant_metrics.items():
            row = f"  {sp:<12}  {vm['MRR']:>8.4f}  " + "  ".join(f"{vm[f'Hit@{k}']:.4f}" for k in k_values)
            print(row)

    # ── Persist results ───────────────────────────────────────────────────────
    # Sanitise label for use as a filename
    safe_label = run_label.replace("/", "_").replace(":", "_").replace("#", "_")
    result_path = output_dir / f"{safe_label}.json"
    top_k_path  = output_dir / f"{safe_label}_top{top_k_exp}.json"

    result = {
        "entry_idx":      entry_idx,
        "dataset":        ds_name,
        "dataset_path":   str(ds_path),
        "embedder_label": run_label,
        "embedder_cfg":   emb_cfg,
        "n_queries":      len(records),
        "n_tables":       len(table_ids),
        "embed_time_tables_s":  round(t_tables, 2),
        "embed_time_queries_s": round(t_queries, 2),
        "metrics":        metrics,
        **({"variant_metrics": variant_metrics} if variant_metrics else {}),
        **({"tsr_variant_metrics": tsr_variant_metrics} if tsr_variant_metrics else {}),
    }

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved → {result_path}")

    top_k_data = build_top_k_export(
        records, ranked_indices, sim_matrix,
        gold_ids, table_ids, table_id_to_idx, top_k_exp,
    )
    with top_k_path.open("w", encoding="utf-8") as f:
        json.dump(top_k_data, f, ensure_ascii=False, indent=2)
    print(f"  Top-{top_k_exp} export saved → {top_k_path}")

    return result


def main(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    retrieval_cfg = cfg.get("retrieval", {})
    output_dir    = PROJECT_ROOT / cfg.get("output_dir", "table_retrieval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve base directories and expand relative embedder paths ───────────
    checkpoints_dir = cfg.get("checkpoints_dir")
    cache_dir       = cfg.get("cache_dir")
    datasets: list[dict] = cfg.get("datasets", [])
    for ds in datasets:
        emb = ds.get("embedder", {})
        if checkpoints_dir and "checkpoint_dir" in emb:
            p = Path(emb["checkpoint_dir"])
            if not p.is_absolute():
                emb["checkpoint_dir"] = str(Path(checkpoints_dir) / p)
        if cache_dir and "cache_file" in emb:
            p = Path(emb["cache_file"])
            if not p.is_absolute():
                emb["cache_file"] = str(Path(cache_dir) / p)

    all_results: list[dict] = []

    print(f"Table Retrieval Evaluation")
    print(f"Config : {config_path}")
    print(f"Entries: {len(datasets)}")
    print(f"Output : {output_dir}")

    for idx, dataset_cfg in enumerate(datasets):
        try:
            result = run_entry(
                entry_idx    = idx,
                dataset_cfg  = dataset_cfg,
                retrieval_cfg = retrieval_cfg,
                output_dir   = output_dir,
                project_root = PROJECT_ROOT,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] Entry {idx} failed: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "entry_idx":      idx,
                "embedder_cfg":   dataset_cfg.get("embedder", {}),
                "error":          str(exc),
            })

    # ── Save combined manifest ─────────────────────────────────────────────────
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved → {manifest_path}")
    print("Run `python table_retrieval/report.py` to compile the summary report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table retrieval evaluation")
    parser.add_argument(
        "--config",
        default=str(_HERE / "config.yaml"),
        help="Path to config.yaml (default: table_retrieval/config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)

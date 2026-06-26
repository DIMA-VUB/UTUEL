#!/usr/bin/env python3
"""
warm_query_cache.py
───────────────────
Pre-embed all dataset inference queries into the on-disk node cache files
declared in config_embed_asses.yaml, so that Cell 8 of retrieve_eval.ipynb
can serve query embeddings without any live model connection.

One Ollama /api/embed request (or SentenceTransformer batch) is issued per
unique base model — never one request per query.

Usage
─────
    python table_retrieval/warm_query_cache.py
    python table_retrieval/warm_query_cache.py --config table_retrieval/config_embed_asses.yaml
    python table_retrieval/warm_query_cache.py --ollama-url http://10.0.0.5:11434
    python table_retrieval/warm_query_cache.py --dry-run      # show plan, embed nothing
    python table_retrieval/warm_query_cache.py --ollama-only  # skip HuggingFace/ST models

Exit codes: 0 = success, 1 = one or more models failed (others still saved).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pt_cache(cache_path: Path) -> tuple[dict, Optional[torch.Tensor]]:
    if not cache_path.exists():
        return {}, None
    ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
    return ckpt.get("text_to_idx", {}), ckpt.get("embed_cache")


def save_pt_cache(cache_path: Path, t2i: dict, embs: torch.Tensor) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"embed_cache": embs, "text_to_idx": t2i, "embed_dim": int(embs.shape[1])},
        cache_path,
    )
    print(f"  saved  {cache_path.name}  ({len(t2i):,} texts, dim={embs.shape[1]})")


# ── Embedding backends ────────────────────────────────────────────────────────

def embed_ollama(texts: list[str], model: str, base_url: str, batch_size: int) -> np.ndarray:
    """POST to Ollama /api/embed in batches.  Returns [N, D] float32."""
    import requests  # type: ignore[import-untyped]

    url = f"{base_url.rstrip('/')}/api/embed"
    vecs: list[list[float]] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": model, "input": batch}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        batch_vecs = data.get("embeddings") or data.get("embedding")
        if batch_vecs is None:
            raise ValueError(f"Ollama response missing 'embeddings' key: {list(data.keys())}")
        vecs.extend(batch_vecs)
        done = min(i + batch_size, n)
        print(f"  [{model}]  {done:,}/{n:,} embedded", end="\r", flush=True)
    print(f"  [{model}]  {n:,}/{n:,} embedded — done           ")
    return np.array(vecs, dtype=np.float32)


def embed_st(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    """Embed with SentenceTransformer.  Returns [N, D] float32."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    print(f"  [ST/{model}]  loading …")
    st = SentenceTransformer(model)
    return st.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    ).astype(np.float32)


# ── Core logic ────────────────────────────────────────────────────────────────

def merge_into_cache(
    cache_path: Path,
    missing_texts: list[str],
    new_embs: np.ndarray,
    existing_t2i: dict,
    existing_embs: Optional[torch.Tensor],
) -> None:
    new_tensor = torch.tensor(new_embs, dtype=torch.float32)
    if existing_embs is not None:
        start = len(existing_t2i)
        merged_t2i = dict(existing_t2i)
        for i, text in enumerate(missing_texts):
            merged_t2i[text] = start + i
        merged_embs = torch.cat([existing_embs, new_tensor], dim=0)
    else:
        merged_t2i = {text: i for i, text in enumerate(missing_texts)}
        merged_embs = new_tensor
    save_pt_cache(cache_path, merged_t2i, merged_embs)


def resolve_run_cfg(ckpt_path: Path) -> Optional[dict]:
    """Load run_config.yaml from checkpoint dir or its parent."""
    for candidate in (ckpt_path / "run_config.yaml", ckpt_path.parent / "run_config.yaml"):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh)
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-embed inference queries into UTUEL node-cache .pt files."
    )
    parser.add_argument(
        "--config",
        default="table_retrieval/config_embed_asses.yaml",
        help="Path to config_embed_asses.yaml (default: table_retrieval/config_embed_asses.yaml)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Ollama/ST batch size override (0 = use value from config, default 64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without embedding anything",
    )
    parser.add_argument(
        "--ollama-only",
        action="store_true",
        help="Skip models that use HuggingFace/SentenceTransformer as their base",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[error] config not found: {config_path}", file=sys.stderr)
        return 1

    # Determine project root: parent of table_retrieval/ directory.
    project_root = config_path.parent.parent

    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    checkpoints_root = Path(cfg["checkpoints_dir"])
    if not checkpoints_root.is_absolute():
        checkpoints_root = (project_root / checkpoints_root).resolve()

    cache_root = Path(cfg["cache_dir"])
    if not cache_root.is_absolute():
        cache_root = (project_root / cache_root).resolve()

    all_entries = cfg.get("datasets") or cfg.get("embeddings") or cfg.get("models") or []

    # ── Collect unique (cache_file, checkpoint_dir, batch_size) per base model ─
    seen_cache_names: dict[str, dict] = {}  # cache filename → info dict
    for entry in all_entries:
        emb_cfg = entry.get("embedder", entry)
        model_name_val = str(emb_cfg.get("model_name", ""))
        is_utuel = (
            emb_cfg.get("type") == "custom"
            or emb_cfg.get("class") == "UTUELTableEmbedder"
            or model_name_val.startswith("UTUEL/")
        )
        if not is_utuel:
            continue

        cache_raw = str(emb_cfg.get("cache_file", "")).strip()
        ckpt_raw  = str(emb_cfg.get("checkpoint_dir", "")).strip()
        if not cache_raw or not ckpt_raw:
            continue

        cache_abs = Path(cache_raw) if Path(cache_raw).is_absolute() else cache_root / cache_raw
        ckpt_abs  = Path(ckpt_raw)  if Path(ckpt_raw).is_absolute()  else checkpoints_root / ckpt_raw

        key = cache_abs.name  # shared across variants of the same base model
        if key not in seen_cache_names:
            seen_cache_names[key] = {
                "cache_file":     cache_abs,
                "checkpoint_dir": ckpt_abs,
                "batch_size":     int(emb_cfg.get("batch_size", 64)),
            }

    if not seen_cache_names:
        print("[error] No UTUEL entries found in config.", file=sys.stderr)
        return 1

    # ── Load unique queries ───────────────────────────────────────────────────
    dataset_paths = list(dict.fromkeys(
        str(e.get("path", "")).strip()
        for e in all_entries
        if str(e.get("path", "")).strip()
    ))
    if not dataset_paths:
        print("[error] No dataset path in config.", file=sys.stderr)
        return 1

    dataset_file = Path(dataset_paths[0])
    if not dataset_file.is_absolute():
        dataset_file = (project_root / dataset_file).resolve()
    if not dataset_file.exists():
        print(f"[error] Dataset not found: {dataset_file}", file=sys.stderr)
        return 1

    records = load_jsonl(dataset_file)
    queries: list[str] = list(dict.fromkeys(
        str(r.get("question", "")).strip()
        for r in records
        if str(r.get("question", "")).strip()
    ))
    print(f"Dataset : {dataset_file.name}")
    print(f"Queries : {len(queries):,} unique")
    print(f"Models  : {len(seen_cache_names)} unique cache files\n")

    # ── Process each cache file ───────────────────────────────────────────────
    n_ok = 0
    n_fail = 0

    for cache_name, info in seen_cache_names.items():
        cache_path: Path = info["cache_file"]
        ckpt_path:  Path = info["checkpoint_dir"]
        batch_size: int  = args.batch_size if args.batch_size > 0 else info["batch_size"]

        print(f"{'─' * 60}")
        print(f"cache  : {cache_name}")
        print(f"ckpt   : {ckpt_path}")

        # Resolve base model type from checkpoint's run_config.yaml
        run_cfg = resolve_run_cfg(ckpt_path)
        if run_cfg is None:
            print(f"  [warn] run_config.yaml not found in {ckpt_path} or parent; skipping")
            n_fail += 1
            continue

        emb_run_cfg = run_cfg.get("embedder", {})
        model_type  = emb_run_cfg.get("model_type", "huggingface")
        base_model  = emb_run_cfg.get("model_name", "")
        print(f"type   : {model_type}  ({base_model})")

        if args.ollama_only and model_type != "ollama":
            print(f"  skipped — not an Ollama model (--ollama-only is set)")
            continue

        # Check which queries are already in the cache
        existing_t2i, existing_embs = load_pt_cache(cache_path)
        missing = [q for q in queries if q not in existing_t2i]
        if not missing:
            print(f"  all {len(queries):,} queries already cached — nothing to do")
            n_ok += 1
            continue

        print(f"  need   : {len(missing):,} / {len(queries):,} queries")

        if args.dry_run:
            print(f"  [dry-run] would embed {len(missing):,} texts via {model_type}")
            n_ok += 1
            continue

        # Embed missing queries
        try:
            if model_type == "ollama":
                new_embs = embed_ollama(missing, base_model, args.ollama_url, batch_size)
            else:
                new_embs = embed_st(missing, base_model, batch_size)
        except Exception as exc:
            print(f"  [FAIL] embedding error: {exc}")
            n_fail += 1
            continue

        # Merge and save
        try:
            merge_into_cache(cache_path, missing, new_embs, existing_t2i, existing_embs)
            n_ok += 1
        except Exception as exc:
            print(f"  [FAIL] save error: {exc}")
            n_fail += 1

    print(f"\n{'=' * 60}")
    print(f"Done.  {n_ok} succeeded,  {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

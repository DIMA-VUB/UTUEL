"""
dataset.py
PyTorch Datasets for Column Type Annotation (CTA).

Dataset format  (.table_col_type — JSON array)
──────────────────────────────────────────────
Each record is a JSON array with the following structure (DIMA-VUB/UTUEL format):

  record[0]  — table_id (string, e.g. "27289759-6")
  record[1]  — table_caption / page title (string)
  record[2]  — numeric table id (int)
  record[3]  — split tag (string, ignored)
  record[4]  — extra annotation (string, may be empty)
  record[5]  — column headers  (list[str])
  record[6]  — cell → entity links  (list of per-column link lists)
               Each inner item: [[[row, col], [entity_id, entity_label]], ...]
  record[7]  — column type labels   (list of per-column type-list)
               Each inner item:    list[str]  (Freebase type strings)

Two dataset classes are provided:

CTASMPDataset   (pretraining stage)
    Reconstructs each CTA table into a (header, rows) grid and generates
    U-path samples via TRL-model's SMP machinery.  Items are identical in
    format to TableEmbedJePADataset — they feed directly into TableEmbedJePA
    training (JEPA + InfoNCE losses).

CTADataset      (fine-tuning stage)
    For each (table, column) pair: collects the cell-text embeddings that
    were precomputed during pretraining (or fresh from the embedder) and
    stores the Freebase type label from type_vocab.

Label encoding
──────────────
The type_vocab file contains one Freebase type string per line.
The integer label for a column is the index of its first type string in
type_vocab.  Columns whose type is not in type_vocab are skipped.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .dataset_utils import get_embedder, load_type_vocab, resolve_data_paths
except ImportError:
    from dataset_utils import get_embedder, load_type_vocab, resolve_data_paths

# TRL-model SMP generation — import from the sibling package
_HERE = Path(__file__).parent
_TRL  = _HERE.parent / "TRL-model"
if str(_TRL) not in sys.path:
    sys.path.insert(0, str(_TRL))

_TRR = _HERE.parent / "table_retrieval"
if str(_TRR) not in sys.path:
    sys.path.insert(0, str(_TRR))

from smp import UPath, generate_u_paths_flat, generate_u_paths_from_graph  # noqa: E402
from embedder import OllamaEmbedder  # type: ignore[import]  # noqa: E402


# ── Dataset ───────────────────────────────────────────────────────────────────

class CTADataset(Dataset):
    """
    One sample = one (table, column) pair → (column_embedding [n_cells, d], label).

    Attributes
    ----------
    items       : list of dicts with keys
                    col_texts   list[str]   — cell texts for this column
                    label       int         — class index in type_vocab
                    table_id    str
                    col_idx     int
                    col_header  str
    embed_cache : dict[str, torch.Tensor]   — text → [d] embedding (CPU)
    """

    def __init__(
        self,
        data_path: str | Path,
        type_vocab_path: str | Path,
        model_type: str = "huggingface",
        base_url: Optional[str] = None,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_cells_per_col: Optional[int] = None,
        use_node_a: bool = True,
        use_node_b: bool = True,
        precompute: bool = True,
        embed_batch_size: int = 64,
        cache_embeddings: bool = True,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        expected_embed_dim: Optional[int] = None,  # if set, cached dim is validated
    ) -> None:
        self.type2idx, self.idx2type = load_type_vocab(type_vocab_path)
        self.num_classes = len(self.type2idx)

        self.items: list[dict] = []
        data_path = Path(data_path)
        records = json.loads(data_path.read_text(encoding="utf-8"))

        for rec in records:
            self._parse_record(rec, use_node_a, use_node_b, max_cells_per_col, max_rows)

        print(
            f"[CTA][dataset] {len(self.items)} (table, col) samples "
            f"from {data_path.name}  ({self.num_classes} classes)"
        )

        # ── Embed all unique cell texts ───────────────────────────────────────
        all_texts: list[str] = []
        for item in self.items:
            all_texts.extend(item["col_texts"])
        unique_texts = list(dict.fromkeys(all_texts))  # preserve order, deduplicate

        self._cache_path: Optional[Path] = None
        if cache_embeddings:
            # Each model gets its own subdirectory so caches never collide.
            # Subdirectory name = last component of model_name, same sanitisation
            # as the run slug used by pretrain.py/finetune.py:
            #   sentence-transformers/all-MiniLM-L6-v2  →  all-MiniLM-L6-v2
            #   qwen3-embedding:0.6b                    →  qwen3-embedding#0.6b
            _model_slug = (model_name or model_type).split("/")[-1] \
                            .replace(":", "#").replace(" ", "_")
            base_cache_dir = Path(embed_cache_dir) if embed_cache_dir else data_path.parent
            cache_dir = base_cache_dir / _model_slug
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / f"{data_path.stem}.embed_cache.pt"
            print(f"[CTA][embed] cache path: {self._cache_path}")

        self.embed_cache: dict[str, torch.Tensor] = {}
        if self._cache_path is not None and self._cache_path.exists():
            _loaded = torch.load(self._cache_path, weights_only=True)
            # Validate embedding dimension against what the config requests.
            # If there is a mismatch the cache was produced by a different model
            # and must be regenerated rather than silently used.
            _cached_dim = next(iter(_loaded.values())).shape[0] if _loaded else None
            if expected_embed_dim is not None and _cached_dim is not None \
                    and _cached_dim != expected_embed_dim:
                print(
                    f"[CTA][embed] WARNING: cached dim={_cached_dim} "
                    f"!= expected dim={expected_embed_dim}. "
                    f"Deleting stale cache and regenerating: {self._cache_path}"
                )
                self._cache_path.unlink(missing_ok=True)
                missing = unique_texts
            else:
                self.embed_cache = _loaded
                print(f"[CTA][embed] loaded cache from {self._cache_path} (dim={_cached_dim})")
                missing = [t for t in unique_texts if t not in self.embed_cache]
        else:
            missing = unique_texts

        self._embedder = None
        if missing and precompute:
            if model_type.strip().lower() == "ollama":
                self._embedder = OllamaEmbedder(
                    base_url=base_url or "http://localhost:11434/",
                    model_name=model_name or "",
                    batch_size=embed_batch_size,
                )
            else:
                self._embedder = get_embedder(
                    model_type=model_type,
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                )
            print(f"[CTA][embed] embedding {len(missing)} unique texts …")
            if hasattr(self._embedder, "encode_documents"):
                vecs = self._embedder.encode_documents(missing)
            else:
                vecs = self._embedder.embed_documents(missing)
            for txt, vec in zip(missing, vecs):
                t = torch.tensor(vec, dtype=torch.float32)
                if expected_embed_dim is not None and t.shape[0] > expected_embed_dim:
                    t = t[:expected_embed_dim]
                self.embed_cache[txt] = t
            if self._cache_path is not None:
                # Log actual stored dim so mismatches are visible
                _stored_dim = next(iter(self.embed_cache.values())).shape[0]
                torch.save(self.embed_cache, self._cache_path)
                print(f"[CTA][embed] saved cache → {self._cache_path}  (dim={_stored_dim})")
        elif not missing and precompute:
            if model_type.strip().lower() == "ollama":
                self._embedder = OllamaEmbedder(
                    base_url=base_url or "http://localhost:11434/",
                    model_name=model_name or "",
                    batch_size=embed_batch_size,
                )
            else:
                self._embedder = get_embedder(
                    model_type=model_type,
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                )

        # Detect embed_dim from cache
        if self.embed_cache:
            self.embed_dim: int = next(iter(self.embed_cache.values())).shape[0]
        else:
            self.embed_dim = 384

    def _parse_record(
        self,
        rec: list,
        use_node_a: bool,
        use_node_b: bool,
        max_cells: Optional[int],
        max_rows: Optional[int],
    ) -> None:
        """
        Parse one .table_col_type record and append (table, col) items.

        Record schema:
          rec[0]  table_id str
          rec[1]  caption  str
          rec[2]  numeric table id
          rec[3]  split tag
          rec[4]  extra annotation
          rec[5]  headers          list[str]
          rec[6]  cell-entity links list-of-col-partitions
                  Each partition is a list of [[row, col], [entity_id, label]] pairs.
          rec[7]  column types     list-of-col-type-lists  (list[list[str]])
        """
        table_id: str = str(rec[0])
        headers: list[str] = rec[5] if len(rec) > 5 else []
        cell_links: list = rec[6] if len(rec) > 6 else []
        col_types: list = rec[7] if len(rec) > 7 else []

        # Organise cell links by column index
        # Each col_partition is a list of [[row, col], [entity_id, entity_label]]
        col_to_cells: dict[int, list[str]] = {}
        for col_partition in cell_links:
            for entry in col_partition:
                (row_idx, col_idx), (entity_id, entity_label) = entry
                if max_rows is not None and row_idx >= max_rows:
                    continue
                col_to_cells.setdefault(col_idx, []).append(str(entity_label))

        # Process type labels per column (rec[7] is a list of per-column type lists)
        for col_idx, types_for_col in enumerate(col_types):
            if not types_for_col:
                continue
            # Use the first type as the primary label (most specific / first listed)
            primary_type = types_for_col[0]
            if primary_type not in self.type2idx:
                continue
            label = self.type2idx[primary_type]

            texts = col_to_cells.get(col_idx, [])
            if max_cells is not None:
                texts = texts[:max_cells]

            self.items.append(
                {
                    "col_texts": texts,
                    "label": label,
                    "table_id": table_id,
                    "col_idx": col_idx,
                    "col_header": headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}",
                    "all_types": types_for_col,
                }
            )

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        texts = item["col_texts"]

        # Retrieve or compute embeddings; on cache miss embed and persist
        vecs = []
        new_entries = False
        for txt in texts:
            if txt in self.embed_cache:
                vecs.append(self.embed_cache[txt])
            else:
                if self._embedder is not None:
                    vec = self._embedder.embed_documents([txt])[0]
                    tensor = torch.tensor(vec, dtype=torch.float32)
                else:
                    tensor = torch.zeros(self.embed_dim, dtype=torch.float32)
                self.embed_cache[txt] = tensor
                new_entries = True
                vecs.append(tensor)
        if new_entries and self._cache_path is not None:
            torch.save(self.embed_cache, self._cache_path)

        # Stack cell embeddings: [n_cells, d]
        cell_embeds = torch.stack(vecs)  # [n_cells, d]

        return {
            "cell_embeds": cell_embeds,          # [n_cells, d]
            "label": torch.tensor(item["label"], dtype=torch.long),
            "table_id": item["table_id"],
            "col_idx": item["col_idx"],
            "col_header": item["col_header"],
        }


# ── Collate function ──────────────────────────────────────────────────────────

def cta_collate_fn(batch: list[dict]) -> dict:
    """
    Collate variable-length cell_embeds by padding to the longest sequence
    in the batch.  Returns:
        cell_embeds  [B, max_n, d]   — padded cell embeddings
        attn_mask    [B, max_n]      — 1 for real cells, 0 for padding
        labels       [B]             — class indices
        table_ids    list[str]
        col_idxs     list[int]
        col_headers  list[str]
    """
    max_n = max(x["cell_embeds"].shape[0] for x in batch)
    d = batch[0]["cell_embeds"].shape[1]

    padded = torch.zeros(len(batch), max_n, d)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    labels = torch.stack([x["label"] for x in batch])

    for i, x in enumerate(batch):
        n = x["cell_embeds"].shape[0]
        padded[i, :n] = x["cell_embeds"]
        mask[i, :n] = True

    return {
        "cell_embeds": padded,          # [B, max_n, d]
        "attn_mask": mask,              # [B, max_n]
        "labels": labels,               # [B]
        "table_ids": [x["table_id"] for x in batch],
        "col_idxs": [x["col_idx"] for x in batch],
        "col_headers": [x["col_header"] for x in batch],
    }


# ── DataModule ────────────────────────────────────────────────────────────────

class CTADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CTA — wraps train / dev / test splits.

    Passed directly to pl.Trainer; handles dataset creation, caching and
    DataLoader construction.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    # shared kwargs forwarded to every CTADataset
    def _dataset_kwargs(self, data_path: str, split_tag: str) -> dict:
        cfg = self.cfg
        return dict(
            data_path=data_path,
            type_vocab_path=self._paths["type_vocab"],
            model_type=cfg.embedder.model_type,
            base_url=cfg.embedder.get("base_url"),
            model_name=cfg.embedder.get("model_name"),
            api_key=cfg.embedder.get("api_key"),
            max_rows=cfg.data.get(f"max_rows_{split_tag}"),
            max_cells_per_col=cfg.data.get("max_cells_per_col"),
            use_node_a=cfg.data.use_node_a,
            use_node_b=cfg.data.use_node_b,
            precompute=cfg.embedder.precompute,
            embed_batch_size=cfg.embedder.embed_batch_size,
            cache_embeddings=cfg.embedder.cache_embeddings,
            embed_cache_dir=cfg.embedder.get("embed_cache_dir"),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        paths = self._paths
        if stage in (None, "fit"):
            self.train_ds = CTADataset(**self._dataset_kwargs(paths["train"], "train"))
            self.val_ds   = CTADataset(**self._dataset_kwargs(paths["dev"],   "dev"))
            self.embed_dim = self.train_ds.embed_dim
            self.num_classes = self.train_ds.num_classes
        if stage in (None, "test", "predict"):
            self.test_ds = CTADataset(**self._dataset_kwargs(paths["test"], "test"))
            if not hasattr(self, "embed_dim"):
                self.embed_dim = self.test_ds.embed_dim
                self.num_classes = self.test_ds.num_classes

    def train_dataloader(self) -> DataLoader:
        cfg = self.cfg
        stage_cfg = cfg.finetuning if hasattr(cfg, "finetuning") else cfg.pretraining
        nw = stage_cfg.dataloader_num_workers
        return DataLoader(
            self.train_ds,
            batch_size=stage_cfg.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=cta_collate_fn,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.cfg
        stage_cfg = cfg.finetuning if hasattr(cfg, "finetuning") else cfg.pretraining
        nw = stage_cfg.dataloader_num_workers
        return DataLoader(
            self.val_ds,
            batch_size=stage_cfg.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=cta_collate_fn,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

    def test_dataloader(self) -> DataLoader:
        cfg = self.cfg
        stage_cfg = cfg.finetuning if hasattr(cfg, "finetuning") else cfg.pretraining
        nw = stage_cfg.dataloader_num_workers
        return DataLoader(
            self.test_ds,
            batch_size=stage_cfg.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=cta_collate_fn,
            pin_memory=True,
            persistent_workers=nw > 0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CTASMPDataset — Pretraining stage
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_table(rec: list, max_rows: Optional[int] = None):
    """
    Reconstruct (header, rows) from a .table_col_type record.

    record[5] — column headers  list[str]
    record[6] — cell–entity links (per-column partitions)
                Each entry: [[row_idx, col_idx], [entity_id, entity_label]]

    Returns
    -------
    table_id : str
    header   : list[str]   column headers
    rows     : list[list[str]]   rows[row_idx][col_idx] = entity_label
    """
    table_id: str = str(rec[0])
    header: list[str] = rec[5] if len(rec) > 5 else []
    cell_links: list = rec[6] if len(rec) > 6 else []

    # Build a sparse cell grid: (row_idx, col_idx) → entity_label
    grid: dict[tuple[int, int], str] = {}
    for col_partition in cell_links:
        for entry in col_partition:
            (row_idx, col_idx), (_, entity_label) = entry
            grid[(row_idx, col_idx)] = str(entity_label)

    if not grid:
        return table_id, header, []

    max_row = max(r for r, _ in grid) + 1
    n_cols  = len(header)
    if max_rows is not None:
        max_row = min(max_row, max_rows)

    rows: list[list[str]] = []
    for r in range(max_row):
        row = [grid.get((r, c), "") for c in range(n_cols)]
        rows.append(row)

    return table_id, header, rows


class CTASMPDataset(Dataset):
    """
    Pretraining dataset for CTA — generates U-path (SMP) samples from
    .table_col_type tables using TRL-model's SMP machinery.

    Each item has the same format as TableEmbedJePADataset.__getitem__:
        smp_embeds       [4, d]  — [pivot_a, node_a, node_b, pivot_b]
        smp_bar_embeds   [4, d]  — reversed: [pivot_b, node_b, node_a, pivot_a]
        query_embeds     [1, d]  — concat query embedding (SMP direction)
        query_bar_embeds [1, d]  — concat query embedding (SMP_bar direction)
        question_emb     [d]     — zero vector (CTA has no retrieval question)
        record_idx       int     — index into self.records

    These items are collated by jepa_collate_fn from TRL-model and fed into
    the standard TableEmbedJePA training loop unchanged.
    """

    def __init__(
        self,
        data_path: str | Path,
        model_type: str = "huggingface",
        base_url: Optional[str] = None,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        max_records: Optional[int] = None,
        max_rows_per_table: Optional[int] = None,
        use_graph_walks: bool = False,
        num_walks: int = 50,
        chunk_size: int = 1,
        precompute: bool = True,
        embed_batch_size: int = 64,
        cache_embeddings: bool = True,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        cat_qry_template: str = "{pivot_b} ... {pivot_a}({node_a})?",
        cat_qry_bar_template: str = "{pivot_a} ... {pivot_b}({node_b})?",
        expected_embed_dim: Optional[int] = None,
    ) -> None:
        data_path = Path(data_path)
        self._model_type   = model_type
        self._model_name   = model_name
        self._base_url     = base_url
        self._api_key      = api_key
        self._cat_qry_tmpl     = cat_qry_template
        self._cat_qry_bar_tmpl = cat_qry_bar_template
        _tag = model_type.strip().lower()
        # Use the table_retrieval OllamaEmbedder directly so CTA reuses the
        # same class and its document/query methods without reimplementation.
        if _tag == "ollama":
            self._embedder = OllamaEmbedder(
                base_url=base_url or "http://localhost:11434/",
                model_name=model_name or "",
                batch_size=embed_batch_size,
            )
        else:
            # Preserve compatibility for backends not covered by table_retrieval.
            self._embedder = get_embedder(
                model_type=model_type,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
            )

        # ── Load and reconstruct tables ───────────────────────────────────────
        raw_records: list = json.loads(data_path.read_text(encoding="utf-8"))
        if max_records is not None:
            raw_records = raw_records[:max_records]

        self.records: list[dict] = []  # {"table_id", "header", "rows"}
        self._samples: list[tuple[int, UPath]] = []  # (record_idx, upath)

        for rec in raw_records:
            table_id, header, rows = _reconstruct_table(rec, max_rows_per_table)
            if not header or not rows:
                continue  # need at least 1 column and 1 row

            rec_idx = len(self.records)
            self.records.append({"table_id": table_id, "header": header, "rows": rows})

            # Generate U-paths using TRL-model SMP generation
            if use_graph_walks:
                upaths = generate_u_paths_from_graph(
                    header, rows,
                    num_walks=num_walks,
                    chunk_size=chunk_size,
                )
            else:
                upaths = generate_u_paths_flat(header, rows)

            for up in upaths:
                up.table_id  = table_id
                up.record_id = table_id
                self._samples.append((rec_idx, up))

        self._embed_dim: Optional[int] = None
        print(
            f"[CTA][SMP] {len(self.records)} tables → "
            f"{len(self._samples)} U-paths from {data_path.name}"
        )

        # ── Precompute embeddings ─────────────────────────────────────────────
        self._embed_cache: Optional[torch.Tensor] = None
        self._text_to_idx: Optional[dict]         = None
        self._smp_idx:         Optional[torch.Tensor] = None  # [N, 4]
        self._qry_cat_idx:     Optional[torch.Tensor] = None  # [N]
        self._qry_bar_cat_idx: Optional[torch.Tensor] = None  # [N]

        if precompute:
            self._precompute(
                cache_embeddings,
                embed_cache_dir,
                data_path,
                expected_embed_dim,
            )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _precompute(
        self,
        cache_embeddings: bool,
        embed_cache_dir: Optional[Union[str, Path]],
        data_path: Path,
        expected_embed_dim: Optional[int] = None,
    ) -> None:
        slug = (
            f"{data_path.stem}_"
            f"{(self._model_name or self._model_type).replace('/', '-')}"
        )
        cache_dir = Path(embed_cache_dir) if embed_cache_dir else data_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{slug}_smp.embed_cache.pt"
        print(
            f"[CTA][SMP][cache] file={cache_file} "
            f"enabled={cache_embeddings} samples={len(self._samples)}",
            flush=True,
        )

        if cache_embeddings and cache_file.exists():
            ckpt = torch.load(cache_file, map_location="cpu")
            self._doc_embed_cache = ckpt.get("doc_embed_cache", ckpt.get("embed_cache"))
            self._qry_embed_cache = ckpt.get("qry_embed_cache")
            self._doc_text_to_idx = ckpt.get("doc_text_to_idx", ckpt.get("text_to_idx", {}))
            self._qry_text_to_idx = ckpt.get("qry_text_to_idx", {})
            self._smp_idx = ckpt.get("smp_idx")
            self._qry_cat_idx = ckpt.get("qry_cat_idx")
            self._qry_bar_cat_idx = ckpt.get("qry_bar_cat_idx")

            if self._qry_embed_cache is None and self._doc_embed_cache is not None:
                self._qry_embed_cache = self._doc_embed_cache

            if self._doc_embed_cache is not None:
                self._embed_cache = self._doc_embed_cache
                self._embed_dim = int(self._doc_embed_cache.shape[1])
                if expected_embed_dim is not None and self._embed_cache.shape[1] > expected_embed_dim:
                    self._doc_embed_cache = self._doc_embed_cache[:, :expected_embed_dim]
                    self._qry_embed_cache = self._qry_embed_cache[:, :expected_embed_dim]
                    self._embed_cache = self._doc_embed_cache
                    self._embed_dim = expected_embed_dim
                self._text_to_idx = self._doc_text_to_idx
                print(
                    f"[CTA][SMP][cache] loaded {cache_file.name} "
                    f"doc_texts={len(self._doc_text_to_idx)} qry_texts={len(self._qry_text_to_idx)}",
                    flush=True,
                )
                return

        doc_text_to_idx: dict[str, int] = {}
        query_text_to_idx: dict[str, int] = {}

        for _, up in self._samples:
            for txt in (up.col_header_a, up.cell_value_a, up.cell_value_b, up.col_header_b):
                if txt not in doc_text_to_idx:
                    doc_text_to_idx[txt] = len(doc_text_to_idx)
            fmt = dict(
                pivot_a=up.col_header_a, node_a=up.cell_value_a,
                node_b=up.cell_value_b, pivot_b=up.col_header_b,
            )
            for txt in (self._cat_qry_tmpl.format(**fmt), self._cat_qry_bar_tmpl.format(**fmt)):
                if txt not in query_text_to_idx:
                    query_text_to_idx[txt] = len(query_text_to_idx)

        ordered_doc = [""] * len(doc_text_to_idx)
        for txt, i in doc_text_to_idx.items():
            ordered_doc[i] = txt

        ordered_query = [""] * len(query_text_to_idx)
        for txt, i in query_text_to_idx.items():
            ordered_query[i] = txt

        print(
            f"[CTA][SMP][embed] {len(ordered_doc)} unique documents + {len(ordered_query)} unique queries …",
            flush=True,
        )
        if hasattr(self._embedder, "encode_documents"):
            self._doc_embed_cache = torch.tensor(self._embedder.encode_documents(ordered_doc), dtype=torch.float32)
        else:
            self._doc_embed_cache = torch.tensor(self._embedder.embed_documents(ordered_doc), dtype=torch.float32)

        if hasattr(self._embedder, "encode_queries"):
            qry_vecs = self._embedder.encode_queries(ordered_query)
        elif hasattr(self._embedder, "embed_query"):
            qry_vecs = [self._embedder.embed_query(t) for t in ordered_query]
        else:
            qry_vecs = self._embedder.embed_documents(ordered_query)
        self._qry_embed_cache = torch.tensor(qry_vecs, dtype=torch.float32)

        self._embed_dim = int(self._doc_embed_cache.shape[1])

        if expected_embed_dim is not None and self._doc_embed_cache.shape[1] > expected_embed_dim:
            self._doc_embed_cache = self._doc_embed_cache[:, :expected_embed_dim]
            self._qry_embed_cache = self._qry_embed_cache[:, :expected_embed_dim]
            self._embed_dim = expected_embed_dim

        self._doc_text_to_idx = doc_text_to_idx
        self._qry_text_to_idx = query_text_to_idx
        self._embed_cache = self._doc_embed_cache
        self._text_to_idx = self._doc_text_to_idx

        smp_rows, qry_rows, qry_bar_rows = [], [], []
        for _, up in self._samples:
            smp_rows.append([
                doc_text_to_idx[up.col_header_a],
                doc_text_to_idx[up.cell_value_a],
                doc_text_to_idx[up.cell_value_b],
                doc_text_to_idx[up.col_header_b],
            ])
            fmt = dict(
                pivot_a=up.col_header_a, node_a=up.cell_value_a,
                node_b=up.cell_value_b, pivot_b=up.col_header_b,
            )
            qry_rows.append(query_text_to_idx[self._cat_qry_tmpl.format(**fmt)])
            qry_bar_rows.append(query_text_to_idx[self._cat_qry_bar_tmpl.format(**fmt)])

        self._smp_idx = torch.tensor(smp_rows, dtype=torch.long)
        self._qry_cat_idx = torch.tensor(qry_rows, dtype=torch.long)
        self._qry_bar_cat_idx = torch.tensor(qry_bar_rows, dtype=torch.long)

        if cache_embeddings:
            torch.save({
                "doc_embed_cache": self._doc_embed_cache,
                "qry_embed_cache": self._qry_embed_cache,
                "doc_text_to_idx": self._doc_text_to_idx,
                "qry_text_to_idx": self._qry_text_to_idx,
                "smp_idx": self._smp_idx,
                "qry_cat_idx": self._qry_cat_idx,
                "qry_bar_cat_idx": self._qry_bar_cat_idx,
                "embed_dim": int(self._doc_embed_cache.shape[1]),
            }, cache_file)
            print(
                f"[CTA][SMP][cache] saved {cache_file.name}  "
                f"({cache_file.stat().st_size / 1024**2:.1f} MB)",
                flush=True,
            )

    # ── Pickle support (spawn multiprocessing) ────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_st_model", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            if self._embed_cache is not None:
                self._embed_dim = int(self._embed_cache.shape[1])
            else:
                self._embed_dim = 384  # fallback
        return self._embed_dim

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, up = self._samples[idx]

        if self._smp_idx is not None:
            smp_embeds = self._doc_embed_cache[self._smp_idx[idx]]  # [4, d]
            query_embeds = self._qry_embed_cache[self._qry_cat_idx[idx]].unsqueeze(0)  # [1, d]
            query_bar_embeds = self._qry_embed_cache[self._qry_bar_cat_idx[idx]].unsqueeze(0)  # [1, d]
        else:
            raise RuntimeError("CTASMPDataset requires precompute=True")

        return {
            "smp_embeds":       smp_embeds,
            "query_embeds":     query_embeds,
            "query_bar_embeds": query_bar_embeds,
            "question_emb":     torch.zeros(self.embed_dim),   # no retrieval question in CTA
            "record_idx":       rec_idx,
            "upath":            up,
        }


# ── SMP DataModule (pretraining) ──────────────────────────────────────────────

class CTASMPDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapping CTASMPDataset for the pretraining stage.

    Provides train + val DataLoaders using jepa_collate_fn from TRL-model
    so that the standard TableEmbedJePA Trainer works without modification.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._dataset: Optional[CTASMPDataset] = None

    def _make_dataset(self, data_path: str, split: str) -> CTASMPDataset:
        cfg = self.cfg
        max_rows_key = f"max_rows_{split}"
        _max_rows = cfg.data.get(max_rows_key)
        return CTASMPDataset(
            data_path=data_path,
            model_type=cfg.embedder.model_type,
            base_url=cfg.embedder.get("base_url"),
            model_name=cfg.embedder.get("model_name"),
            api_key=cfg.embedder.get("api_key"),
            max_records=cfg.data.get("max_records"),
            max_rows_per_table=_max_rows,
            use_graph_walks=cfg.smp.use_graph_walks,
            num_walks=cfg.smp.num_walks,
            chunk_size=cfg.smp.chunk_size,
            precompute=cfg.embedder.precompute,
            embed_batch_size=cfg.embedder.embed_batch_size,
            cache_embeddings=cfg.embedder.cache_embeddings,
            embed_cache_dir=cfg.embedder.get("embed_cache_dir"),
            cat_qry_template=cfg.query.cat_qry_template,
            cat_qry_bar_template=cfg.query.cat_qry_bar_template,
            expected_embed_dim=int(cfg.embedder.embed_dim) if cfg.embedder.get("embed_dim") else None,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        paths = resolve_data_paths(self.cfg.data)
        if stage in (None, "fit"):
            t0 = time.perf_counter()
            print("[CTA][SMP][setup] building train dataset …", flush=True)
            self.train_ds = self._make_dataset(paths["train"], "train")
            print("[CTA][SMP][setup] building dev dataset …", flush=True)
            self.val_ds   = self._make_dataset(paths["dev"],   "dev")
            self.embed_dim = self.train_ds.embed_dim
            self._dataset = self.train_ds
            print(
                f"[CTA][SMP][setup] fit datasets ready "
                f"train_tables={len(self.train_ds.records)} train_paths={len(self.train_ds)} "
                f"dev_tables={len(self.val_ds.records)} dev_paths={len(self.val_ds)} "
                f"embed_dim={self.embed_dim} elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    def train_dataloader(self) -> DataLoader:
        nw = self.cfg.pretraining.dataloader_num_workers
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.pretraining.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=_smp_collate,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self) -> DataLoader:
        nw = self.cfg.pretraining.dataloader_num_workers
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.pretraining.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=_smp_collate,
            pin_memory=True,
            persistent_workers=nw > 0,
        )


def _smp_collate(batch: list[dict]) -> dict:
    """
    Collate U-path samples for TableEmbedJePA training.

    Mirrors jepa_collate_fn from TRL-model/dataset.py exactly so that
    CTASMPDataModule is a drop-in replacement for TableEmbedJePADataModule.

    SMP_bar reverses node order: [pivot_a, node_a, node_b, pivot_b]
                               → [pivot_b, node_b, node_a, pivot_a]
    MASK_SMP_LEVEL_LOSS [2B, 2B]: positive pair = (SMP[i], SMP_bar[i]).
    """
    B = len(batch)
    smp_embeds       = torch.stack([b["smp_embeds"]       for b in batch])  # [B, 4, d]
    query_embeds     = torch.stack([b["query_embeds"]     for b in batch])  # [B, 1, d]
    query_bar_embeds = torch.stack([b["query_bar_embeds"] for b in batch])  # [B, 1, d]
    question_embs    = torch.stack([b["question_emb"]     for b in batch])  # [B, d]
    record_indices   = torch.tensor([b["record_idx"] for b in batch], dtype=torch.long)

    smp_bar_embeds = smp_embeds[:, [3, 2, 1, 0], :]  # [B, 4, d]

    mask = torch.zeros(2 * B, 2 * B)
    idx  = torch.arange(B)
    mask[idx, B + idx] = 1.0
    mask[B + idx, idx] = 1.0
    mask.fill_diagonal_(4.0)

    return {
        "smp_embeds":          smp_embeds,
        "smp_bar_embeds":      smp_bar_embeds,
        "query_embeds":        query_embeds,
        "query_bar_embeds":    query_bar_embeds,
        "MASK_SMP_LEVEL_LOSS": mask,
        "question_embeds":     question_embs,
        "record_indices":      record_indices,
    }

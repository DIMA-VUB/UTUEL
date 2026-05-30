"""
dataset.py
PyTorch Dataset for training / evaluating TableEmbedJePA on WikiSQL JSONL data.

U-path sequence layout
──────────────────────
Each U-path produces three sequences (pre-computed LLM node embeddings):

  SMP     : [pivot_a, node_a, node_b, pivot_b]   — 4 nodes  [4, d]
  SMP_bar : [pivot_b, node_b, node_a, pivot_a]   — reversed  [4, d]
  Query   : [concat(pivot_a, node_b, pivot_b)]    — single concatenated LLM embed  [1, d]

The model prepends a learnable CLS token at runtime, making sequences of
length 5 (SMP / SMP_bar) and 4 (query).

Batch layout
────────────
  smp_embeds          [B, 4, d]   — SMP node embeddings
  smp_bar_embeds      [B, 4, d]   — SMP_bar node embeddings (reversed)
  query_embeds        [B, 1, d]   — single concatenated context query embedding
  MASK_SMP_LEVEL_LOSS [2B, 2B]    — positive-pair mask for global SMP loss
                                    1 = positive (SMP[i] <-> SMP_bar[i])
                                    0 = negative
                                    4 = diagonal (self, ignored)
  question_embeds     [B, d]      — question embedding (for evaluation)
  record_indices      [B]         — index into self.records

Embedder backends
─────────────────
  get_embedder() returns a LangChain embedder:
    - Ollama  (default)  — any model served by a local Ollama instance
    - OpenAI             — text-embedding-3-large
    - HuggingFace Hub    — sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import json
import requests
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

try:
    from .smp import UPath, generate_u_paths_flat, generate_u_paths_from_graph
except ImportError:
    from smp import UPath, generate_u_paths_flat, generate_u_paths_from_graph

SMP_NODE_LEN = 4   # pivot_a, node_a, node_b, pivot_b
QRY_NODE_LEN = 1   # concat(pivot_a, node_b, pivot_b) → single LLM embedding


# ── Embedder factory ──────────────────────────────────────────────────────────

def get_embedder(
    model_type: str = "llama3",
    base_url: Optional[str] = "http://134.184.22.126:10434/",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Return a LangChain embedder for the requested backend.

    Args:
        model_type:  ``"openai"``, ``"huggingface"``, or any Ollama model name
                     (e.g. ``"llama3"``, ``"nomic-embed-text"``).
        base_url:    Ollama server URL. Ignored for OpenAI / HuggingFace.
        model_name:  Overrides the HuggingFace Hub model name when
                     ``model_type == "huggingface"``.
        api_key:     API key for OpenAI or HuggingFace Hub backends.
    """
    tag = model_type.strip().lower()

    if tag == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

    if tag == "huggingface":
        from langchain_huggingface import HuggingFaceHubEmbeddings
        hf_model = model_name or "sentence-transformers/all-mpnet-base-v2"
        return HuggingFaceHubEmbeddings(
            model=hf_model,
            task="feature-extraction",
            huggingfacehub_api_token=api_key,
        )

    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(base_url=base_url, model=tag)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TableEmbedJePADataset(Dataset):
    """
    Loads WikiSQL JSONL records and produces U-path JEPA training samples.

    Each item corresponds to one U-path from a table record and provides:
      - smp_embeds   [4, d]: pivot_a, node_a, node_b, pivot_b
      - query_embeds [1, d]: LLM embedding of concat(pivot_a_text, node_b_text, pivot_b_text)
      - question_emb [d]   : embedded question (for evaluation)
      - record_idx   int   : index into self.records

    All embeddings are precomputed once by embedding every unique node text
    in a single bulk call, then indexed at __getitem__ time.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        model_type: str = "llama3",
        base_url: Optional[str] = "http://134.184.22.126:10434/",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_records: Optional[int] = None,
        filter_table_id: Optional[str] = None,
        precompute: bool = True,
        embed_batch_size: int = 256,
        use_graph_walks: bool = False,
        num_walks: int = 50,
        chunk_size: int = 1,
        truncate_embed_dim: Optional[int] = None,
        cache_embeddings: bool = False,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        cat_qry_template: str = "what is {pivot_a} of {node_b}({pivot_b})?",
        cat_qry_bar_template: str = "what is {pivot_b} of {node_a}({pivot_a})?",
    ) -> None:
        # ── Load records ──────────────────────────────────────────────────────
        # Always loads ALL records; unique_tables_only is a DataModule concern
        # that uses SubsetRandomSampler to limit *training*, not data loading.
        self.records: list[dict] = []
        path = Path(jsonl_path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if filter_table_id is not None and str(rec.get("table_id", "")) != filter_table_id:
                    continue
                self.records.append(rec)
                if max_records and len(self.records) >= max_records:
                    break
        tag = f" [table_id={filter_table_id}]" if filter_table_id else ""
        print(f"[dataset][load] {len(self.records)} records from {path.name}{tag}")

        self._model_type = model_type
        self._base_url   = (base_url or "").rstrip("/")
        self._model_name = model_name
        self._api_key    = api_key
        # HuggingFace backend uses direct REST / sentence_transformers — no LangChain needed.
        # _embedder is only used for Ollama/OpenAI single-query calls (embed_query).
        _tag = model_type.strip().lower()
        self._embedder = (
            None if _tag == "huggingface"
            else get_embedder(model_type, base_url, model_name, api_key)
        )
        self._embed_dim: Optional[int] = None
        self._truncate_embed_dim: Optional[int] = truncate_embed_dim
        self._jsonl_path       = path
        self._max_records      = max_records
        self._filter_table_id  = filter_table_id
        self._use_graph_walks  = use_graph_walks
        self._chunk_size       = chunk_size
        self._cache_embeddings = cache_embeddings
        self._embed_cache_dir  = Path(embed_cache_dir) if embed_cache_dir else None
        self._cat_qry_template     = cat_qry_template
        self._cat_qry_bar_template = cat_qry_bar_template

        # ── Generate U-paths per record ───────────────────────────────────────
        _gen = (
            (lambda h, r: generate_u_paths_from_graph(h, r, num_walks=num_walks,
                                                       chunk_size=chunk_size))
            if use_graph_walks
            else (lambda h, r: generate_u_paths_flat(h, r))
        )
        self._all_upaths: list[list[UPath]] = []
        for rec in self.records:
            rows = [list(map(str, row)) for row in rec["rows"]]
            _upaths = _gen(rec["header"], rows)
            _tid = str(rec.get("table_id", ""))
            _rid = str(rec.get("id", ""))
            for up in _upaths:
                up.table_id  = _tid
                up.record_id = _rid
            self._all_upaths.append(_upaths)

        # ── Build flat sample list ────────────────────────────────────────────
        self._samples: list[tuple[int, UPath]] = [
            (rec_idx, upath)
            for rec_idx, upaths in enumerate(self._all_upaths)
            for upath in upaths
        ]
        print(f"[dataset][smp_gen] {len(self._samples)} U-path samples "
              f"from {len(self.records)} records")

        # ── Index of training samples for unique-table mode ───────────────────
        # Contains ALL flat sample indices (into _samples) whose record is the
        # *first* record encountered for that table_id.  The DataModule uses
        # SubsetRandomSampler on this list when unique_tables_only=True so the
        # full dataset is precomputed once and evaluation always has all records.
        _first_rec_per_table: dict[str, int] = {}
        for _r, _rec in enumerate(self.records):
            _tid = str(_rec.get("table_id", _r))
            if _tid not in _first_rec_per_table:
                _first_rec_per_table[_tid] = _r
        _unique_idxs: list[int] = [
            _j for _j, (_r, _) in enumerate(self._samples)
            if _first_rec_per_table.get(str(self.records[_r].get("table_id", _r))) == _r
        ]
        self._unique_table_train_idxs: list[int] = _unique_idxs
        print(f"[dataset][unique_tables] {len(_first_rec_per_table)} unique tables → "
              f"{len(_unique_idxs)} training samples (of {len(self._samples)} total)")

        # ── Precompute embeddings ─────────────────────────────────────────────
        self._embed_cache: Optional[torch.Tensor] = None
        self._text_to_idx: Optional[dict] = None
        self._question_cache: Optional[torch.Tensor] = None

        # Pre-built integer index tensors for O(1) tensor-slice __getitem__.
        # Built during _precompute_embeddings; None in live-embed mode.
        self._smp_idx:         Optional[torch.Tensor] = None   # [N, 4]
        self._qry_cat_idx:     Optional[torch.Tensor] = None   # [N]  — index into _embed_cache for concat query (SMP)
        self._qry_bar_cat_idx: Optional[torch.Tensor] = None   # [N]  — index into _embed_cache for concat query (SMP_bar)
        self._rec_idx:         Optional[torch.Tensor] = None   # [N]

        if precompute:
            self._precompute_embeddings(embed_batch_size)

    # ── Embedding helpers ─────────────────────────────────────────────────────

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            if self._embedder is not None:
                self._embed_dim = len(self._embedder.embed_query("probe"))
            else:
                # HuggingFace path — derive dim from one live embed
                self._embed_dim = len(self._embed_batch_chunk(["probe"])[0])
        return self._embed_dim

    def _embed_batch_chunk(self, texts: list[str]) -> list[list[float]]:
        """
        Embed one chunk via the backend's native batch API.

        Ollama       — POST /api/embed  ``{input: [...]}``
                        Each string is tokenised and encoded independently.
        HuggingFace  — with api_key: POST to HF Inference API
                        ``/pipeline/feature-extraction/{model}``
                      — without api_key: local sentence_transformers.encode()
        OpenAI       — LangChain (their SDK already batches independently).
        """
        tag = self._model_type.strip().lower()

        if tag == "huggingface":
            model = self._model_name or "sentence-transformers/all-mpnet-base-v2"
            if self._api_key:
                resp = requests.post(
                    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"inputs": texts},
                    timeout=120000,
                )
                resp.raise_for_status()
                return resp.json()  # [[d], [d], ...]
            else:
                from sentence_transformers import SentenceTransformer
                if not hasattr(self, "_st_model"):
                    self._st_model = SentenceTransformer(model)
                vecs = self._st_model.encode(
                    texts, batch_size=len(texts), show_progress_bar=False
                )
                return vecs.tolist()

        if tag != "openai":
            model = self._model_name or tag
            resp = requests.post(
                f"{self._base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=120000,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

        return self._embedder.embed_documents(texts)

    def _batch_embed(self, texts: list, batch_size: int) -> torch.Tensor:
        """Embed *texts* in chunks; each text is processed independently."""
        all_vecs = []
        n = len(texts)
        for start in range(0, n, batch_size):
            chunk = texts[start : start + batch_size]
            all_vecs.extend(self._embed_batch_chunk(chunk))
            done = min(start + batch_size, n)
            print(f"\r[dataset][embed] {done}/{n}  ({100 * done / n:.1f}%)", end="", flush=True)
        print()
        tensor = torch.tensor(all_vecs, dtype=torch.float32)
        self._embed_dim = tensor.shape[1]
        return tensor

    def _precompute_embeddings(self, batch_size: int) -> None:
        """
        Collect all unique node texts and question texts, embed in bulk,
        then build integer index tensors so __getitem__ is a pure tensor slice.

        Memory layout after this call
        ─────────────────────────────
        _embed_cache     [N_unique, d]  — one row per *unique* node/query text
        _question_cache  [R, d]         — one row per record question
        _smp_idx         [N, 4]  long   — row indices into _embed_cache
        _qry_cat_idx     [N]     long   — row index for concatenated query text
        _rec_idx         [N]     long   — row indices into _question_cache

        Cache behaviour
        ───────────────
        When cache_embeddings=True a .embed_cache.pt file is saved with the
        *full* embedding dim (e.g. _dim_768.embed_cache.pt).  On future runs the
        code searches for any cache file for the same dataset+model whose stored
        dim is >= the requested truncate_embed_dim, loads it, and trims in-memory.
        This means a single full-dim cache can serve all smaller truncation targets.
        """
        _model_slug  = (self._model_name or self._model_type).replace("/", "-").replace("\\", "-")
        _cache_dir   = self._embed_cache_dir if self._embed_cache_dir else self._jsonl_path.parent
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _stem        = self._jsonl_path.stem
        _prefix      = f"{_stem}_{_model_slug}_dim_"          # e.g. myset_nomic-embed-text_dim_
        _target_dim  = self._truncate_embed_dim               # None = keep full dim

        # ── Find the smallest existing cache whose dim >= target_dim ────────────
        _best_file: Optional[Path] = None
        _best_dim:  Optional[int]  = None
        for _f in sorted(_cache_dir.glob(f"{_prefix}*.embed_cache.pt")):
            try:
                _dim_val = int(_f.name[len(_prefix):].split(".embed_cache.pt")[0])
            except ValueError:
                continue
            if _target_dim is None or _dim_val >= _target_dim:
                if _best_dim is None or _dim_val < _best_dim:   # prefer smallest sufficient
                    _best_dim, _best_file = _dim_val, _f

        # ── Try loading from disk ───────────────────────────────────────────────
        if _best_file is not None:
            print(f"[dataset][cache] loading from {_best_file.name}  (stored dim={_best_dim})")
            _ckpt = torch.load(_best_file, map_location="cpu", weights_only=False)
            self._embed_cache     = _ckpt["embed_cache"]
            self._question_cache  = _ckpt["question_cache"]
            self._text_to_idx     = _ckpt["text_to_idx"]
            self._smp_idx         = _ckpt["smp_idx"]
            self._qry_cat_idx     = _ckpt["qry_cat_idx"]
            self._qry_bar_cat_idx = _ckpt["qry_bar_cat_idx"]
            self._rec_idx         = _ckpt["rec_idx"]
            self._embed_dim       = _ckpt["embed_dim"]
            if _target_dim is not None and self._embed_dim > _target_dim:
                self._embed_cache    = self._embed_cache[:, :_target_dim].contiguous()
                self._question_cache = self._question_cache[:, :_target_dim].contiguous()
                self._embed_dim      = _target_dim
                print(f"[dataset][cache] trimmed in-memory to first {_target_dim} dims")
            print(f"[dataset][cache] loaded  embed_cache={tuple(self._embed_cache.shape)}  "
                  f"embed_dim={self._embed_dim}")
            return

        # ── Compute embeddings ──────────────────────────────────────────────────
        text_to_idx: dict = {}
        for _, upath in self._samples:
            for text in (upath.col_header_a, upath.cell_value_a,
                         upath.cell_value_b, upath.col_header_b):
                if text not in text_to_idx:
                    text_to_idx[text] = len(text_to_idx)
            _fmt = dict(pivot_a=upath.col_header_a, node_a=upath.cell_value_a,
                        node_b=upath.cell_value_b,  pivot_b=upath.col_header_b)
            for text in (self._cat_qry_template.format(**_fmt),
                         self._cat_qry_bar_template.format(**_fmt)):
                if text not in text_to_idx:
                    text_to_idx[text] = len(text_to_idx)

        all_texts = [""] * len(text_to_idx)
        for text, idx in text_to_idx.items():
            all_texts[idx] = text

        print(f"[dataset][embed] {len(all_texts)} unique texts (batch_size={batch_size})...")
        self._embed_cache = self._batch_embed(all_texts, batch_size)
        self._text_to_idx = text_to_idx

        questions = [rec.get("question", "") for rec in self.records]
        print(f"[dataset][embed_questions] {len(questions)} question embeddings...")
        self._question_cache = self._batch_embed(questions, batch_size)

        # Build index tensors once — avoids per-sample dict lookups at train time
        smp_rows, qry_cat_rows, qry_bar_cat_rows, rec_list = [], [], [], []
        for r_idx, upath in self._samples:
            pa = text_to_idx[upath.col_header_a]
            na = text_to_idx[upath.cell_value_a]
            nb = text_to_idx[upath.cell_value_b]
            pb = text_to_idx[upath.col_header_b]
            smp_rows.append([pa, na, nb, pb])
            _fmt = dict(pivot_a=upath.col_header_a, node_a=upath.cell_value_a,
                        node_b=upath.cell_value_b,  pivot_b=upath.col_header_b)
            cat_qry     = self._cat_qry_template.format(**_fmt)
            cat_qry_bar = self._cat_qry_bar_template.format(**_fmt)
            qry_cat_rows.append(text_to_idx[cat_qry])
            qry_bar_cat_rows.append(text_to_idx[cat_qry_bar])
            rec_list.append(r_idx)
        self._smp_idx         = torch.tensor(smp_rows,         dtype=torch.long)  # [N, 4]
        self._qry_cat_idx     = torch.tensor(qry_cat_rows,     dtype=torch.long)  # [N]
        self._qry_bar_cat_idx = torch.tensor(qry_bar_cat_rows, dtype=torch.long)  # [N]
        self._rec_idx         = torch.tensor(rec_list,         dtype=torch.long)  # [N]

        # ── Save full-dim cache — must happen before in-memory truncation ────────
        # File is named with the *actual* embed dim so any smaller truncate_embed_dim
        # can reuse it on future runs without re-computing.
        _actual_dim  = int(self._embed_cache.shape[1])
        _cache_file  = _cache_dir / f"{_prefix}{_actual_dim}.embed_cache.pt"
        print(f"[dataset][build_index] done  embed_dim={_actual_dim}")
        if self._cache_embeddings:
            print(f"[dataset][cache] saving full-dim ({_actual_dim}) cache to {_cache_file.name} …")
            torch.save({
                "embed_cache":     self._embed_cache,
                "question_cache":  self._question_cache,
                "text_to_idx":     self._text_to_idx,
                "smp_idx":         self._smp_idx,
                "qry_cat_idx":     self._qry_cat_idx,
                "qry_bar_cat_idx": self._qry_bar_cat_idx,
                "rec_idx":         self._rec_idx,
                "embed_dim":       _actual_dim,
            }, _cache_file)
            print(f"[dataset][cache] saved   ({_cache_file.stat().st_size / 1024**2:.1f} MB)")

        # Truncate in-memory after saving (saved file keeps full dim for reuse)
        if _target_dim is not None:
            self._embed_cache    = self._embed_cache[:, :_target_dim].contiguous()
            self._question_cache = self._question_cache[:, :_target_dim].contiguous()
            self._embed_dim      = _target_dim
            print(f"[dataset][truncate] in-memory truncated to first {_target_dim} dims")

    def _get_node_embed(self, text: str) -> torch.Tensor:
        return self._embed_cache[self._text_to_idx[text]]

    def _embed_live(self, text: str) -> torch.Tensor:
        vecs = self._embed_batch_chunk([text])
        self._embed_dim = len(vecs[0])
        return torch.tensor(vecs[0], dtype=torch.float32)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    # Windows multiprocessing (spawn) pickles the dataset for each DataLoader
    # worker.  SentenceTransformer holds _thread.RLock objects that cannot be
    # pickled.  Workers only need the precomputed tensors (_embed_cache, etc.),
    # so we simply drop _st_model and _embedder from the pickled state.
    # The main process retains them for evaluation / on-demand embedding calls.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_st_model", None)   # SentenceTransformer — has thread locks
        state.pop("_embedder", None)   # LangChain embedder  — may also have locks
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Workers operate purely via the fast tensor path (__getitem__ precomputed).
        # _st_model / _embedder are not restored; the main process keeps its own copy.

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, upath = self._samples[idx]

        if self._smp_idx is not None:
            # Fast path: pure tensor indexing — no dict lookup, no string hashing
            smp_embeds       = self._embed_cache[self._smp_idx[idx]]                      # [4, d]
            query_embeds     = self._embed_cache[self._qry_cat_idx[idx]].unsqueeze(0)     # [1, d]
            query_bar_embeds = self._embed_cache[self._qry_bar_cat_idx[idx]].unsqueeze(0) # [1, d]
            question_emb     = self._question_cache[self._rec_idx[idx]]                   # [d]
        else:
            # Live-embed fallback (precompute=False)
            pa = self._embed_live(upath.col_header_a)
            na = self._embed_live(upath.cell_value_a)
            nb = self._embed_live(upath.cell_value_b)
            pb = self._embed_live(upath.col_header_b)
            smp_embeds   = torch.stack([pa, na, nb, pb])  # [4, d]
            _fmt = dict(pivot_a=upath.col_header_a, node_a=upath.cell_value_a,
                        node_b=upath.cell_value_b,  pivot_b=upath.col_header_b)
            cat_qry_text     = self._cat_qry_template.format(**_fmt)
            cat_qry_bar_text = self._cat_qry_bar_template.format(**_fmt)
            query_embeds     = self._embed_live(cat_qry_text).unsqueeze(0)      # [1, d]
            query_bar_embeds = self._embed_live(cat_qry_bar_text).unsqueeze(0)  # [1, d]
            question_emb     = self._embed_live(self.records[rec_idx].get("question", ""))

        return {
            "smp_embeds":       smp_embeds,       # [4, d]
            "query_embeds":     query_embeds,     # [1, d]
            "query_bar_embeds": query_bar_embeds, # [1, d]
            "question_emb":     question_emb,     # [d]
            "record_idx":       rec_idx,
            "upath":            upath,
        }

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def get_table_upaths(self, record_idx: int) -> list:
        return self._all_upaths[record_idx]

    def embed_question(self, question: str) -> torch.Tensor:
        vecs = self._embed_batch_chunk([question])
        return torch.tensor(vecs[0], dtype=torch.float32)


# ── Collation ─────────────────────────────────────────────────────────────────

def jepa_collate_fn(batch: list) -> dict:
    """
    Collate U-path samples into a JEPA training batch.

    SMP_bar is the SMP with its 4-node sequence reversed:
      SMP     : [pivot_a, node_a, node_b, pivot_b]
      SMP_bar : [pivot_b, node_b, node_a, pivot_a]   <- index [3,2,1,0]

    MASK_SMP_LEVEL_LOSS [2B, 2B]:
      Combined sequence = [SMP_0..SMP_{B-1} | SMP_bar_0..SMP_bar_{B-1}]
      Positive pairs : (i, B+i) and (B+i, i)
      Diagonal       : 4  (self, ignored)
      Others         : 0  (negatives)
    """
    B = len(batch)

    smp_embeds       = torch.stack([b["smp_embeds"]       for b in batch])  # [B, 4, d]
    query_embeds     = torch.stack([b["query_embeds"]     for b in batch])  # [B, 1, d]
    query_bar_embeds = torch.stack([b["query_bar_embeds"] for b in batch])  # [B, 1, d]
    question_embs    = torch.stack([b["question_emb"]     for b in batch])  # [B, d]
    record_indices = torch.tensor([b["record_idx"] for b in batch], dtype=torch.long)

    # SMP_bar: reverse node order
    smp_bar_embeds = smp_embeds[:, [3, 2, 1, 0], :]  # [B, 4, d]

    # Build MASK_SMP_LEVEL_LOSS [2B, 2B]
    mask = torch.zeros(2 * B, 2 * B)
    idx = torch.arange(B)
    mask[idx, B + idx] = 1.0      # SMP[i]     positive = SMP_bar[i]
    mask[B + idx, idx] = 1.0      # SMP_bar[i] positive = SMP[i]
    mask.fill_diagonal_(4.0)      # self, ignored

    return {
        "smp_embeds":          smp_embeds,       # [B, 4, d]
        "smp_bar_embeds":      smp_bar_embeds,   # [B, 4, d]
        "query_embeds":        query_embeds,     # [B, 1, d]  SMP query:     pivot_a + node_b + pivot_b
        "query_bar_embeds":    query_bar_embeds, # [B, 1, d]  SMP_bar query: pivot_b + node_a + pivot_a
        "MASK_SMP_LEVEL_LOSS": mask,             # [2B, 2B]
        "question_embeds":     question_embs,    # [B, d]
        "record_indices":      record_indices,   # [B]
    }


# ── PyTorch Lightning DataModule ──────────────────────────────────────────────

class TableEmbedJePADataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping TableEmbedJePADataset."""

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 0,
        model_type: str = "llama3",
        base_url: Optional[str] = "http://134.184.22.126:10434/",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_records: Optional[int] = None,
        filter_table_id: Optional[str] = None,
        unique_tables_only: bool = False,
        precompute: bool = True,
        embed_batch_size: int = 256,
        use_graph_walks: bool = False,
        num_walks: int = 50,
        chunk_size: int = 1,
        truncate_embed_dim: Optional[int] = None,
        cache_embeddings: bool = False,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        cat_qry_template: str = "what is {pivot_a} of {node_b}({pivot_b})?",
        cat_qry_bar_template: str = "what is {pivot_b} of {node_a}({pivot_a})?",
    ):
        super().__init__()
        self.jsonl_path  = jsonl_path
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self._unique_tables_only = unique_tables_only
        self._ds_kwargs  = dict(
            model_type=model_type,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            max_records=max_records,
            filter_table_id=filter_table_id,
            precompute=precompute,
            embed_batch_size=embed_batch_size,
            use_graph_walks=use_graph_walks,
            num_walks=num_walks,
            chunk_size=chunk_size,
            truncate_embed_dim=truncate_embed_dim,
            cache_embeddings=cache_embeddings,
            embed_cache_dir=embed_cache_dir,
            cat_qry_template=cat_qry_template,
            cat_qry_bar_template=cat_qry_bar_template,
        )
        self._dataset: Optional[TableEmbedJePADataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._dataset is None:
            self._dataset = TableEmbedJePADataset(
                jsonl_path=self.jsonl_path, **self._ds_kwargs)

    @property
    def embed_dim(self) -> int:
        if self._dataset is None:
            raise RuntimeError("Call setup() before accessing embed_dim.")
        return self._dataset.embed_dim

    def train_dataloader(self) -> DataLoader:
        _common = dict(
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=jepa_collate_fn,
        )
        if self._unique_tables_only:
            # Restrict training to U-paths from the first record of each table.
            # The full dataset (all records) remains available for evaluation.
            _idxs = self._dataset._unique_table_train_idxs
            print(f"[datamodule] unique_tables_only — training on "
                  f"{len(_idxs):,} samples from "
                  f"{len(set(str(self._dataset.records[r].get('table_id', r)) for r, _ in (self._dataset._samples[i] for i in _idxs))):,} tables "
                  f"(full dataset: {len(self._dataset):,} samples)")
            return DataLoader(
                self._dataset,
                sampler=SubsetRandomSampler(_idxs),
                **_common,
            )
        return DataLoader(self._dataset, shuffle=True, **_common)

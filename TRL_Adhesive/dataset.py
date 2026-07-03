"""
dataset.py
PyTorch Dataset for training / evaluating TableEmbedJePA on the **complex /
nested Adhesive** table dataset.

Difference from ``TRL-model/dataset.py``
─────────────────────────────────────────
``TRL-model`` reads a single WikiSQL-style JSONL and builds U-paths from a
regular header-row + data-row grid.  This variant instead reads a *directory*
of pre-computed Semantic Meta-Path walks (``.txt``) plus per-table cell JSON and
per-table question/answer JSON, and builds ``header | node | header`` U-paths
via :mod:`smp` (see that module for the parsing rules).

Directory layout expected at ``data_dir``
  AdhesiveTable_SMP_format/<uuid>.txt      — SMP walks
  AdhesiveTable_json_format/<uuid>.json    — cell descriptions
  QUESTIONS_ANSWERS_PER_TABLE/<uuid>.json  — questions + answer_cell_id

Records
───────
Each *question* becomes one record (mirroring ``TRL-model`` where every question
is a record that shares its table's U-paths).  A record carries::

  table_id, id, question, answer, answer_cell_id, row_header, column_header

U-paths are generated **once per table** and shared by every record of that
table, so ``unique_tables_only`` (first record per table) trains without
duplicated tables — exactly as in ``TRL-model``.

SMP sequence layout (variable-length, role-masked — one slot per walk cell)
  SMP     : [a_h1, …, node, …, b_h1]      [L, d]   role 1 at node, 0 at headers
  SMP_bar : reversed(SMP)                  [L, d]   (nested-header order flipped)
  Query   : concat(pivot_a, node, pivot_b) [1, d]

Samples are right-padded to the dataset's max walk length; a per-sample length
drives the padding / attention mask so the model can locate the node via the
role mask instead of a fixed index.
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
    from .smp import UPath, generate_upaths_for_table, generate_upaths_from_json
except ImportError:
    from smp import UPath, generate_upaths_for_table, generate_upaths_from_json

try:
    from .hit_util import high_informative_query
except ImportError:
    from hit_util import high_informative_query

SMP_NODE_LEN = 4   # legacy constant (kept for signature parity; unused)
QRY_NODE_LEN = 1   # concat(pivot_a, node, pivot_b) → single LLM embedding

# Sub-folder names inside the Adhesive dataset directory.
SMP_SUBDIR  = "AdhesiveTable_SMP_format"
JSON_SUBDIR = "AdhesiveTable_json_format"
QA_SUBDIR   = "QUESTIONS_ANSWERS_PER_TABLE"


# ── Embedder factory ──────────────────────────────────────────────────────────

def get_embedder(
    model_type: str = "llama3",
    base_url: Optional[str] = "http://134.184.22.126:10434/",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Return a LangChain embedder for the requested backend (see TRL-model)."""
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


# ── Record loading ────────────────────────────────────────────────────────────

def _discover_table_ids(data_dir: Path, require_smp: bool = True) -> list[str]:
    """Return UUIDs that have JSON + QA (and, if ``require_smp``, SMP) artefacts."""
    smp_dir  = data_dir / SMP_SUBDIR
    json_dir = data_dir / JSON_SUBDIR
    qa_dir   = data_dir / QA_SUBDIR
    json_ids = {p.stem for p in json_dir.glob("*.json")}
    qa_ids   = {p.stem for p in qa_dir.glob("*.json")}
    ids = json_ids & qa_ids
    if require_smp:
        ids &= {p.stem for p in smp_dir.glob("*.txt")}
    return sorted(ids)


def _load_questions(qa_path: Path) -> list[dict]:
    """Read the questions[] list from a QUESTIONS_ANSWERS_PER_TABLE file."""
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    return data.get("questions", []) or []


# ── Dataset ───────────────────────────────────────────────────────────────────

class TableEmbedJePADataset(Dataset):
    """
    Loads the Adhesive SMP/JSON/QA directory and produces U-path JEPA samples.

    Each item corresponds to one U-path (variable-length ``header … node … header``)
    and provides
      - smp_seq      [L, d]: ordered walk embeddings (a-headers, node, b-headers)
      - smp_role     [L]   : 1 at the node slot, 0 at header slots (padded 0)
      - smp_bar_seq  [L, d]: reversed walk embeddings
      - smp_bar_role [L]   : role mask aligned with the reversed walk
      - smp_len      int   : number of valid (non-pad) slots
      - query_embeds [1, d]: LLM embedding of concat(pivot_a, node, pivot_b)
      - question_emb [d]   : embedded question (for evaluation)
      - record_idx   int   : index into self.records
    """

    def __init__(
        self,
        data_dir: str | Path,
        model_type: str = "huggingface",
        base_url: Optional[str] = "http://134.184.22.126:10434/",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_records: Optional[int] = None,
        filter_table_id: Optional[str] = None,
        precompute: bool = True,
        embed_batch_size: int = 256,
        truncate_embed_dim: Optional[int] = None,
        cache_embeddings: bool = False,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        cat_qry_template: str = "{pivot_a} ... {pivot_b}?",
        cat_qry_bar_template: str = "{pivot_b} ... {pivot_a}?",
        upath_source: str = "walk",
        hit_threshold: float = 0.5,
        # accepted for signature parity with TRL-model (unused here)
        use_graph_walks: bool = False,
        num_walks: int = 50,
        chunk_size: int = 1,
    ) -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self._data_dir}")

        self._upath_source = str(upath_source).strip().lower()
        if self._upath_source not in ("walk", "json"):
            raise ValueError(f"upath_source must be 'walk' or 'json', got {upath_source!r}")
        self._hit_threshold = float(hit_threshold)

        table_ids = _discover_table_ids(
            self._data_dir, require_smp=(self._upath_source == "walk"))
        if filter_table_id is not None:
            table_ids = [t for t in table_ids if t == filter_table_id]
        print(f"[dataset][discover] {len(table_ids)} tables under {self._data_dir.name}"
              f" [source={self._upath_source}]"
              + (f" [table_id={filter_table_id}]" if filter_table_id else ""))

        # ── Parse U-paths per table (once) + build question records ────────────
        smp_dir  = self._data_dir / SMP_SUBDIR
        json_dir = self._data_dir / JSON_SUBDIR
        qa_dir   = self._data_dir / QA_SUBDIR

        self.records: list[dict] = []
        self._all_upaths: list[list[UPath]] = []   # aligned with self.records
        _table_upaths: dict[str, list[UPath]] = {}

        for tid in table_ids:
            if self._upath_source == "json":
                ups, _ = generate_upaths_from_json(json_dir / f"{tid}.json")
            else:
                ups, _ = generate_upaths_for_table(
                    smp_dir / f"{tid}.txt", json_dir / f"{tid}.json")
            if not ups:
                continue
            _table_upaths[tid] = ups
            questions = _load_questions(qa_dir / f"{tid}.json")
            if not questions:
                # No questions — still keep one placeholder record so the table's
                # U-paths participate in training.
                questions = [{}]
            for qi, q in enumerate(questions):
                _ans_id = q.get("answer_cell_id")
                try:
                    _ans_id = int(_ans_id) if _ans_id is not None and str(_ans_id) != "" else -1
                except (ValueError, TypeError):
                    _ans_id = -1
                _q_raw = str(q.get("question", ""))
                # SW (stop-word-removed) variant read directly from the persisted
                # `question_stopword` field (written by add_stopword_questions.py).
                _q_sw = str(q.get("question_stopword", "") or _q_raw)
                rec = {
                    "table_id":       tid,
                    "id":             f"{tid}#{qi}",
                    "question":       _q_raw,
                    "question_sw":    _q_sw,
                    "question_hit":   "",   # filled during embedding precompute
                    "answer":         str(q.get("answer", "")),
                    "answer_cell_id": _ans_id,
                    "row_header":     str(q.get("row_header", "")),
                    "column_header":  str(q.get("column_header", "")),
                }
                _rid = rec["id"]
                # tag each shared U-path list's record_id lazily at sample build time
                self.records.append(rec)
                self._all_upaths.append(ups)
                if max_records and len(self.records) >= max_records:
                    break
            if max_records and len(self.records) >= max_records:
                break

        print(f"[dataset][load] {len(self.records)} question-records from "
              f"{len(_table_upaths)} tables")

        self._model_type = model_type
        self._base_url   = (base_url or "").rstrip("/")
        self._model_name = model_name
        self._api_key    = api_key
        _tag = model_type.strip().lower()
        self._embedder = (
            None if _tag == "huggingface"
            else get_embedder(model_type, base_url, model_name, api_key)
        )
        self._embed_dim: Optional[int] = None
        self._truncate_embed_dim: Optional[int] = truncate_embed_dim
        self._filter_table_id  = filter_table_id
        self._cache_embeddings = cache_embeddings
        self._embed_cache_dir  = Path(embed_cache_dir) if embed_cache_dir else None
        self._cat_qry_template     = cat_qry_template
        self._cat_qry_bar_template = cat_qry_bar_template

        # ── Build flat sample list ────────────────────────────────────────────
        self._samples: list[tuple[int, UPath]] = [
            (rec_idx, upath)
            for rec_idx, upaths in enumerate(self._all_upaths)
            for upath in upaths
        ]
        _n_before = len(self._samples)

        # ── De-duplicate SMPs by *text content* (not node/header ids) ──────────
        # Two U-paths are the same SMP when their filled-in, role-masked token
        # sequence is identical within the same table.  This collapses the
        # per-question record replication and any distinct cells that share the
        # exact same header/value text, so every unique SMP appears (and is
        # embedded) only once.  Cross-table SMPs are kept separate via table_id
        # so per-table eval / node matching is unaffected.  First occurrence is
        # kept, which carries the table's first-record index and node id.
        _seen_smp: set = set()
        _deduped:  list[tuple[int, UPath]] = []
        for rec_idx, up in self._samples:
            key = (getattr(up, "table_id", ""),
                   tuple(up.seq_texts()), tuple(up.seq_roles()))
            if key in _seen_smp:
                continue
            _seen_smp.add(key)
            _deduped.append((rec_idx, up))
        self._samples = _deduped
        print(f"[dataset][smp_dedup] kept {len(self._samples)} unique-text SMPs "
              f"(removed {_n_before - len(self._samples)} text-duplicates)")

        print(f"[dataset][smp_gen] {len(self._samples)} U-path samples "
              f"from {len(self.records)} records")

        # ── Index of first-record-per-table samples (unique-table training) ───
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
        self._question_sw_cache: Optional[torch.Tensor] = None   # stop-word variant
        self._question_hit_cache: Optional[torch.Tensor] = None  # high-informative-token variant
        self._pad_idx:         int = 0                        # zero-embedding row
        self._seq_idx:         Optional[torch.Tensor] = None  # [N, Lmax] fwd slot text-ids
        self._seq_idx_bar:     Optional[torch.Tensor] = None  # [N, Lmax] reversed slot text-ids
        self._role:            Optional[torch.Tensor] = None  # [N, Lmax] fwd node-role mask
        self._role_bar:        Optional[torch.Tensor] = None  # [N, Lmax] reversed node-role mask
        self._seq_len:         Optional[torch.Tensor] = None  # [N] valid slot count
        self._qry_cat_idx:     Optional[torch.Tensor] = None  # [N]
        self._qry_bar_cat_idx: Optional[torch.Tensor] = None  # [N]
        self._rec_idx:         Optional[torch.Tensor] = None  # [N]

        if precompute:
            self._precompute_embeddings(embed_batch_size)

    # ── Embedding helpers ─────────────────────────────────────────────────────

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            if self._embedder is not None:
                _native = len(self._embedder.embed_query("probe"))
            else:
                _native = len(self._embed_batch_chunk(["probe"])[0])
            self._embed_dim = self._dim_cut(_native)
        return self._embed_dim

    def _embed_batch_chunk(self, texts: list[str]) -> list[list[float]]:
        """Embed one chunk via the backend's native batch API (see TRL-model)."""
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
                return resp.json()
            else:
                from sentence_transformers import SentenceTransformer
                if not hasattr(self, "_st_model"):
                    self._st_model = SentenceTransformer(model)
                vecs = self._st_model.encode(
                    texts, batch_size=len(texts), show_progress_bar=False)
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
            chunk = texts[start: start + batch_size]
            all_vecs.extend(self._embed_batch_chunk(chunk))
            done = min(start + batch_size, n)
            print(f"\r[dataset][embed] {done}/{n}  ({100 * done / n:.1f}%)", end="", flush=True)
        print()
        tensor = torch.tensor(all_vecs, dtype=torch.float32)
        self._embed_dim = tensor.shape[1]
        return tensor

    def _dim_cut(self, current_dim: int) -> int:
        """Effective embedding width: native dim when no cap is set, else the
        configured ``truncate_embed_dim`` (``embed_dim`` in config)."""
        if self._truncate_embed_dim is None:
            return int(current_dim)
        return min(int(current_dim), int(self._truncate_embed_dim))

    def _fmt(self, up: UPath) -> dict:
        return dict(pivot_a=up.col_header_a, node_a=up.cell_value,
                    node_b=up.cell_value,   pivot_b=up.col_header_b)

    def _precompute_embeddings(self, batch_size: int) -> None:
        """Bulk-embed every unique node/query/question text, build index tensors."""
        _model_slug = (self._model_name or self._model_type).replace("/", "-").replace("\\", "-")
        _cache_dir  = self._embed_cache_dir if self._embed_cache_dir else self._data_dir
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _stem       = f"adhesive_{self._filter_table_id}" if self._filter_table_id else "adhesive"
        _hit_tag    = f"hit{str(self._hit_threshold).replace('.', 'p')}"
        _prefix     = f"{_stem}_{_model_slug}_{self._upath_source}_useqsw{_hit_tag}_dim_"
        _target_dim = self._truncate_embed_dim

        # ── Find smallest existing cache whose dim >= target_dim ───────────────
        _best_file: Optional[Path] = None
        _best_dim:  Optional[int]  = None
        for _f in sorted(_cache_dir.glob(f"{_prefix}*.embed_cache.pt")):
            try:
                _dim_val = int(_f.name[len(_prefix):].split(".embed_cache.pt")[0])
            except ValueError:
                continue
            if _target_dim is None or _dim_val >= _target_dim:
                if _best_dim is None or _dim_val < _best_dim:
                    _best_dim, _best_file = _dim_val, _f

        print(f"[dataset][cache] looking for cache files with prefix {_prefix} …")
        if _best_file is not None:
            print(f"[dataset][cache] loading from {_best_file.name}  (stored dim={_best_dim})")
            _ckpt = torch.load(_best_file, map_location="cpu", weights_only=False)
            self._embed_cache     = _ckpt["embed_cache"]
            self._question_cache  = _ckpt["question_cache"]
            self._question_sw_cache = _ckpt.get("question_sw_cache", _ckpt["question_cache"])
            self._question_hit_cache = _ckpt.get("question_hit_cache", _ckpt["question_cache"])
            self._text_to_idx     = _ckpt["text_to_idx"]
            self._pad_idx         = _ckpt["pad_idx"]
            self._seq_idx         = _ckpt["seq_idx"]
            self._seq_idx_bar     = _ckpt["seq_idx_bar"]
            self._role            = _ckpt["role"]
            self._role_bar        = _ckpt["role_bar"]
            self._seq_len         = _ckpt["seq_len"]
            self._qry_cat_idx     = _ckpt["qry_cat_idx"]
            self._qry_bar_cat_idx = _ckpt["qry_bar_cat_idx"]
            self._rec_idx         = _ckpt["rec_idx"]
            self._embed_dim       = _ckpt["embed_dim"]
            # Restore the derived HIT question texts onto the records (they are
            # not re-derived when loading from cache).
            _hit_texts = _ckpt.get("question_hit_texts")
            if _hit_texts is not None and len(_hit_texts) == len(self.records):
                for _r, _t in zip(self.records, _hit_texts):
                    _r["question_hit"] = _t
            if _target_dim is not None and self._embed_dim > _target_dim:
                self._embed_cache    = self._embed_cache[:, :_target_dim].contiguous()
                self._question_cache = self._question_cache[:, :_target_dim].contiguous()
                self._question_sw_cache = self._question_sw_cache[:, :_target_dim].contiguous()
                self._question_hit_cache = self._question_hit_cache[:, :_target_dim].contiguous()
                self._embed_dim      = _target_dim
                print(f"[dataset][cache] trimmed in-memory to first {_target_dim} dims")
            print(f"[dataset][cache] loaded  embed_cache={tuple(self._embed_cache.shape)}  "
                  f"embed_dim={self._embed_dim}")
            return

        # ── Compute embeddings ─────────────────────────────────────────────────
        text_to_idx: dict = {}
        for _, up in self._samples:
            for text in up.seq_texts():          # every ordered walk-slot text
                if text not in text_to_idx:
                    text_to_idx[text] = len(text_to_idx)
            _fmt = self._fmt(up)
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
        # Dedicated zero pad embedding appended at the end of the cache so that
        # padded sequence slots gather a harmless all-zero vector (masked out).
        self._pad_idx = self._embed_cache.shape[0]
        self._embed_cache = torch.cat(
            [self._embed_cache, torch.zeros(1, self._embed_cache.shape[1])], dim=0)

        questions = [rec.get("question", "") for rec in self.records]
        print(f"[dataset][embed_questions] {len(questions)} question embeddings...")
        self._question_cache = self._batch_embed(questions, batch_size)

        questions_sw = [rec.get("question_sw", "") or rec.get("question", "")
                        for rec in self.records]
        print(f"[dataset][embed_questions_sw] {len(questions_sw)} stop-word question embeddings...")
        self._question_sw_cache = self._batch_embed(questions_sw, batch_size)

        # ── HIT (high-informative-token) question variant ─────────────────────
        # Score each question token by min-max normalised cosine similarity to
        # its full-question embedding and keep tokens above the model threshold.
        q_tokens = [q.split() for q in questions]
        _tok_to_idx: dict = {}
        for toks in q_tokens:
            for t in toks:
                if t not in _tok_to_idx:
                    _tok_to_idx[t] = len(_tok_to_idx)
        _tok_texts = [""] * len(_tok_to_idx)
        for t, i in _tok_to_idx.items():
            _tok_texts[i] = t
        print(f"[dataset][embed_hit] {len(_tok_texts)} unique question tokens "
              f"(threshold={self._hit_threshold})...")
        _tok_emb = self._batch_embed(_tok_texts, batch_size) if _tok_texts \
            else torch.zeros(0, self._embed_dim or 1)
        questions_hit: list[str] = []
        for _r, (q, toks) in enumerate(zip(questions, q_tokens)):
            if not toks:
                questions_hit.append(q)
                self.records[_r]["question_hit"] = q
                continue
            _idxs = torch.tensor([_tok_to_idx[t] for t in toks], dtype=torch.long)
            _hit = high_informative_query(
                toks, _tok_emb[_idxs], self._question_cache[_r], self._hit_threshold)
            questions_hit.append(_hit)
            self.records[_r]["question_hit"] = _hit
        print(f"[dataset][embed_questions_hit] {len(questions_hit)} HIT question embeddings...")
        self._question_hit_cache = self._batch_embed(questions_hit, batch_size)


        # ── Build variable-length, right-padded sequence index tensors ─────────
        pad  = self._pad_idx
        Lmax = max((up.seq_len for _, up in self._samples), default=1)
        seq_rows, seq_bar_rows, role_rows, role_bar_rows, len_rows = [], [], [], [], []
        qry_cat_rows, qry_bar_cat_rows, rec_list = [], [], []
        for r_idx, up in self._samples:
            f_txt, f_role = up.seq_texts(),     up.seq_roles()
            b_txt, b_role = up.seq_texts_bar(), up.seq_roles_bar()
            L = len(f_txt)
            seq_rows.append(     [text_to_idx[t] for t in f_txt] + [pad] * (Lmax - L))
            seq_bar_rows.append( [text_to_idx[t] for t in b_txt] + [pad] * (Lmax - L))
            role_rows.append(     f_role + [0] * (Lmax - L))
            role_bar_rows.append( b_role + [0] * (Lmax - L))
            len_rows.append(L)
            _fmt = self._fmt(up)
            qry_cat_rows.append(text_to_idx[self._cat_qry_template.format(**_fmt)])
            qry_bar_cat_rows.append(text_to_idx[self._cat_qry_bar_template.format(**_fmt)])
            rec_list.append(r_idx)
        self._seq_idx         = torch.tensor(seq_rows,         dtype=torch.long)
        self._seq_idx_bar     = torch.tensor(seq_bar_rows,     dtype=torch.long)
        self._role            = torch.tensor(role_rows,        dtype=torch.float32)
        self._role_bar        = torch.tensor(role_bar_rows,    dtype=torch.float32)
        self._seq_len         = torch.tensor(len_rows,         dtype=torch.long)
        self._qry_cat_idx     = torch.tensor(qry_cat_rows,     dtype=torch.long)
        self._qry_bar_cat_idx = torch.tensor(qry_bar_cat_rows, dtype=torch.long)
        self._rec_idx         = torch.tensor(rec_list,         dtype=torch.long)

        _actual_dim = int(self._embed_cache.shape[1])
        _cache_file = _cache_dir / f"{_prefix}{_actual_dim}.embed_cache.pt"
        print(f"[dataset][build_index] done  embed_dim={_actual_dim}")
        if self._cache_embeddings:
            print(f"[dataset][cache] saving full-dim ({_actual_dim}) cache to {_cache_file.name} …")
            torch.save({
                "embed_cache":     self._embed_cache,
                "question_cache":  self._question_cache,
                "question_sw_cache": self._question_sw_cache,
                "question_hit_cache": self._question_hit_cache,
                "question_hit_texts": [rec.get("question_hit", "") for rec in self.records],
                "text_to_idx":     self._text_to_idx,
                "pad_idx":         self._pad_idx,
                "seq_idx":         self._seq_idx,
                "seq_idx_bar":     self._seq_idx_bar,
                "role":            self._role,
                "role_bar":        self._role_bar,
                "seq_len":         self._seq_len,
                "qry_cat_idx":     self._qry_cat_idx,
                "qry_bar_cat_idx": self._qry_bar_cat_idx,
                "rec_idx":         self._rec_idx,
                "embed_dim":       _actual_dim,
            }, _cache_file)
            print(f"[dataset][cache] saved   ({_cache_file.stat().st_size / 1024**2:.1f} MB)")

        if _target_dim is not None:
            self._embed_cache    = self._embed_cache[:, :_target_dim].contiguous()
            self._question_cache = self._question_cache[:, :_target_dim].contiguous()
            self._question_sw_cache = self._question_sw_cache[:, :_target_dim].contiguous()
            self._question_hit_cache = self._question_hit_cache[:, :_target_dim].contiguous()
            self._embed_dim      = _target_dim
            print(f"[dataset][truncate] in-memory truncated to first {_target_dim} dims")

    def _embed_live(self, text: str) -> torch.Tensor:
        vecs = self._embed_batch_chunk([text])
        vec = torch.tensor(vecs[0], dtype=torch.float32)
        _cut = self._dim_cut(vec.shape[0])
        if _cut < vec.shape[0]:
            vec = vec[:_cut].contiguous()
        self._embed_dim = vec.shape[0]
        return vec

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_st_model", None)
        state.pop("_embedder", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, up = self._samples[idx]

        if self._seq_idx is not None:
            smp_seq          = self._embed_cache[self._seq_idx[idx]]                       # [L, d]
            smp_bar_seq      = self._embed_cache[self._seq_idx_bar[idx]]                   # [L, d]
            smp_role         = self._role[idx]                                            # [L]
            smp_bar_role     = self._role_bar[idx]                                        # [L]
            smp_len          = self._seq_len[idx]                                         # scalar
            query_embeds     = self._embed_cache[self._qry_cat_idx[idx]].unsqueeze(0)     # [1, d]
            query_bar_embeds = self._embed_cache[self._qry_bar_cat_idx[idx]].unsqueeze(0) # [1, d]
            question_emb     = self._question_cache[self._rec_idx[idx]]                   # [d]
        else:
            f_txt, f_role = up.seq_texts(),     up.seq_roles()
            b_txt, b_role = up.seq_texts_bar(), up.seq_roles_bar()
            smp_seq      = torch.stack([self._embed_live(t) for t in f_txt])   # [L, d]
            smp_bar_seq  = torch.stack([self._embed_live(t) for t in b_txt])   # [L, d]
            smp_role     = torch.tensor(f_role, dtype=torch.float32)
            smp_bar_role = torch.tensor(b_role, dtype=torch.float32)
            smp_len      = torch.tensor(len(f_txt), dtype=torch.long)
            _fmt = self._fmt(up)
            query_embeds     = self._embed_live(self._cat_qry_template.format(**_fmt)).unsqueeze(0)
            query_bar_embeds = self._embed_live(self._cat_qry_bar_template.format(**_fmt)).unsqueeze(0)
            question_emb     = self._embed_live(self.records[rec_idx].get("question", ""))

        return {
            "smp_seq":          smp_seq,
            "smp_bar_seq":      smp_bar_seq,
            "smp_role":         smp_role,
            "smp_bar_role":     smp_bar_role,
            "smp_len":          smp_len,
            "query_embeds":     query_embeds,
            "query_bar_embeds": query_bar_embeds,
            "question_emb":     question_emb,
            "record_idx":       rec_idx,
            "upath":            up,
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
    Collate variable-length U-path samples into a right-padded JEPA batch.

    All sequences are padded to the batch-max walk length.  ``smp_pad_mask``
    (1 = valid slot, 0 = pad) drives the model's attention + loss masking, and
    ``smp_role`` / ``smp_bar_role`` mark the single node slot in each orientation.
    """
    B    = len(batch)
    lens = torch.stack([
        b["smp_len"] if torch.is_tensor(b["smp_len"]) else torch.tensor(b["smp_len"])
        for b in batch
    ]).long()
    Lmax = int(max(b["smp_seq"].shape[0] for b in batch))
    d    = int(batch[0]["smp_seq"].shape[1])

    def _pad_seq(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == Lmax:
            return x
        return torch.cat([x, x.new_zeros(Lmax - x.shape[0], d)], dim=0)

    def _pad_role(r: torch.Tensor) -> torch.Tensor:
        if r.shape[0] == Lmax:
            return r
        return torch.cat([r, r.new_zeros(Lmax - r.shape[0])], dim=0)

    smp_embeds       = torch.stack([_pad_seq(b["smp_seq"])      for b in batch])  # [B, L, d]
    smp_bar_embeds   = torch.stack([_pad_seq(b["smp_bar_seq"])  for b in batch])  # [B, L, d]
    smp_role         = torch.stack([_pad_role(b["smp_role"])     for b in batch])  # [B, L]
    smp_bar_role     = torch.stack([_pad_role(b["smp_bar_role"]) for b in batch])  # [B, L]
    query_embeds     = torch.stack([b["query_embeds"]     for b in batch])  # [B, 1, d]
    query_bar_embeds = torch.stack([b["query_bar_embeds"] for b in batch])  # [B, 1, d]
    question_embs    = torch.stack([b["question_emb"]     for b in batch])  # [B, d]
    record_indices   = torch.tensor([b["record_idx"] for b in batch], dtype=torch.long)

    ar = torch.arange(Lmax)
    smp_pad_mask = (ar.unsqueeze(0) < lens.unsqueeze(1)).float()  # [B, L] — 1 valid, 0 pad

    return {
        "smp_embeds":       smp_embeds,
        "smp_role":         smp_role,
        "smp_pad_mask":     smp_pad_mask,
        "smp_bar_embeds":   smp_bar_embeds,
        "smp_bar_role":     smp_bar_role,
        "query_embeds":     query_embeds,
        "query_bar_embeds": query_bar_embeds,
        "question_embeds":  question_embs,
        "record_indices":   record_indices,
    }


# ── PyTorch Lightning DataModule ──────────────────────────────────────────────

class TableEmbedJePADataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the Adhesive TableEmbedJePADataset."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 0,
        model_type: str = "huggingface",
        base_url: Optional[str] = "http://134.184.22.126:10434/",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_records: Optional[int] = None,
        filter_table_id: Optional[str] = None,
        unique_tables_only: bool = False,
        precompute: bool = True,
        embed_batch_size: int = 256,
        truncate_embed_dim: Optional[int] = None,
        cache_embeddings: bool = False,
        embed_cache_dir: Optional[Union[str, Path]] = None,
        cat_qry_template: str = "{pivot_a} ... {pivot_b}?",
        cat_qry_bar_template: str = "{pivot_b} ... {pivot_a}?",
        upath_source: str = "walk",
        hit_threshold: float = 0.5,
        # signature parity (unused)
        use_graph_walks: bool = False,
        num_walks: int = 50,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.data_dir    = data_dir
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
            truncate_embed_dim=truncate_embed_dim,
            cache_embeddings=cache_embeddings,
            embed_cache_dir=embed_cache_dir,
            cat_qry_template=cat_qry_template,
            cat_qry_bar_template=cat_qry_bar_template,
            upath_source=upath_source,
            hit_threshold=hit_threshold,
        )
        self._dataset: Optional[TableEmbedJePADataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._dataset is None:
            self._dataset = TableEmbedJePADataset(
                data_dir=self.data_dir, **self._ds_kwargs)

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
            _idxs = self._dataset._unique_table_train_idxs
            print(f"[datamodule] unique_tables_only — training on "
                  f"{len(_idxs):,} samples (full dataset: {len(self._dataset):,})")
            return DataLoader(
                self._dataset,
                sampler=SubsetRandomSampler(_idxs),
                **_common,
            )
        return DataLoader(self._dataset, shuffle=True, **_common)

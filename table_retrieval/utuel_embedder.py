"""
utuel_embedder.py
UTUEL / TableEmbedJePA embedder for table retrieval benchmarking.

Table embedding
───────────────
For each unique table:
  1. Generate all U-paths (SMP sequences) via generate_u_paths_flat.
  2. Batch-embed all unique node texts with the base sentence-transformer.
  3. Build SMP tensor  [B, 4, d_in]: [pivot_a, node_a, node_b, pivot_b]
     Build SMP_bar     [B, 4, d_in]: reversed → [pivot_b, node_b, node_a, pivot_a]
  4. Prepend learned CLS token and pass through the online transformer_encoder.
  5. Extract:
       node_a  = enc_smp[:, 2, :]    (position 2 = node_a after CLS)
       node_b  = enc_bar[:, 2, :]    (position 2 = node_b in SMP_bar after CLS)
  6. Mean-pool  [node_a_0..N, node_b_0..N]  → L2-normalise → table embedding.

Query embedding
───────────────
  question  →  base ST embed  →  input_projection  →  transformer_encoder
            →  position 0  →  L2-normalise  →  query embedding.

  (Mirrors the evaluation in TRL-model/dev_check.ipynb §7.0.)

Without a checkpoint
─────────────────────
  Raises FileNotFoundError — a checkpoint (.ckpt) is required.

YAML entry
──────────
  embedder:
    type: custom
    class: UTUELTableEmbedder
    model_name: UTUEL/all-MiniLM-L6-v2
    checkpoint_dir: TRL-model/checkpoints/all-MiniLM-L6-v2/<run_id>
    batch_size: 64
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from abstract import TableRetrieverBase, register_embedder  # type: ignore[import]


@register_embedder
class UTUELTableEmbedder(TableRetrieverBase):
    """
    TableEmbedJePA-based embedder.  See module docstring for details.

    Parameters (from embedder_cfg dict)
    ────────────────────────────────────
    checkpoint_dir : str   path relative to project_root
    model_name     : str   optional label (default: UTUEL/<ckpt_dir_name>)
    batch_size     : int   U-path batch size for the encoder (default 128)
    pool           : str   "mean" (default) or "max" — per-table node pooling
    """

    def __init__(self, embedder_cfg: dict, project_root: str | Path = "."):
        self.project_root  = Path(project_root)
        self.batch_size    = int(embedder_cfg.get("batch_size", 128))
        self.pool_mode     = embedder_cfg.get("pool", "mean")
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = self.project_root / embedder_cfg["checkpoint_dir"]

        # ── Load run_config.yaml ──────────────────────────────────────────────
        run_cfg_path = ckpt_dir / "run_config.yaml"
        if not run_cfg_path.exists():
            raise FileNotFoundError(f"run_config.yaml not found in {ckpt_dir}")
        with run_cfg_path.open(encoding="utf-8") as f:
            self.run_cfg = yaml.safe_load(f)

        # ── Require a .pt ──────────────────────────────────────────────────────
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"[UTUEL] No checkpoint (.pt) found in {ckpt_dir}.\n"
                f"Train a model first or point checkpoint_dir at a directory "
                f"that contains a .pt file."
            )

        self._load_trl_model(ckpt_files[-1])
        self.dim = self.run_cfg["model"]["hidden_size"]

        # Full path to the node-embedding cache file.
        # Resolved from embedder_cfg["cache_file"] (relative to project_root)
        # when provided; otherwise defaults to <checkpoint_dir>/node_embed_cache.pt.
        _cache_file_cfg = embedder_cfg.get("cache_file")
        self._node_cache_file: Path = (
            self.project_root / _cache_file_cfg
            if _cache_file_cfg
            else ckpt_dir / "node_embed_cache.pt"
        )

        _default_label = f"UTUEL/{ckpt_dir.parent.name}+pt"
        self.label = embedder_cfg.get("model_name", _default_label)
        if not self.label.endswith("+pt"):
            self.label = self.label + "+pt"

        print(f"  [UTUEL] label={self.label}  dim={self.dim}  device={self.device}")

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_trl_model(self, ckpt_path: Path) -> None:
        trl_dir = self.project_root / "TRL-model"
        if str(trl_dir) not in sys.path:
            sys.path.insert(0, str(trl_dir))

        from config import TableEmbedJePAConfig  # type: ignore[import]
        from model  import TableEmbedJePA         # type: ignore[import]

        mcfg        = self.run_cfg["model"]
        hidden_size = mcfg["hidden_size"]
        embed_dim   = self.run_cfg.get("embedder", {}).get("embed_dim")
        if embed_dim is None:
            raise KeyError(
                "run_config.yaml is missing embedder.embed_dim. "
                "Re-train or add the key manually."
            )

        cfg = TableEmbedJePAConfig(
            hidden_size                  = hidden_size,
            num_hidden_layers            = mcfg["num_layers"],
            num_attention_heads          = mcfg["num_heads"],
            intermediate_size            = mcfg.get("intermediate_size") or hidden_size * 4,
            hidden_dropout_prob          = mcfg["hidden_dropout"],
            attention_probs_dropout_prob = mcfg["attention_dropout"],
            layer_norm_eps               = mcfg["layer_norm_eps"],
            tempeture                    = mcfg["temperature"],
            beta                         = mcfg["beta"],
            embedding_dim                = embed_dim,
        )

        loss_cfg = self.run_cfg.get("loss", {})

        # Instantiate model then load weights directly — avoids Lightning's
        # checkpoint migration which requires a 'pytorch-lightning_version' key.
        model = TableEmbedJePA(
            config          = cfg,
            jepa_weight     = loss_cfg.get("jepa",     1.0),
            jepa_bar_weight = loss_cfg.get("jepa_bar", 1.0),
            local_weight    = loss_cfg.get("local",    1.0),
            global_weight   = loss_cfg.get("global",   1.0),
            ablate_proj     = mcfg.get("ablate_proj", False),
        )
        raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = raw.get("state_dict", raw)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  [UTUEL] missing keys ({len(result.missing_keys)}): {result.missing_keys[:3]}")
        self.trl_model = model.eval().to(self.device)
        print(f"  [UTUEL] Loaded checkpoint: {ckpt_path.name}")

    # ── Node embedding cache ──────────────────────────────────────────────────

    def _get_node_embeddings_cached(self, all_texts: list[str]) -> np.ndarray:
        """
        Embed ``all_texts`` with the base model, using a persistent on-disk cache.

        Cache file
        ──────────
        Saved at ``self._node_cache_file`` using the same keys as
        TRL-model/dataset.py so a training-run cache can also be reused:
          ``embed_cache``  [N, d]  float32
          ``text_to_idx``  dict[str, int]
          ``embed_dim``    int

        Behaviour
        ─────────
        * Full hit   — all texts found → returns immediately, no network/GPU call.
        * Partial hit — missing texts are embedded and merged into the cache.
        * Miss        — all texts embedded and saved fresh.

        Returns [len(all_texts), d_in] float32 ndarray in the order of all_texts.
        """
        cache_file = self._node_cache_file

        # ── Try loading existing cache ────────────────────────────────────────
        cached_t2i:  dict[str, int]    = {}
        cached_embs: torch.Tensor | None = None

        if cache_file.exists():
            print(f"  [UTUEL] loading node cache: {cache_file}")
            ckpt         = torch.load(cache_file, map_location="cpu", weights_only=False)
            cached_t2i   = ckpt.get("text_to_idx", {})
            cached_embs  = ckpt.get("embed_cache")   # same key as training cache
            print(f"  [UTUEL] cache: {len(cached_t2i):,} texts  dim={ckpt.get('embed_dim')}")

        # ── Partition into cached / uncached ──────────────────────────────────
        need_embed = [t for t in all_texts if t not in cached_t2i]

        if need_embed:
            print(f"  [UTUEL] embedding {len(need_embed):,} uncached node texts …")
            new_np     = self._embed_texts(need_embed)           # [M, d_in]
            new_tensor = torch.tensor(new_np, dtype=torch.float32)

            if cached_embs is not None:
                start      = len(cached_t2i)
                merged_t2i = dict(cached_t2i)
                for i, t in enumerate(need_embed):
                    merged_t2i[t] = start + i
                merged_embs = torch.cat([cached_embs, new_tensor], dim=0)
            else:
                merged_t2i  = {t: i for i, t in enumerate(need_embed)}
                merged_embs = new_tensor

            # ── Persist updated cache ─────────────────────────────────────────
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "embed_cache": merged_embs,
                    "text_to_idx": merged_t2i,
                    "embed_dim":   int(merged_embs.shape[1]),
                },
                cache_file,
            )
            print(f"  [UTUEL] saved node cache: {cache_file.name}  "
                  f"({len(merged_t2i):,} texts  dim={merged_embs.shape[1]})")
        else:
            merged_t2i  = cached_t2i
            merged_embs = cached_embs
            print(f"  [UTUEL] all {len(all_texts):,} node texts served from cache  (no embed call)")

        # ── Assemble result in all_texts order ────────────────────────────────
        indices = [merged_t2i[t] for t in all_texts]
        return merged_embs[indices].numpy().astype(np.float32)

    # ── Text embedding helpers ────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts with the base model (lazy-loaded). Returns [N, d_in] unnormalised."""
        if not hasattr(self, "_base_embedder"):
            emb_cfg    = self.run_cfg.get("embedder", {})
            model_type = emb_cfg.get("model_type", "huggingface")
            model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if model_type == "ollama":
                from langchain_ollama import OllamaEmbeddings  # type: ignore[import-untyped]
                base_url = emb_cfg.get("base_url", "http://localhost:11434")
                self._base_embedder = ("ollama", OllamaEmbeddings(
                    base_url = base_url.rstrip("/"),
                    model    = model_name,
                ))
            else:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
                self._base_embedder = ("st", SentenceTransformer(model_name))

        kind, model = self._base_embedder
        if kind == "ollama":
            vecs: list[list[float]] = []
            for i in range(0, len(texts), self.batch_size):
                vecs.extend(model.embed_documents(texts[i : i + self.batch_size]))
            return np.array(vecs, dtype=np.float32)
        else:
            return model.encode(
                texts,
                batch_size           = self.batch_size,
                show_progress_bar    = True,
                normalize_embeddings = False,
                convert_to_numpy     = True,
            ).astype(np.float32)

    # ── Per-chunk encoder pass ────────────────────────────────────────────────

    def _encode_smp_chunk(
        self,
        smp_chunk: torch.Tensor,     # [B, 4, d_in]  [pivot_a, node_a, node_b, pivot_b]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pass one SMP chunk and its SMP_bar through the online transformer_encoder.

        Returns (node_a_embs, node_b_embs) both [B, d_out], L2-normalised.
        SMP_bar is obtained by reversing the 4-node order: [3, 2, 1, 0].
        """
        B = smp_chunk.shape[0]
        m = self.trl_model
        smp_bar = smp_chunk[:, [3, 2, 1, 0], :]       # reverse → SMP_bar

        with torch.no_grad():
            smp_proj = m.input_projection(smp_chunk)      # [B, 4, d_out]
            bar_proj = m.input_projection(smp_bar)        # [B, 4, d_out]
            cls      = m.input_projection(m.cls_token).expand(B, -1, -1)  # [B, 1, d_out]

            # Prepend CLS → sequence length 5
            smp_in = torch.cat([cls, smp_proj], dim=1)    # [B, 5, d_out]
            bar_in = torch.cat([cls, bar_proj], dim=1)    # [B, 5, d_out]

            enc_smp, _, _ = m.transformer_encoder(smp_in)  # [B, 5, d_out]
            enc_bar, _, _ = m.transformer_encoder(bar_in)  # [B, 5, d_out]

            # Position 2 after CLS = node_a (SMP) / node_b (SMP_bar)
            node_a = F.normalize(enc_smp[:, 2, :], dim=-1)   # [B, d_out]
            node_b = F.normalize(enc_bar[:, 2, :], dim=-1)   # [B, d_out]

        return node_a, node_b

    # ── Pool a collection of node embeddings into one table vector ────────────

    def _pool(self, node_embs: torch.Tensor) -> np.ndarray:
        """Mean- or max-pool [N, d] → [d], L2-normalised."""
        if self.pool_mode == "max":
            vec, _ = node_embs.max(dim=0)
        else:
            vec = node_embs.mean(dim=0)
        return F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()

    # ── Aggregation / TSR helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_aggregated_embeddings(
        na: torch.Tensor,   # [n, d] L2-normalised node_a embeddings (CPU)
        ups: list,           # UPath objects for this table
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate node_a embeddings to column, row, and table level.

        col level — one L2-normalised mean vector per unique col_idx_a
        row level — one L2-normalised mean vector per unique tbl_row
        tbl level — one L2-normalised mean vector for the whole table

        Returns (col_embs [n_cols, d], row_embs [n_rows, d], tbl_emb [1, d]).
        Mirrors compute_aggregated_embeddings() in TRL-model/train.py.
        """
        col_groups: dict[int, list[int]] = defaultdict(list)
        row_groups: dict[int, list[int]] = defaultdict(list)
        for i, up in enumerate(ups):
            col_groups[up.col_idx_a].append(i)
            row_groups[up.tbl_row].append(i)

        col_ids  = sorted(col_groups.keys())
        row_ids  = sorted(row_groups.keys())
        col_embs = F.normalize(
            torch.stack([na[col_groups[c]].mean(dim=0) for c in col_ids]), dim=-1)
        row_embs = F.normalize(
            torch.stack([na[row_groups[r]].mean(dim=0) for r in row_ids]), dim=-1)
        tbl_emb  = F.normalize(na.mean(dim=0, keepdim=True), dim=-1)
        return col_embs, row_embs, tbl_emb

    @staticmethod
    def _top_score_rank(
        g_embs:  torch.Tensor,   # [N, d]  all node embeddings (on device)
        g_tids:  list[str],       # [N]     table_id per embedding
        q_norm:  torch.Tensor,    # [d,]    L2-normalised query embedding (on device)
        top_k:   int = 2000,
        top_n:   int = 20,
    ) -> list[str]:
        """
        Rank tables by max cosine score of any of their nodes to the query.

        Computes cosine similarities over all N nodes, takes the top-k by score,
        then builds an ordered set of unique table_ids by first-appearance
        (= descending max-score per table).  Stops once top_n tables are found.

        Mirrors _top_score_rank() in TRL-model/train.py.
        """
        sims       = (g_embs @ q_norm.unsqueeze(-1)).squeeze(-1)  # [N]
        k          = min(top_k, sims.shape[0])
        top_sc, top_ix = sims.topk(k)
        seen: dict[str, float] = {}
        for j, i in enumerate(top_ix.tolist()):
            t = g_tids[i]
            if t not in seen:
                seen[t] = float(top_sc[j])
                if len(seen) == top_n:
                    break
        return list(seen.keys())

    def compute_tsr_metrics(
        self,
        query_embeddings: np.ndarray,       # [Q, d]
        gold_ids:         list[str],         # [Q]
        k_values:         list[int] | None = None,
        top_k_table:      int = 2000,
        mrr_depth:        int = 2000,
    ) -> dict[str, dict]:
        """
        Compute Top-Score-Rank (TSR) table retrieval metrics across four spaces:
          ``tsr``     — global node_a + node_b  (matches training eval)
          ``col_tsr`` — col-aggregated node_a
          ``row_tsr`` — row-aggregated node_a
          ``tbl_tsr`` — table-aggregated node_a

        Must be called *after* ``encode_table_corpus_variants()`` so that the
        global node indexes (``self._global_*_embs`` / ``self._global_*_tids``)
        are populated.

        Returns a dict ``{space_name: {"Hit@k": float, ...}}`` for each space.
        """
        if not hasattr(self, "_global_node_embs"):
            raise RuntimeError(
                "compute_tsr_metrics() requires encode_table_corpus_variants() "
                "to be called first so the global node index is built."
            )
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        spaces = {
            "tsr":     (self._global_node_embs, self._global_node_tids),
            "col_tsr": (self._global_col_embs,  self._global_col_tids),
            "row_tsr": (self._global_row_embs,  self._global_row_tids),
            "tbl_tsr": (self._global_tbl_embs,  self._global_tbl_tids),
        }
        # Move to inference device once
        spaces_dev = {k: (e.to(self.device), t) for k, (e, t) in spaces.items()}

        hits = {sp: {k: 0 for k in k_values} for sp in spaces}
        rr   = {sp: [] for sp in spaces}   # reciprocal ranks for MRR
        n    = len(gold_ids)
        # Collect enough unique tables to satisfy both Hit@k and MRR
        hit_depth = max(k_values)       # minimum unique tables needed for Hit@k
        top_n     = max(hit_depth, mrr_depth)

        q_tensor = torch.tensor(
            query_embeddings, dtype=torch.float32, device=self.device,
        )   # [Q, d]

        for q_idx, gold_tid in enumerate(gold_ids):
            q_norm = q_tensor[q_idx]   # [d]
            for sp, (g_embs, g_tids) in spaces_dev.items():
                ranked = self._top_score_rank(g_embs, g_tids, q_norm, top_k_table, top_n)
                for k in k_values:
                    if gold_tid in ranked[:k]:
                        hits[sp][k] += 1
                # MRR: reciprocal rank within the top mrr_depth unique tables
                mrr_ranked = ranked[:mrr_depth]
                try:
                    rank = mrr_ranked.index(gold_tid) + 1   # 1-based
                    rr[sp].append(1.0 / rank)
                except ValueError:
                    rr[sp].append(0.0)
            if (q_idx + 1) % 200 == 0 or q_idx == n - 1:
                print(f"  [UTUEL] TSR eval: {q_idx + 1}/{n}", end="\r", flush=True)
        print()

        return {
            sp: {
                "MRR": round(sum(rr[sp]) / n, 4),
                **{f"Hit@{k}": round(hits[sp][k] / n, 4) for k in k_values},
            }
            for sp in spaces
        }

    # ── Abstract interface ────────────────────────────────────────────────────

    def encode_table_corpus_variants(self, records: list[dict]) -> dict[str, np.ndarray]:
        """
        Encode all unique tables using the U-path / SMP pipeline.

        Returns a dict with three pooling strategies (all ``[T, D]`` float32):
          ``node_a`` — L2-norm of mean(node_a) per table
          ``node_b`` — L2-norm of mean(node_b) per table
          ``both``   — mean of the two L2-normalised means, re-normalised

        Processing is fully batched across all tables for efficiency:
          - All U-path node texts from every table are embedded in one bulk call.
          - All SMP tensors are stacked and passed through the encoder in chunks.
          - Per-table pooling is applied using pre-computed index ranges.
        """
        trl_dir = self.project_root / "TRL-model"
        if str(trl_dir) not in sys.path:
            sys.path.insert(0, str(trl_dir))
        from smp import generate_u_paths_flat, UPath  # type: ignore[import]

        # ── Step 1: generate U-paths per table ───────────────────────────────
        table_ranges: list[tuple[int, int]] = []   # (start, end) into flat_upaths
        flat_upaths:  list[UPath]           = []

        for rec in records:
            header   = [str(h) for h in rec.get("header", [])]
            rows     = [[str(c) for c in row] for row in rec.get("rows", [])]
            upaths   = generate_u_paths_flat(header, rows)
            start    = len(flat_upaths)
            flat_upaths.extend(upaths)
            table_ranges.append((start, len(flat_upaths)))

        total_upaths = len(flat_upaths)
        print(f"  [UTUEL] {len(records)} tables → {total_upaths} U-paths")

        # Pre-allocate result matrices for three pooling variants
        T = len(records)
        _zero = np.zeros((T, self.dim), dtype=np.float32)

        if total_upaths == 0:
            return {"node_a": _zero.copy(), "node_b": _zero.copy(), "both": _zero.copy()}

        # ── Step 2: collect unique node texts + bulk-embed ────────────────────
        text_to_idx: dict[str, int] = {}
        for up in flat_upaths:
            for t in (up.col_header_a, up.cell_value_a, up.cell_value_b, up.col_header_b):
                if t not in text_to_idx:
                    text_to_idx[t] = len(text_to_idx)

        all_texts  = list(text_to_idx.keys())
        print(f"  [UTUEL] embedding {len(all_texts):,} unique node texts …")
        node_embs_np = self._get_node_embeddings_cached(all_texts)  # [N_unique, d_in]

        # ── Step 3: build SMP tensor [total_upaths, 4, d_in] ─────────────────
        pa_idx = np.array([text_to_idx[up.col_header_a]  for up in flat_upaths])
        na_idx = np.array([text_to_idx[up.cell_value_a]  for up in flat_upaths])
        nb_idx = np.array([text_to_idx[up.cell_value_b]  for up in flat_upaths])
        pb_idx = np.array([text_to_idx[up.col_header_b]  for up in flat_upaths])

        smp_np = np.stack([                  # [T, 4, d_in]
            node_embs_np[pa_idx],            # pivot_a
            node_embs_np[na_idx],            # node_a
            node_embs_np[nb_idx],            # node_b
            node_embs_np[pb_idx],            # pivot_b
        ], axis=1)

        # ── Step 4: encoder pass in batches, collect per-upath node embeddings─
        all_node_a: list[torch.Tensor] = []
        all_node_b: list[torch.Tensor] = []

        for start in range(0, total_upaths, self.batch_size):
            chunk_np = smp_np[start : start + self.batch_size]
            chunk    = torch.tensor(chunk_np, dtype=torch.float32, device=self.device)
            na, nb   = self._encode_smp_chunk(chunk)
            all_node_a.append(na.cpu())
            all_node_b.append(nb.cpu())
            done = min(start + self.batch_size, total_upaths)
            print(f"  [UTUEL] encode U-paths: {done}/{total_upaths}", end="\r", flush=True)
        print()

        node_a_all = torch.cat(all_node_a, dim=0)   # [total_upaths, d_out]
        node_b_all = torch.cat(all_node_b, dim=0)   # [total_upaths, d_out]

        # ── Step 5: build global node index (unpooled, for TSR eval) ──────────
        #   Mirrors Phase 1 in TRL-model/train.py:evaluate_model().
        #   node_a + node_b from every table are concatenated into one index;
        #   col/row/table aggregates are built from node_a only.
        _glob_node_parts: list[torch.Tensor] = []
        _glob_node_tids:  list[str]          = []
        _glob_col_parts:  list[torch.Tensor] = []
        _glob_col_tids:   list[str]          = []
        _glob_row_parts:  list[torch.Tensor] = []
        _glob_row_tids:   list[str]          = []
        _glob_tbl_parts:  list[torch.Tensor] = []
        _glob_tbl_tids:   list[str]          = []

        for tbl_idx, (s, e) in enumerate(table_ranges):
            if s == e:
                continue
            tid      = str(records[tbl_idx].get("table_id", tbl_idx))
            na_chunk = node_a_all[s:e]    # [n_paths, d_out]  CPU
            nb_chunk = node_b_all[s:e]    # [n_paths, d_out]  CPU
            n_up     = e - s

            # node_a and node_b together span the global node search space
            _glob_node_parts.append(na_chunk)
            _glob_node_parts.append(nb_chunk)
            _glob_node_tids.extend([tid] * (n_up * 2))

            # Col / row / table aggregation from node_a only
            ups_tbl = flat_upaths[s:e]
            col_embs, row_embs, tbl_emb = self._compute_aggregated_embeddings(
                na_chunk, ups_tbl,
            )
            _glob_col_parts.append(col_embs)
            _glob_col_tids.extend([tid] * col_embs.shape[0])
            _glob_row_parts.append(row_embs)
            _glob_row_tids.extend([tid] * row_embs.shape[0])
            _glob_tbl_parts.append(tbl_emb)
            _glob_tbl_tids.append(tid)

        # Store as instance attributes; evaluation via compute_tsr_metrics()
        self._global_node_embs = torch.cat(_glob_node_parts, dim=0)   # [N_nodes, d]
        self._global_node_tids = _glob_node_tids
        self._global_col_embs  = torch.cat(_glob_col_parts,  dim=0)   # [N_cols,  d]
        self._global_col_tids  = _glob_col_tids
        self._global_row_embs  = torch.cat(_glob_row_parts,  dim=0)   # [N_rows,  d]
        self._global_row_tids  = _glob_row_tids
        self._global_tbl_embs  = torch.cat(_glob_tbl_parts,  dim=0)   # [N_tbls,  d]
        self._global_tbl_tids  = _glob_tbl_tids
        print(f"  [UTUEL] Global node index — "
              f"node={self._global_node_embs.shape[0]:,}  "
              f"col={self._global_col_embs.shape[0]:,}  "
              f"row={self._global_row_embs.shape[0]:,}  "
              f"tbl={self._global_tbl_embs.shape[0]:,}")

        # ── Step 6: per-table pooling (three variants) ────────────────────────
        #   node_a : L2-norm(mean(node_a))              — mirrors train.py tbl for SMP
        #   node_b : L2-norm(mean(node_b))              — mirrors train.py tbl for SMP_bar
        #   both   : mean of the two unit vectors, re-normalised
        table_embs_a    = np.zeros((T, self.dim), dtype=np.float32)
        table_embs_b    = np.zeros((T, self.dim), dtype=np.float32)
        table_embs_both = np.zeros((T, self.dim), dtype=np.float32)

        for tbl_idx, (s, e) in enumerate(table_ranges):
            if s == e:
                continue   # table had no U-paths — stays zero
            na_chunk = node_a_all[s:e]                          # [n_paths, d_out]
            nb_chunk = node_b_all[s:e]                          # [n_paths, d_out]
            mean_a = F.normalize(na_chunk.mean(dim=0, keepdim=True), dim=-1)  # [1, d_out]
            mean_b = F.normalize(nb_chunk.mean(dim=0, keepdim=True), dim=-1)  # [1, d_out]
            table_embs_a[tbl_idx]    = mean_a.squeeze(0).cpu().numpy()
            table_embs_b[tbl_idx]    = mean_b.squeeze(0).cpu().numpy()
            table_embs_both[tbl_idx] = self._pool(torch.cat([mean_a, mean_b], dim=0))

        return {"node_a": table_embs_a, "node_b": table_embs_b, "both": table_embs_both}

    def encode_table_corpus(self, records: list[dict]) -> np.ndarray:
        """Backward-compatible wrapper — returns the ``both`` pooling variant."""
        return self.encode_table_corpus_variants(records)["both"]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """
        Encode queries.

        question  →  base ST embed  →  input_projection  →  transformer_encoder
                  →  position 0   →  L2-normalise
        """
        base_vecs = self._embed_texts(queries)    # [Q, d_in]

        all_out: list[np.ndarray] = []
        m = self.trl_model

        for i in range(0, len(base_vecs), self.batch_size):
            raw = torch.tensor(
                base_vecs[i : i + self.batch_size],
                dtype=torch.float32,
                device=self.device,
            )                                               # [B, d_in]
            with torch.no_grad():
                q_proj = m.input_projection(raw.unsqueeze(1))  # [B, 1, d_out]
                enc_q, _, _ = m.transformer_encoder(q_proj)    # [B, 1, d_out]
                q_norm = F.normalize(enc_q[:, 0, :], dim=-1)   # [B, d_out]
            all_out.append(q_norm.cpu().numpy())
            done = min(i + self.batch_size, len(base_vecs))
            print(f"  [UTUEL] encode queries: {done}/{len(base_vecs)}", end="\r", flush=True)
        print()

        return np.concatenate(all_out, axis=0).astype(np.float32)

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
    def _top_score_rank(
        g_embs:  "torch.Tensor | tuple",  # [N, d] tensor OR (na_np, nb_np) tuple
        g_tids:  list[str],                # [2N]   table_id per embedding
        q_norm:  torch.Tensor,             # [d,]   L2-normalised query (on device)
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
        sims: torch.Tensor
        if isinstance(g_embs, tuple):
            # Two-part numpy path: avoids allocating a 4.3 GB contiguous array.
            # Compute sims for na and nb separately; result vectors are ~22 MB.
            q_arr = q_norm.cpu().numpy().astype(np.float32)   # [d] fp32
            sims_a = torch.from_numpy(g_embs[0] @ q_arr)     # [N] fp32
            sims_b = torch.from_numpy(g_embs[1] @ q_arr)     # [N] fp32
            sims   = torch.cat([sims_a, sims_b])              # [2N] fp32
        elif g_embs.is_cuda:
            sims = (g_embs @ q_norm.to(g_embs.dtype).unsqueeze(-1)).squeeze(-1)
        else:
            # CPU tensor path.
            g_arr = g_embs.numpy()                            # [N, d] fp16 view
            q_arr = q_norm.cpu().numpy().astype(g_arr.dtype)  # [d]    fp16
            sims  = torch.from_numpy(g_arr @ q_arr)           # [N]    fp32
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
        q_batch_size:     int = 64,
    ) -> dict[str, dict]:
        """
        Compute Top-Score-Rank (TSR) table retrieval metrics across two spaces:
          ``tsr``  — global node_a ∪ node_b  (matches training eval)
          ``tbl``  — table-level avg embedding (one L2-normalised vector per table)

        Must be called *after* ``encode_table_corpus_variants()`` so that the
        global node indexes (``self._global_*_embs`` / ``self._global_*_tids``)
        are populated.

        ``q_batch_size`` controls how many queries are batched into one BLAS-3
        matmul for the ``tsr`` space.  Larger = faster (more cache-efficient)
        but uses more RAM (each batch ≈ 2 × N × C × 4 bytes).

        Returns a dict ``{space_name: {"MRR": float, "Hit@k": float, ...}}``.
        """
        if not hasattr(self, "_global_node_embs"):
            raise RuntimeError(
                "compute_tsr_metrics() requires encode_table_corpus_variants() "
                "to be called first so the global node index is built."
            )
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        hits = {"tsr": {k: 0 for k in k_values},
                "tbl":  {k: 0 for k in k_values}}
        rr: dict[str, list[float]] = {"tsr": [], "tbl": []}
        n    = len(gold_ids)
        top_n = max(max(k_values), mrr_depth)

        tsr_na, tsr_nb = self._global_node_embs      # [N, d] fp32 numpy each
        tsr_tids       = self._global_node_tids       # list[str] length 2N

        tbl_embs = self._global_tbl_embs              # [T', d] fp32 GPU tensor
        tbl_tids = self._global_tbl_tids              # list[str] length T'

        Q_arr    = query_embeddings.astype(np.float32)
        q_tensor = torch.tensor(query_embeddings, dtype=torch.float32, device=self.device)

        def _dedup_rank(sims_np: np.ndarray, g_tids: list[str]) -> list[str]:
            """
            Sort ALL node sims descending → walk in order → collect unique
            table_ids by first appearance (= max cosine per table) until top_n
            unique tables are found.
            """
            order = np.argsort(-sims_np)
            seen: dict[str, float] = {}
            for ix in order:
                t = g_tids[int(ix)]
                if t not in seen:
                    seen[t] = float(sims_np[ix])
                    if len(seen) == top_n:
                        break
            return list(seen.keys())

        def _score(ranked: list[str], gold_tid: str, sp: str) -> None:
            for k in k_values:
                if gold_tid in ranked[:k]:
                    hits[sp][k] += 1
            try:
                rr[sp].append(1.0 / (ranked[:mrr_depth].index(gold_tid) + 1))
            except ValueError:
                rr[sp].append(0.0)

        for q_start in range(0, n, q_batch_size):
            q_end   = min(q_start + q_batch_size, n)
            q_chunk = Q_arr[q_start:q_end]                    # [C, d]

            # BLAS-3 matmuls — shared for both node spaces
            sims_a_batch = tsr_na @ q_chunk.T                 # [N, C]
            sims_b_batch = tsr_nb @ q_chunk.T                 # [N, C]

            for j in range(q_end - q_start):
                q_idx    = q_start + j
                gold_tid = gold_ids[q_idx]

                # tsr: node_a ∪ node_b
                sims_ab = np.concatenate([sims_a_batch[:, j], sims_b_batch[:, j]])
                _score(_dedup_rank(sims_ab, tsr_tids), gold_tid, "tsr")

                # tbl: table-level avg embedding (already one per table)
                q_norm   = q_tensor[q_idx]
                tbl_sims = (tbl_embs @ q_norm.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                tbl_ranked = [tbl_tids[i] for i in np.argsort(-tbl_sims)[:top_n]]
                _score(tbl_ranked, gold_tid, "tbl")

            if q_end % 200 < q_batch_size or q_end == n:
                print(f"  [UTUEL] TSR eval: {q_end}/{n}", end="\r", flush=True)
        print()

        return {
            sp: {
                "MRR": round(sum(rr[sp]) / n, 4),
                **{f"Hit@{k}": round(hits[sp][k] / n, 4) for k in k_values},
            }
            for sp in ("tsr", "tbl")
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

        print(f"  [UTUEL] concatenating node embeddings …", flush=True)
        node_a_all = torch.cat(all_node_a, dim=0)   # [total_upaths, d_out]  CPU  fp32
        node_b_all = torch.cat(all_node_b, dim=0)   # [total_upaths, d_out]  CPU  fp32

        # ── Steps 5 & 6: vectorized scatter_add ──────────────────────────────
        # Build per-upath index arrays from flat_upaths metadata — no Python loop.
        N  = total_upaths
        d  = node_a_all.shape[1]
        dev = self.device

        print(f"  [UTUEL] building upath index arrays …", flush=True)
        n_up_per_tbl    = np.array([e - s for s, e in table_ranges], dtype=np.int64)
        table_assign_np = np.repeat(np.arange(T, dtype=np.int64), n_up_per_tbl)  # [N]

        tid_per_table  = [str(records[i].get("table_id", i)) for i in range(T)]
        tids_per_upath = [tid_per_table[i] for i in table_assign_np.tolist()]

        # ── Table-level aggregation: sort + reduceat on CPU ───────────────────
        # scatter_add_ and index_add_ on CUDA both use atomicAdd internally.
        # With N≈1.4M paths aggregating into T≈4K tables (≈330 paths/table),
        # atomic write-contention is extreme — every output element receives
        # ~330 concurrent atomic writes, which the SM serialises completely.
        # Fix: sort by table index on CPU, then use np.add.reduceat — a
        # contiguous segmented sum with zero atomic contention.
        print(f"  [UTUEL] sort + reduceat aggregation (CPU) …", flush=True)
        na_np = node_a_all.numpy()   # [N, d] — fp32  CPU
        nb_np = node_b_all.numpy()   # [N, d] — fp32  CPU

        order       = np.argsort(table_assign_np, kind='stable')          # [N]
        na_s        = na_np[order]                                         # [N, d] sorted by table
        nb_s        = nb_np[order]                                         # [N, d]
        cnt_np      = np.bincount(table_assign_np, minlength=T)           # [T] int64
        seg_starts  = cnt_np.cumsum() - cnt_np                            # [T] start index per segment
        nonempty_ix = np.where(cnt_np > 0)[0]

        tbl_sum_a_np = np.zeros((T, d), dtype=np.float32)
        tbl_sum_b_np = np.zeros((T, d), dtype=np.float32)
        if len(nonempty_ix):
            tbl_sum_a_np[nonempty_ix] = np.add.reduceat(na_s, seg_starts[nonempty_ix])
            tbl_sum_b_np[nonempty_ix] = np.add.reduceat(nb_s, seg_starts[nonempty_ix])

        # Transfer only the small aggregated result [T, d] to device for F.normalize
        tbl_sum_a = torch.from_numpy(tbl_sum_a_np).to(dev)                # [T, d]
        tbl_sum_b = torch.from_numpy(tbl_sum_b_np).to(dev)                # [T, d]
        tbl_cnt   = torch.from_numpy(cnt_np.astype(np.float32)).to(dev)   # [T]
        safe_cnt  = tbl_cnt.unsqueeze(1).clamp(min=1)

        tbl_mean_a = F.normalize(tbl_sum_a / safe_cnt, dim=-1)   # [T, d]
        tbl_mean_b = F.normalize(tbl_sum_b / safe_cnt, dim=-1)   # [T, d]

        # ── Step 6: pooled table embeddings (three variants) ─────────────────
        table_embs_a = tbl_mean_a.cpu().numpy().astype(np.float32)
        table_embs_b = tbl_mean_b.cpu().numpy().astype(np.float32)
        if self.pool_mode == "max":
            table_embs_both = F.normalize(
                torch.maximum(tbl_mean_a, tbl_mean_b), dim=-1,
            ).cpu().numpy().astype(np.float32)
        else:
            table_embs_both = F.normalize(
                (tbl_mean_a + tbl_mean_b) / 2, dim=-1,
            ).cpu().numpy().astype(np.float32)

        # ── Step 5a: node global index — kept on CPU fp32 ────────────────────
        # np.concatenate([na_np, nb_np]) would allocate ~8.6 GB of contiguous
        # RAM (2.8M × 768 × 4 bytes) — that hangs too.  Store as a tuple;
        # _top_score_rank computes sims in two passes (result vectors are tiny).
        print(f"  [UTUEL] building global node index (CPU fp32, split) …", flush=True)
        self._global_node_embs = (na_np, nb_np)  # tuple of [N, d] fp32 numpy
        self._global_node_tids = tids_per_upath + tids_per_upath

        # ── Step 5b: table-level global index — no longer used for TSR ──────────
        # (kept so external callers can still access _global_tbl_embs if needed)
        non_empty = tbl_cnt.nonzero(as_tuple=True)[0]
        _both_tensor = torch.from_numpy(table_embs_both).to(dev)
        self._global_tbl_embs = _both_tensor[non_empty]
        self._global_tbl_tids = [tid_per_table[i] for i in non_empty.tolist()]

        print(f"  [UTUEL] Global node index ({dev}) — "
              f"node={len(self._global_node_tids):,}  "
              f"tbl={self._global_tbl_embs.shape[0]:,}")

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

"""
train.py
Training script for TableJEPA — Table Embedding JEPA, PyTorch Lightning edition.

Run from the UTUEL repo root:
    python TRL-model/train.py

Override any config.yaml value on the CLI:
    python TRL-model/train.py training.epochs=30 model.num_layers=6
    python TRL-model/train.py data.path=datasets/summaries.jsonl training.fp16=true

Lightning handles:
  - Gradient clipping, mixed-precision (fp16/bf16-mixed)
  - Automatic checkpoint saving / resumption
  - TensorBoard / W&B logging (via logger argument)
  - Distributed / multi-GPU training (no code changes required)

Loss weights are configured in the `loss:` section of config.yaml.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import datetime

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Allow `python TRL-model/train.py` from the repo root
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from config  import TableEmbedJePAConfig                       # noqa: E402
from model   import TableEmbedJePA                                  # noqa: E402
from dataset import TableEmbedJePADataModule                   # noqa: E402


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig, embed_dim_in: int) -> TableEmbedJePA:
    """
    Instantiate a TableEmbedJePA model.

    embed_dim_in  — raw LLM embedder output dimension (from dm.embed_dim).
    hidden_size   — transformer working dimension (from cfg.model.hidden_size).
    An input_projection layer inside the model bridges the two.
    """
    hidden_size = cfg.model.hidden_size
    num_heads = max(1, min(cfg.model.num_heads, hidden_size // 64))
    while hidden_size % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    intermediate_size = cfg.model.intermediate_size or (hidden_size * 4)

    model_cfg = TableEmbedJePAConfig(
        hidden_size=hidden_size,
        num_hidden_layers=cfg.model.num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        attention_probs_dropout_prob=cfg.model.attention_dropout,
        hidden_dropout_prob=cfg.model.hidden_dropout,
        layer_norm_eps=cfg.model.layer_norm_eps,
        embedding_dim=embed_dim_in,
        tempeture=cfg.model.temperature,
        beta=cfg.model.beta,
    )
    ablate_proj = bool(OmegaConf.select(cfg, "model.ablate_proj", default=False))
    return TableEmbedJePA(
        config=model_cfg,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.epochs,
        ema_decay=cfg.model.ema_decay,
        jepa_weight=cfg.loss.jepa,
        jepa_bar_weight=cfg.loss.jepa_bar,
        local_weight=cfg.loss.local,
        global_weight=cfg.loss["global"],
        ablate_proj=ablate_proj,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_table_ids(data_path: Path, max_records: int | None = None) -> list[str]:
    """Return ordered unique table_ids from a JSONL file."""
    seen: set[str] = set()
    ids: list[str] = []
    count = 0
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tid = str(json.loads(line).get("table_id", ""))
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
            count += 1
            if max_records and count >= max_records:
                break
    return ids


# ── Embedding aggregation helpers ───────────────────────────────────────────

def compute_aggregated_embeddings(
    na: torch.Tensor,    # [n, d] L2-normalised node_a embeddings (CPU)
    ups: list,
) -> tuple[torch.Tensor, list, torch.Tensor, list, torch.Tensor]:
    """
    Aggregate node_a embeddings from a single table to column, row and table level.

    All output embeddings are L2-normalised means of the node_a vectors that
    belong to each structural unit:

      col  level  —  one vector per unique col_idx_a
                      mean of node_a(row_i, j) for all rows i in column j
      row  level  —  one vector per unique tbl_row
                      mean of node_a(i, col_j) for all columns j in row i
      tbl  level  —  one vector for the whole table
                      global mean of all node_a

    Returns
    ───────
    col_embs  [n_cols, d]  sorted by col_idx_a
    col_ids   list[int]
    row_embs  [n_rows, d]  sorted by tbl_row
    row_ids   list[int]
    tbl_emb   [1, d]
    """
    col_groups: dict[int, list[int]] = defaultdict(list)
    row_groups: dict[int, list[int]] = defaultdict(list)
    for i, up in enumerate(ups):
        col_groups[up.col_idx_a].append(i)
        row_groups[up.tbl_row].append(i)

    col_ids  = sorted(col_groups.keys())
    row_ids  = sorted(row_groups.keys())
    col_embs = F.normalize(
        torch.stack([na[col_groups[c]].mean(dim=0) for c in col_ids]), dim=-1)  # [n_cols, d]
    row_embs = F.normalize(
        torch.stack([na[row_groups[r]].mean(dim=0) for r in row_ids]), dim=-1)  # [n_rows, d]
    tbl_emb  = F.normalize(na.mean(dim=0, keepdim=True), dim=-1)               # [1, d]
    return col_embs, col_ids, row_embs, row_ids, tbl_emb


# ── Post-training retrieval evaluation ────────────────────────────────────────────────

def evaluate_model(
    model: TableEmbedJePA,
    dm: TableEmbedJePADataModule,
    cfg: DictConfig,
    out_path: Path,
    device: str | None = None,
    top_k_table: int = 2000,
) -> dict:
    """
    Compute cell-retrieval metrics for a trained TableEmbedJePA model.

    Metrics (matching UTUEL paper, Table 2):
      MRR           Mean Reciprocal Rank over all evaluated records
      Hit@1         Exact cell (tbl_row, col_idx) match at rank 1
      Hit@1_Row     Row-only match at rank 1 (any GT row)
      Hit@1_Col     Column-only match at rank 1 (GT target column)
      Hit@3, Hit@5  Exact cell match within top-3 / top-5

    Reported for three search spaces in one pass:
      SMP_node_a      SMP encoder, node_a embeddings (col_idx_a)
      SMP_bar_node_b  SMP_bar encoder, node_b embeddings (col_idx_b)
      both            Combined [SMP_node_a ; SMP_bar_node_b]

    Search-space grouping (correct multi-question evaluation):
      The dataset may contain multiple records (questions) sharing the same
      table_id — they differ only in their question and target cell.
      The U-paths are built from the physical table, so all records with the
      same table_id produce IDENTICAL U-paths.

      Evaluation therefore:
        1. Groups records by table_id.
        2. Encodes the table's U-paths ONCE (from the first record only,
           since all are identical).
        3. Evaluates EVERY question (record) against that shared search space.
        4. Tracks record id ("id" field from the JSONL) in the output.

    Results are written to `out_path` as a JSON file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = dm._dataset
    if ds is None:
        dm.setup()
        ds = dm._dataset

    model = model.eval().to(device)

    # ── Group by table_id ─────────────────────────────────────────────────────
    # tid_to_first_rec : table_id → first rec_idx that carries this table
    # tid_to_rec_idxs  : table_id → all rec_idxs (all questions for the table)
    tid_to_first_rec: dict[str, int] = {}
    tid_to_rec_idxs: dict[str, list[int]] = defaultdict(list)
    for rec_idx, rec in enumerate(ds.records):
        tid = str(rec.get("table_id", ""))
        if tid not in tid_to_first_rec:
            tid_to_first_rec[tid] = rec_idx
        tid_to_rec_idxs[tid].append(rec_idx)

    # tid_to_sample_js : table_id → flat sample indices from the FIRST record
    # (U-paths are identical across records with the same table_id, so using
    # only the first record avoids a duplicated search space.)
    tid_to_sample_js: dict[str, list[int]] = defaultdict(list)
    for j, (rec_idx, _) in enumerate(ds._samples):
        tid = str(ds.records[rec_idx].get("table_id", ""))
        if rec_idx == tid_to_first_rec.get(tid):
            tid_to_sample_js[tid].append(j)

    n_tables  = len(tid_to_rec_idxs)
    n_records = len(ds.records)

    _SPACES = ("SMP_node_a", "SMP_bar_node_b", "both")
    accs: dict[str, dict[str, list]] = {
        sp: {"mrr": [], "h1": [], "h1r": [], "h1c": [], "h3": [], "h5": []}
        for sp in _SPACES
    }
    tbl_hits: dict[str, list] = {
        "node_h1": [], "node_h3": [],
        "col_h1":  [], "col_h3":  [],
        "row_h1":  [], "row_h3":  [],
        "tbl_h1":  [], "tbl_h3":  [],
    }
    per_record: list[dict]             = []
    skipped = 0

    print(f"[eval] {n_records} questions across {n_tables} unique tables  device={device}")

    # ── Phase 1: encode all table U-paths; build global node index ─────────────
    print(f"[eval] Building global node index ({n_tables} tables) …")
    tid_to_na:   dict[str, torch.Tensor] = {}
    tid_to_nb:   dict[str, torch.Tensor] = {}
    tid_to_ups:  dict[str, list]         = {}
    _glob_parts:     list[torch.Tensor]  = []
    _glob_tids:      list[str]           = []
    _glob_col_parts: list[torch.Tensor]  = []
    _glob_col_tids:  list[str]           = []
    _glob_row_parts: list[torch.Tensor]  = []
    _glob_row_tids:  list[str]           = []
    _glob_tbl_parts: list[torch.Tensor]  = []
    _glob_tbl_tids:  list[str]           = []

    for tid, sample_js in tid_to_sample_js.items():
        ups  = [ds._samples[j][1] for j in sample_js]
        n_up = len(ups)
        idx_t   = torch.tensor(sample_js, dtype=torch.long)
        smp_raw = ds._embed_cache[ds._smp_idx[idx_t]].to(device)

        with torch.no_grad():
            proj = model.input_projection(smp_raw)
            cls  = model.cls_token.expand(n_up, -1, -1)
            enc, _, _ = model.transformer_encoder(torch.cat([cls, proj], dim=1))
            na = F.normalize(enc[:, 2, :], dim=-1).cpu()          # [n, d]

            smp_bar = smp_raw[:, [3, 2, 1, 0], :]
            proj_b  = model.input_projection(smp_bar)
            cls_b   = model.cls_token.expand(n_up, -1, -1)
            enc_b, _, _ = model.transformer_encoder(torch.cat([cls_b, proj_b], dim=1))
            nb = F.normalize(enc_b[:, 2, :], dim=-1).cpu()        # [n, d]

        tid_to_na[tid]  = na
        tid_to_nb[tid]  = nb
        tid_to_ups[tid] = ups
        _glob_parts.extend([na, nb])
        _glob_tids.extend([tid] * (n_up * 2))

        # Aggregate node_a to column, row, and table level
        _col_e, _, _row_e, _, _tbl_e = compute_aggregated_embeddings(na, ups)
        _glob_col_parts.append(_col_e);  _glob_col_tids.extend([tid] * _col_e.shape[0])
        _glob_row_parts.append(_row_e);  _glob_row_tids.extend([tid] * _row_e.shape[0])
        _glob_tbl_parts.append(_tbl_e);  _glob_tbl_tids.append(tid)
        print(f"\r[eval] encoded {len(tid_to_na)}/{n_tables} tables", end="", flush=True)

    print()
    global_embs     = torch.cat(_glob_parts,     dim=0)  # [N_node_total, d]  CPU
    global_col_embs = torch.cat(_glob_col_parts, dim=0)  # [N_cols_total, d]
    global_row_embs = torch.cat(_glob_row_parts, dim=0)  # [N_rows_total, d]
    global_tbl_embs = torch.cat(_glob_tbl_parts, dim=0)  # [N_tables, d]
    print(f"[eval] Global indexes \u2014 "
          f"node={global_embs.shape[0]:,}  "
          f"col={global_col_embs.shape[0]:,}  "
          f"row={global_row_embs.shape[0]:,}  "
          f"tbl={global_tbl_embs.shape[0]:,}")

    # ── Phase 2: evaluate questions (cell retrieval + table retrieval) ─────────
    for tid, rec_idxs in tid_to_rec_idxs.items():
        na   = tid_to_na.get(tid)
        nb   = tid_to_nb.get(tid)
        ups  = tid_to_ups.get(tid)
        if na is None or nb is None or ups is None:
            skipped += len(rec_idxs)
            continue

        n_up      = len(ups)
        both_embs = torch.cat([na, nb], dim=0)   # [2n, d]

        def coord_a(i: int) -> tuple[int, int]:
            return (ups[i].tbl_row, ups[i].col_idx_a)

        def coord_b(i: int) -> tuple[int, int]:
            return (ups[i].tbl_row, ups[i].col_idx_b)

        def coord_both(i: int) -> tuple[int, int]:
            if i < n_up:
                return (ups[i].tbl_row, ups[i].col_idx_a)
            return (ups[i - n_up].tbl_row, ups[i - n_up].col_idx_b)

        # Evaluate every question (record) against the shared table search space
        for rec_idx in rec_idxs:
            rec         = ds.records[rec_idx]
            target_col  = rec.get("target_column")
            target_rows = rec.get("target_rows", [])
            question    = rec.get("question", "")
            record_id   = str(rec.get("id", ""))

            if target_col is None or not target_rows or not question:
                skipped += 1
                continue

            gt_cells = {(rr + 1, target_col) for rr in target_rows}
            gt_rows  = {rr + 1 for rr in target_rows}

            # Encode question through the online encoder using precomputed LLM embedding
            q_raw = ds._question_cache[rec_idx].unsqueeze(0).to(device)  # [1, d_in]
            with torch.no_grad():
                q_proj = model.input_projection(q_raw.unsqueeze(1))   # [1, 1, d_out]
                enc_q, _, _ = model.transformer_encoder(q_proj)
                q_norm = F.normalize(enc_q[:, 0, :], dim=-1).cpu()    # [1, d_out]

            def _rank_space(embs: torch.Tensor, coord_fn) -> dict:
                sims   = (embs @ q_norm.T).squeeze(-1)
                ranked = sims.argsort(descending=True).tolist()
                top1   = coord_fn(ranked[0])
                top3   = [coord_fn(i) for i in ranked[:3]]
                top5   = [coord_fn(i) for i in ranked[:5]]
                rr = 0.0
                for rank, idx in enumerate(ranked, 1):
                    if coord_fn(idx) in gt_cells:
                        rr = 1.0 / rank
                        break
                return {
                    "top1": list(top1),
                    "rr":   rr,
                    "h1":   top1 in gt_cells,
                    "h1r":  top1[0] in gt_rows,
                    "h1c":  top1[1] == target_col,
                    "h3":   any(c in gt_cells for c in top3),
                    "h5":   any(c in gt_cells for c in top5),
                }

            res = {
                "SMP_node_a":     _rank_space(na,        coord_a),
                "SMP_bar_node_b": _rank_space(nb,        coord_b),
                "both":           _rank_space(both_embs, coord_both),
            }

            for sp in _SPACES:
                r = res[sp]
                accs[sp]["mrr"].append(r["rr"])
                accs[sp]["h1"].append(float(r["h1"]))
                accs[sp]["h1r"].append(float(r["h1r"]))
                accs[sp]["h1c"].append(float(r["h1c"]))
                accs[sp]["h3"].append(float(r["h3"]))
                accs[sp]["h5"].append(float(r["h5"]))

            # ── Table retrieval: node / col / row / table level ───────────────────
            def _majority_vote(g_embs: torch.Tensor, g_tids: list) -> tuple:
                """Top-k cosine sim → majority vote by table_id → top-3 tables."""
                _k_   = min(top_k_table, g_embs.shape[0])
                _tops = [g_tids[i]
                         for i in (g_embs @ q_norm.T).squeeze(-1).topk(_k_).indices.tolist()]
                _v    = [t for t, _ in Counter(_tops).most_common(3)]
                return bool(_v) and _v[0] == tid, tid in _v, _v

            # Table level: one vector per table → direct cosine rank (no majority vote)
            _sim_tbl  = (global_tbl_embs @ q_norm.T).squeeze(-1)
            _top_t    = [_glob_tbl_tids[i]
                         for i in _sim_tbl.argsort(descending=True).tolist()[:3]]
            _e1, _e3  = bool(_top_t) and _top_t[0] == tid, tid in _top_t

            _n1, _n3, _top_node = _majority_vote(global_embs,     _glob_tids)
            _c1, _c3, _top_col  = _majority_vote(global_col_embs, _glob_col_tids)
            _r1, _r3, _top_row  = _majority_vote(global_row_embs, _glob_row_tids)

            tbl_hits["node_h1"].append(float(_n1)); tbl_hits["node_h3"].append(float(_n3))
            tbl_hits["col_h1"].append(float(_c1));  tbl_hits["col_h3"].append(float(_c3))
            tbl_hits["row_h1"].append(float(_r1));  tbl_hits["row_h3"].append(float(_r3))
            tbl_hits["tbl_h1"].append(float(_e1));  tbl_hits["tbl_h3"].append(float(_e3))

            per_record.append({
                "rec_idx":     rec_idx,
                "table_id":    tid,
                "record_id":   record_id,
                "question":    question,
                "target_col":  target_col,
                "target_rows": list(target_rows),
                "SMP_node_a":     res["SMP_node_a"],
                "SMP_bar_node_b": res["SMP_bar_node_b"],
                "both":           res["both"],
                "table_retrieval": {
                    "node": {"Table@1": _n1, "Table@3": _n3, "top_tables": _top_node},
                    "col":  {"Table@1": _c1, "Table@3": _c3, "top_tables": _top_col},
                    "row":  {"Table@1": _r1, "Table@3": _r3, "top_tables": _top_row},
                    "tbl":  {"Table@1": _e1, "Table@3": _e3, "top_tables": _top_t},
                },
            })

    def _agg(acc: dict) -> dict:
        n = max(len(acc["mrr"]), 1)
        return {
            "MRR":         round(sum(acc["mrr"]) / n, 4),
            "Hit@1":       round(100 * sum(acc["h1"])  / n, 2),
            "Hit@1_Row":   round(100 * sum(acc["h1r"]) / n, 2),
            "Hit@1_Col":   round(100 * sum(acc["h1c"]) / n, 2),
            "Hit@3":       round(100 * sum(acc["h3"])  / n, 2),
            "Hit@5":       round(100 * sum(acc["h5"])  / n, 2),
            "n_evaluated": len(acc["mrr"]),
            "n_skipped":   skipped,
        }

    def _agg_tbl(hits: dict) -> dict:
        n = max(len(hits["node_h1"]), 1)
        def _p(k): return round(100 * sum(hits[k]) / n, 2)
        return {
            "node": {"Table@1": _p("node_h1"), "Table@3": _p("node_h3"),
                     "search_space": f"node_a+node_b (top-{top_k_table} majority vote)"},
            "col":  {"Table@1": _p("col_h1"),  "Table@3": _p("col_h3"),
                     "search_space": f"col_mean(node_a) (top-{top_k_table} majority vote)"},
            "row":  {"Table@1": _p("row_h1"),  "Table@3": _p("row_h3"),
                     "search_space": f"row_mean(node_a) (top-{top_k_table} majority vote)"},
            "tbl":  {"Table@1": _p("tbl_h1"),  "Table@3": _p("tbl_h3"),
                     "search_space": "table_mean(node_a) (direct cosine rank)"},
            "n_evaluated": n,
        }

    metrics                 = {sp: _agg(accs[sp]) for sp in _SPACES}
    table_retrieval_metrics = _agg_tbl(tbl_hits)

    results = {
        "metadata": {
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
            "model_class":   type(model).__name__,
            "hidden_size":   int(cfg.model.hidden_size),
            "num_layers":    int(cfg.model.num_layers),
            "num_heads":     int(cfg.model.num_heads),
            "ema_decay":     float(cfg.model.ema_decay),
            "ablate_proj":   bool(OmegaConf.select(cfg, "model.ablate_proj", default=False)),
            "embedder":      cfg.embedder.model_name,
            "data_path":     str(cfg.data.path),
            "max_records":   cfg.data.max_records,
            "training_mode": cfg.training.get("mode", "global"),
            "n_tables":      n_tables,
            "n_questions":   n_records,
            "loss_weights": {
                "jepa":     float(cfg.loss.jepa),
                "jepa_bar": float(cfg.loss.jepa_bar),
                "local":    float(cfg.loss.local),
                "global":   float(cfg.loss["global"]),
            },
        },
        "metrics":          metrics,
        "table_retrieval":  table_retrieval_metrics,
        "per_record":       per_record,
    }

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── Print summary table ───────────────────────────────────────────────────
    _COL = 18
    header = f"[eval] {'Metric':<14}" + "".join(f"{sp:>{_COL}}" for sp in _SPACES)
    sep    = "[eval] " + "-" * (14 + _COL * len(_SPACES))
    print(f"\n{header}")
    print(sep)
    for m in ("MRR", "Hit@1", "Hit@1_Row", "Hit@1_Col", "Hit@3", "Hit@5"):
        vals = "".join(
            f"{metrics[sp][m]:>{_COL}.4f}" if m == "MRR"
            else f"{str(metrics[sp][m]) + '%':>{_COL}}"
            for sp in _SPACES
        )
        print(f"[eval] {m:<14}{vals}")
    print(sep)
    _TLEVELS = ("node", "col", "row", "tbl")
    _TW = 11
    _n_eval = table_retrieval_metrics["n_evaluated"]
    print(f"\n[eval] Table retrieval  (n={_n_eval}  top-{top_k_table} \u2192 majority vote for node/col/row;  direct rank for tbl)")
    print("[eval]   " + f"{'level':<10}" + "".join(f"{l:>{_TW}}" for l in _TLEVELS))
    print("[eval]   " + "-" * (10 + _TW * len(_TLEVELS)))
    for _tm in ("Table@1", "Table@3"):
        print(f"[eval]   {_tm:<10}" + "".join(
            f"{str(table_retrieval_metrics[l][_tm]) + '%':>{_TW}}"
            for l in _TLEVELS))
    print(f"[eval] Results \u2192 {out_path}")
    return results


def _train_one(
    cfg: DictConfig,
    dm: TableEmbedJePADataModule,
    out_dir: Path,
    run_label: str,
) -> float:
    """Train one model and return the Optuna objective (MRR of the 'both' space)."""
    """Build model, run training, and save for one DataModule."""
    embed_dim_in = dm.embed_dim   # already truncated if truncate_embed_dim was set
    print(f"[train] embed_dim_in={embed_dim_in}  hidden_size={cfg.model.hidden_size}"
          f"  samples={len(dm._dataset)}")

    lightning_model = build_model(cfg, embed_dim_in)
    n_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params:,}")

    # Subfolder = last component of the embedder model name (sanitized)
    _model_slug = cfg.embedder.model_name.split("/")[-1].replace(" ", "_")
    ckpt_dir = out_dir / _model_slug
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the full resolved config alongside the checkpoints immediately,
    # so it is present even if training is interrupted before completion.
    OmegaConf.save(cfg, str(ckpt_dir / "run_config.yaml"))
    print(f"[train] checkpoint dir   : {ckpt_dir}")
    print(f"[train] embedder slug    : {_model_slug}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_label}-{{epoch:02d}}-{{train_loss:.4f}}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=cfg.training.save_total_limit,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if cfg.training.bf16:
        precision = "bf16-mixed"
    elif cfg.training.fp16:
        precision = "16-mixed"
    else:
        precision = 32

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=precision,
        gradient_clip_val=cfg.training.max_grad_norm,
        log_every_n_steps=cfg.training.logging_steps,
        callbacks=[checkpoint_cb, lr_monitor],
        deterministic=False,
    )
    ckpt_path = OmegaConf.select(cfg, "training.ckpt_path", default=None) or None
    if ckpt_path:
        print(f"[train] Resuming from checkpoint: {ckpt_path}")
    trainer.fit(lightning_model, datamodule=dm, ckpt_path=ckpt_path)

    final_path = ckpt_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(lightning_model.state_dict(), str(final_path / "model.pt"))
    OmegaConf.save(cfg, str(final_path / "run_config.yaml"))
    print(f"[train] Saved → {final_path}")
    # ── Evaluate and save retrieval metrics ──────────────────────────────────
    results = evaluate_model(
        model=lightning_model,
        dm=dm,
        cfg=cfg,
        out_path=ckpt_dir / "eval_results.json",
    )
    # Return MRR ("both" space) as the Optuna objective; higher is better.
    # Fallback to −train_loss when eval yields no evaluated records.
    try:
        mrr = float(results["metrics"]["both"]["MRR"])
        if mrr > 0.0:
            return mrr
    except (KeyError, TypeError):
        pass
    _logged = trainer.callback_metrics
    return -float(_logged.get("train_loss_epoch", float("inf")))

# ── Main training function ────────────────────────────────────────────────────

def train(cfg: DictConfig) -> float:
    """Run training and return the Optuna objective value (MRR or −loss)."""
    print("=" * 60)
    print("TableJEPA — Table Embedding JEPA (PyTorch Lightning)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    pl.seed_everything(cfg.training.seed, workers=True)

    # ── Tensor Core precision ─────────────────────────────────────────────────
    _matmul_prec = OmegaConf.select(cfg, "matmul_precision", default="highest")
    torch.set_float32_matmul_precision(_matmul_prec)
    print(f"[train] float32 matmul precision : {_matmul_prec}")

    data_path = Path(cfg.data.path)
    if not data_path.is_absolute():
        data_path = (_HERE.parent / data_path).resolve()

    out_base = Path(cfg.training.output_dir)
    if not out_base.is_absolute():
        out_base = (_HERE.parent / out_base).resolve()

    mode = cfg.training.get("mode", "global")

    # ── Common DataModule kwargs ──────────────────────────────────────────────
    _dm_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.dataloader_num_workers,
        model_type=cfg.embedder.model_type,
        base_url=cfg.embedder.base_url,
        model_name=cfg.embedder.model_name,
        api_key=cfg.embedder.api_key,
        max_records=cfg.data.max_records,
        precompute=cfg.embedder.precompute,
        embed_batch_size=cfg.embedder.embed_batch_size,
        use_graph_walks=cfg.smp.use_graph_walks,
        num_walks=cfg.smp.num_walks,
        chunk_size=cfg.smp.chunk_size,
        truncate_embed_dim=int(cfg.embedder.embed_dim) if cfg.embedder.embed_dim else None,
        cache_embeddings=bool(OmegaConf.select(cfg, "embedder.cache_embeddings", default=False)),
        embed_cache_dir=OmegaConf.select(cfg, "embedder.embed_cache_dir", default=None),
    )

    if mode == "per_table":
        # ── Per-table mode: one model per table_id ────────────────────────────
        table_ids = get_table_ids(data_path, max_records=cfg.data.max_records)
        print(f"[train] per_table mode — {len(table_ids)} tables found")

        scores: list[float] = []
        for i, tid in enumerate(table_ids, 1):
            print(f"\n{'='*60}")
            print(f"[train] Table {i}/{len(table_ids)}: {tid}")
            print(f"{'='*60}")

            dm = TableEmbedJePADataModule(
                jsonl_path=data_path,
                filter_table_id=tid,
                **_dm_kwargs,
            )
            dm.setup()

            n_samples = len(dm._dataset)
            if n_samples < cfg.training.batch_size:
                print(f"[train] Skipping — only {n_samples} samples "
                      f"(< batch_size {cfg.training.batch_size})")
                continue

            score = _train_one(cfg, dm, out_dir=out_base / tid, run_label=tid)
            scores.append(score)
        return float(sum(scores) / max(len(scores), 1)) if scores else float("-inf")

    else:
        # ── Global mode: single model on all tables ───────────────────────────
        dm = TableEmbedJePADataModule(jsonl_path=data_path, **_dm_kwargs)
        dm.setup()
        return _train_one(cfg, dm, out_dir=out_base, run_label="table-jepa")


# ── Hydra entry point ─────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()


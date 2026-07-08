"""
train.py
Training script for TableEmbedJePA on the complex / nested **Adhesive** tables,
PyTorch Lightning edition.

Run from the UTUEL repo root:
    python TRL_Adhesive/train.py

Override any config.yaml value on the CLI:
    python TRL_Adhesive/train.py training.epochs=30 model.num_layers=6
    python TRL_Adhesive/train.py data.dir=/path/to/ReleasedTableDatasetAdhesive

The model architecture (``model/TableEmbedJePA_v1.py``) and the config class
(``config.py``) are identical to ``TRL-model``; only the dataset / SMP parsing
and the evaluation differ, because the answer key here is a cell **id**
(``answer_cell_id``) rather than a (row, col) coordinate.
"""

from __future__ import annotations

import hashlib
import json
import sys
import datetime
import logging
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Allow `python TRL_Adhesive/train.py` from the repo root
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from config  import TableEmbedJePAConfig            # noqa: E402
from model   import TableEmbedJePA                   # noqa: E402
from dataset import TableEmbedJePADataModule         # noqa: E402

for _noisy in ("httpx", "httpcore", "langchain", "langchain_core", "ollama"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig, embed_dim_in: int) -> TableEmbedJePA:
    """Instantiate a TableEmbedJePA model (see TRL-model for details)."""
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


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode_nodes(
    model,
    seq_raw: torch.Tensor,
    role: torch.Tensor,
    pad_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Return L2-normalised node embeddings for a padded, role-masked SMP batch.

    Args:
        seq_raw  [n, L, d]: ordered walk embeddings (padded)
        role     [n, L]   : 1 at the node slot, 0 at headers / pad
        pad_mask [n, L]   : 1 for valid slots, 0 for padding
    """
    n = seq_raw.shape[0]
    NEG = -1e9
    with torch.no_grad():
        proj      = model.input_projection(seq_raw)                  # [n, L, d]
        cls_proj  = model.input_projection(model.cls_token)
        cls       = cls_proj.expand(n, -1, -1)                       # [n, 1, d]
        inp       = torch.cat([cls, proj], dim=1)                    # [n, L+1, d]
        cls_ones  = torch.ones(n, 1, device=device)
        cls_zeros = torch.zeros(n, 1, device=device)
        pad_full  = torch.cat([cls_ones,  pad_mask], dim=1)          # [n, L+1]
        role_full = torch.cat([cls_zeros, role],     dim=1)          # [n, L+1]
        attn_add  = (1.0 - pad_full)[:, None, None, :] * NEG         # [n, 1, 1, L+1]
        enc, _, _ = model.transformer_encoder(inp, mask=attn_add)    # [n, L+1, d]
        r = role_full.unsqueeze(-1)                                  # [n, L+1, 1]
        node = (enc * r).sum(1) / r.sum(1).clamp_min(1e-6)           # [n, d]
        return F.normalize(node, dim=-1).cpu()


# ── Post-training retrieval evaluation ────────────────────────────────────────

def evaluate_model(
    model: TableEmbedJePA,
    dm: TableEmbedJePADataModule,
    cfg: DictConfig,
    out_path: Path,
    device: str | None = None,
    top_k_table: int = 2000,
    run_start_ts: str | None = None,
) -> dict:
    """
    Cell-retrieval + table-retrieval metrics for complex Adhesive tables.

    Ground truth is the ``answer_cell_id`` of each question.  A U-path matches
    the answer when ``UPath.node_id == answer_cell_id``.

    Metrics per question:
      MRR, Hit@1, Hit@3, Hit@5   over U-paths ranked by cosine(node, question).

    Search spaces:
      SMP_node       — node embeddings from the SMP orientation
      SMP_bar_node   — node embeddings from the reversed (SMP_bar) orientation
      both           — concatenation of the two
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = dm._dataset
    if ds is None:
        dm.setup()
        ds = dm._dataset
    model = model.eval().to(device)

    # ── Group records by table_id ─────────────────────────────────────────────
    tid_to_first_rec: dict[str, int] = {}
    tid_to_rec_idxs: dict[str, list[int]] = defaultdict(list)
    for rec_idx, rec in enumerate(ds.records):
        tid = str(rec.get("table_id", ""))
        if tid not in tid_to_first_rec:
            tid_to_first_rec[tid] = rec_idx
        tid_to_rec_idxs[tid].append(rec_idx)

    tid_to_sample_js: dict[str, list[int]] = defaultdict(list)
    for j, (rec_idx, _) in enumerate(ds._samples):
        tid = str(ds.records[rec_idx].get("table_id", ""))
        if rec_idx == tid_to_first_rec.get(tid):
            tid_to_sample_js[tid].append(j)

    n_tables  = len(tid_to_rec_idxs)
    n_records = len(ds.records)

    _SPACES = ("SMP_node", "SMP_bar_node", "both")
    _VARIANTS = ("raw", "sw", "hit")   # raw=original, sw=stop-word-removed, hit=high-informative-token
    accs = {var: {sp: {"mrr": [], "h1": [], "h2": [], "h3": [], "h10": [], "h20": []}
                  for sp in _SPACES}
            for var in _VARIANTS}
    per_record: list[dict] = []
    skipped = 0

    print(f"[eval] {n_records} questions across {n_tables} tables  device={device}")

    # ── Phase 1: encode every table's node embeddings once ─────────────────────
    tid_to_na:  dict[str, torch.Tensor] = {}
    tid_to_nb:  dict[str, torch.Tensor] = {}
    tid_to_ups: dict[str, list]         = {}
    _glob_parts: list[torch.Tensor] = []
    _glob_tids:  list[str]          = []
    _glob_tbl_parts: list[torch.Tensor] = []
    _glob_tbl_tids:  list[str]          = []

    for tid, sample_js in tid_to_sample_js.items():
        ups     = [ds._samples[j][1] for j in sample_js]
        n_up    = len(ups)
        idx_t   = torch.tensor(sample_js, dtype=torch.long)
        _L          = ds._seq_idx.shape[1]
        seq_raw     = ds._embed_cache[ds._seq_idx[idx_t]].to(device)      # [n, L, d]
        seq_bar_raw = ds._embed_cache[ds._seq_idx_bar[idx_t]].to(device)  # [n, L, d]
        role        = ds._role[idx_t].to(device)                         # [n, L]
        role_bar    = ds._role_bar[idx_t].to(device)                     # [n, L]
        lens        = ds._seq_len[idx_t]                                 # [n]
        pad_mask    = (torch.arange(_L).unsqueeze(0) < lens.unsqueeze(1)).float().to(device)

        na = _encode_nodes(model, seq_raw,     role,     pad_mask, device)  # [n, d]
        nb = _encode_nodes(model, seq_bar_raw, role_bar, pad_mask, device)  # [n, d]

        tid_to_na[tid]  = na
        tid_to_nb[tid]  = nb
        tid_to_ups[tid] = ups
        _glob_parts.extend([na, nb]);      _glob_tids.extend([tid] * (n_up * 2))
        _tbl_e = F.normalize(na.mean(dim=0, keepdim=True), dim=-1)
        _glob_tbl_parts.append(_tbl_e);    _glob_tbl_tids.append(tid)
        print(f"\r[eval] encoded {len(tid_to_na)}/{n_tables} tables", end="", flush=True)
    print()

    global_embs     = torch.cat(_glob_parts,     dim=0).to(device)
    global_tbl_embs = torch.cat(_glob_tbl_parts, dim=0).to(device)
    print(f"[eval] Global index — node={global_embs.shape[0]:,}  tbl={global_tbl_embs.shape[0]:,}")

    tbl_hits = {var: {f"{lvl}_{k}": [] for lvl in ("node", "tbl")
                      for k in ("h1", "h3", "h5", "h10", "h20")}
                for var in _VARIANTS}

    # ── Phase 2: evaluate each question ────────────────────────────────────────
    for tid, rec_idxs in tid_to_rec_idxs.items():
        na_cpu, nb_cpu, ups = tid_to_na.get(tid), tid_to_nb.get(tid), tid_to_ups.get(tid)
        if na_cpu is None or nb_cpu is None or ups is None:
            skipped += len(rec_idxs)
            continue

        na        = na_cpu.to(device)
        nb        = nb_cpu.to(device)
        n_up      = len(ups)
        both_embs = torch.cat([na, nb], dim=0)
        node_ids  = [up.node_id for up in ups]

        def node_of(i: int) -> int:
            return node_ids[i] if i < n_up else node_ids[i - n_up]

        for rec_idx in rec_idxs:
            rec         = ds.records[rec_idx]
            question    = rec.get("question", "")
            question_sw = rec.get("question_sw", "") or question
            question_hit = rec.get("question_hit", "") or question
            ans_id      = int(rec.get("answer_cell_id", -1))
            record_id   = str(rec.get("id", ""))

            if not question or ans_id < 0:
                skipped += 1
                continue

            def _rank(embs: torch.Tensor, id_fn, q_norm) -> dict:
                sims   = (embs @ q_norm.T).squeeze(-1)
                ranked = sims.argsort(descending=True).tolist()
                rr = 0.0
                for rank, i in enumerate(ranked, 1):
                    if id_fn(i) == ans_id:
                        rr = 1.0 / rank
                        break
                top1  = id_fn(ranked[0])
                top2  = {id_fn(i) for i in ranked[:2]}
                top3  = {id_fn(i) for i in ranked[:3]}
                top10 = {id_fn(i) for i in ranked[:10]}
                top20 = {id_fn(i) for i in ranked[:20]}
                return {
                    "top1": top1, "rr": rr,
                    "h1":  top1 == ans_id,
                    "h2":  ans_id in top2,
                    "h3":  ans_id in top3,
                    "h10": ans_id in top10,
                    "h20": ans_id in top20,
                }

            def _majority(g_embs, g_tids, q_norm) -> tuple:
                _k = min(top_k_table, g_embs.shape[0])
                _tops = [g_tids[i] for i in (g_embs @ q_norm.T).squeeze(-1).topk(_k).indices.tolist()]
                _v = [t for t, _ in Counter(_tops).most_common(20)]
                return (bool(_v) and _v[0] == tid, tid in _v[:3], tid in _v[:5],
                        tid in _v[:10], tid in _v, _v[:3])

            # ── Evaluate each query variant (RAW / SW / HIT) ──────────────────
            _q_src = {"raw": ds._question_cache,
                      "sw":  ds._question_sw_cache,
                      "hit": ds._question_hit_cache}
            _pr_variants: dict[str, dict] = {}
            for var in _VARIANTS:
                q_raw = _q_src[var][rec_idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    q_proj = model.input_projection(q_raw.unsqueeze(1))     # [1, 1, d]
                    enc_q, _, _ = model.transformer_encoder(q_proj)
                    q_norm = F.normalize(enc_q[:, 0, :], dim=-1)            # [1, d]

                res = {
                    "SMP_node":     _rank(na,        lambda i: node_ids[i], q_norm),
                    "SMP_bar_node": _rank(nb,        lambda i: node_ids[i], q_norm),
                    "both":         _rank(both_embs, node_of, q_norm),
                }
                for sp in _SPACES:
                    r = res[sp]
                    accs[var][sp]["mrr"].append(r["rr"])
                    accs[var][sp]["h1"].append(float(r["h1"]))
                    accs[var][sp]["h2"].append(float(r["h2"]))
                    accs[var][sp]["h3"].append(float(r["h3"]))
                    accs[var][sp]["h10"].append(float(r["h10"]))
                    accs[var][sp]["h20"].append(float(r["h20"]))

                # ── Table retrieval ───────────────────────────────────────────
                _n1, _n3, _n5, _n10, _n20, _top_node = _majority(global_embs, _glob_tids, q_norm)
                _sim_tbl = (global_tbl_embs @ q_norm.T).squeeze(-1)
                _top_t   = [_glob_tbl_tids[i]
                            for i in _sim_tbl.topk(min(20, _sim_tbl.shape[0])).indices.tolist()]
                _t1, _t3, _t5, _t10, _t20 = (_top_t[:1] == [tid], tid in _top_t[:3],
                                             tid in _top_t[:5], tid in _top_t[:10], tid in _top_t)

                for k, v in (("node_h1", _n1), ("node_h3", _n3), ("node_h5", _n5),
                             ("node_h10", _n10), ("node_h20", _n20),
                             ("tbl_h1", _t1), ("tbl_h3", _t3), ("tbl_h5", _t5),
                             ("tbl_h10", _t10), ("tbl_h20", _t20)):
                    tbl_hits[var][k].append(float(v))

                _pr_variants[var] = {
                    "SMP_node": res["SMP_node"], "SMP_bar_node": res["SMP_bar_node"],
                    "both": res["both"],
                    "table_retrieval": {
                        "node": {"Table@1": _n1, "Table@3": _n3, "Table@5": _n5,
                                 "Table@10": _n10, "Table@20": _n20, "top_tables": _top_node},
                        "tbl":  {"Table@1": _t1, "Table@3": _t3, "Table@5": _t5,
                                 "Table@10": _t10, "Table@20": _t20, "top_tables": _top_t[:3]},
                    },
                }

            per_record.append({
                "rec_idx": rec_idx, "table_id": tid, "record_id": record_id,
                "question": question, "question_sw": question_sw,
                "question_hit": question_hit,
                "answer_cell_id": ans_id,
                "raw": _pr_variants["raw"], "sw": _pr_variants["sw"],
                "hit": _pr_variants["hit"],
            })

    def _agg(acc: dict) -> dict:
        n = max(len(acc["mrr"]), 1)
        return {
            "MRR":    round(sum(acc["mrr"]) / n, 4),
            "Hit@1":  round(100 * sum(acc["h1"]) / n, 2),
            "Hit@2":  round(100 * sum(acc["h2"]) / n, 2),
            "Hit@3":  round(100 * sum(acc["h3"]) / n, 2),
            "Hit@10": round(100 * sum(acc["h10"]) / n, 2),
            "Hit@20": round(100 * sum(acc["h20"]) / n, 2),
            "n_evaluated": len(acc["mrr"]),
            "n_skipped":   skipped,
        }

    def _agg_tbl(th: dict) -> dict:
        n = max(len(th["node_h1"]), 1)
        def _p(k): return round(100 * sum(th[k]) / n, 2)
        return {
            "node": {f"Table@{s}": _p(f"node_h{s}") for s in (1, 3, 5, 10, 20)},
            "tbl":  {f"Table@{s}": _p(f"tbl_h{s}")  for s in (1, 3, 5, 10, 20)},
            "n_evaluated": n,
        }

    metrics = {var: {sp: _agg(accs[var][sp]) for sp in _SPACES} for var in _VARIANTS}
    table_retrieval_metrics = {var: _agg_tbl(tbl_hits[var]) for var in _VARIANTS}

    _eval_ts = datetime.datetime.now().isoformat(timespec="seconds")
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_total     = sum(p.numel() for p in model.parameters())

    results = {
        "metadata": {
            "run_start": run_start_ts,
            "eval_timestamp": _eval_ts,
            "model_class": type(model).__name__,
            "training_mode": cfg.training.get("mode", "global"),
            "seed": int(cfg.training.seed),
            "embedder": cfg.embedder.model_name,
            "data_dir": str(cfg.data.dir),
            "n_tables": n_tables,
            "n_questions": n_records,
            "hit_threshold": float(getattr(ds, "_hit_threshold", 0.5)),
            "trainable_params": _n_trainable,
            "total_params": _n_total,
            "encoder": {
                "hidden_size": int(cfg.model.hidden_size),
                "num_layers": int(cfg.model.num_layers),
                "num_heads": int(cfg.model.num_heads),
                "temperature": float(cfg.model.temperature),
                "ema_decay": float(cfg.model.ema_decay),
                "ablate_proj": bool(OmegaConf.select(cfg, "model.ablate_proj", default=False)),
            },
            "loss_weights": {
                "jepa": float(cfg.loss.jepa), "jepa_bar": float(cfg.loss.jepa_bar),
                "local": float(cfg.loss.local), "global": float(cfg.loss["global"]),
            },
        },
        "metrics": metrics,
        "table_retrieval": table_retrieval_metrics,
        "per_record": per_record,
    }
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print summary ─────────────────────────────────────────────────────────
    _COL = 16
    _VLABEL = {"raw": "RAW query", "sw": "SW query (stop-words removed)",
               "hit": "HIT query (high-informative tokens)"}
    for var in _VARIANTS:
        _mv = metrics[var]
        print(f"\n[eval] === {_VLABEL[var]} ===")
        print(f"[eval] {'Metric':<10}" + "".join(f"{sp:>{_COL}}" for sp in _SPACES))
        print("[eval] " + "-" * (10 + _COL * len(_SPACES)))
        for m in ("MRR", "Hit@1", "Hit@2", "Hit@3", "Hit@10", "Hit@20"):
            vals = "".join(
                f"{_mv[sp][m]:>{_COL}.4f}" if m == "MRR"
                else f"{str(_mv[sp][m]) + '%':>{_COL}}"
                for sp in _SPACES)
            print(f"[eval] {m:<10}{vals}")
        _trm = table_retrieval_metrics[var]
        print(f"[eval] Table retrieval (n={_trm['n_evaluated']}): "
              f"node@1={_trm['node']['Table@1']}%  tbl@1={_trm['tbl']['Table@1']}%")
    print(f"[eval] Results \u2192 {out_path}")
    return results


# ── Single training run ───────────────────────────────────────────────────────

def _train_one(cfg: DictConfig, dm: TableEmbedJePADataModule, out_dir: Path, run_label: str) -> float:
    """Build model, train, evaluate; return MRR ('both' space) as Optuna objective."""
    embed_dim_in = dm.embed_dim
    print(f"[train] embed_dim_in={embed_dim_in}  hidden_size={cfg.model.hidden_size}"
          f"  samples={len(dm._dataset)}")

    lightning_model = build_model(cfg, embed_dim_in)
    n_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params:,}")

    _model_slug = cfg.embedder.model_name.split("/")[-1].replace(" ", "_").replace(":", "#")
    _run_ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:8]
    ckpt_dir = out_dir / _model_slug / f"{_run_ts}_{_cfg_hash}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(ckpt_dir / "run_config.yaml"))
    print(f"[train] checkpoint dir   : {ckpt_dir}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_label}-{{epoch:02d}}-{{train_loss:.4f}}",
        monitor="train_loss_epoch", mode="min",
        save_top_k=cfg.training.save_total_limit, save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    _es_patience  = int(OmegaConf.select(cfg, "training.early_stopping_patience", default=0))
    _es_min_delta = float(OmegaConf.select(cfg, "training.early_stopping_min_delta", default=1e-4))
    early_stop_cb = EarlyStopping(
        monitor="train_loss_epoch", patience=_es_patience,
        min_delta=_es_min_delta, mode="min", verbose=False)

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
        callbacks=[checkpoint_cb, lr_monitor] + ([early_stop_cb] if _es_patience > 0 else []),
        deterministic=False,
    )
    ckpt_path = OmegaConf.select(cfg, "training.ckpt_path", default=None) or None
    if ckpt_path:
        print(f"[train] Resuming from checkpoint: {ckpt_path}")
    trainer.fit(lightning_model, datamodule=dm, ckpt_path=ckpt_path)

    if checkpoint_cb.best_model_path:
        print(f"[train] Loading best checkpoint for evaluation: {checkpoint_cb.best_model_path}")
        _best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=False)
        lightning_model.load_state_dict(_best_ckpt["state_dict"])

    final_path = ckpt_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(lightning_model.state_dict(), str(final_path / "model.pt"))
    OmegaConf.save(cfg, str(final_path / "run_config.yaml"))
    print(f"[train] Saved → {final_path}")

    results = evaluate_model(
        model=lightning_model, dm=dm, cfg=cfg,
        out_path=ckpt_dir / "eval_results.json", run_start_ts=_run_ts)
    try:
        mrr = float(results["metrics"]["raw"]["both"]["MRR"])
        if mrr > 0.0:
            return mrr
    except (KeyError, TypeError):
        pass
    return -float(trainer.callback_metrics.get("train_loss_epoch", float("inf")))


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: DictConfig) -> float:
    print("=" * 60)
    print("TableEmbedJePA — complex Adhesive tables (PyTorch Lightning)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    pl.seed_everything(cfg.training.seed, workers=True)
    _matmul_prec = OmegaConf.select(cfg, "matmul_precision", default="highest")
    torch.set_float32_matmul_precision(_matmul_prec)
    print(f"[train] float32 matmul precision : {_matmul_prec}")

    data_dir = Path(cfg.data.dir)
    if not data_dir.is_absolute():
        data_dir = (_HERE.parent / data_dir).resolve()

    out_base = Path(cfg.training.output_dir)
    if not out_base.is_absolute():
        out_base = (_HERE.parent / out_base).resolve()

    mode = cfg.training.get("mode", "global")

    _model_name = cfg.embedder.model_name
    _hit_thr_map = OmegaConf.select(cfg, "hit.thresholds", default={}) or {}
    _hit_threshold = float(
        _hit_thr_map.get(_model_name,
                         OmegaConf.select(cfg, "hit.default_threshold", default=0.5)))

    _dm_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.dataloader_num_workers,
        model_type=cfg.embedder.model_type,
        base_url=cfg.embedder.base_url,
        model_name=cfg.embedder.model_name,
        api_key=cfg.embedder.api_key,
        max_records=cfg.data.max_records,
        unique_tables_only=bool(OmegaConf.select(cfg, "data.unique_tables_only", default=False)),
        precompute=cfg.embedder.precompute,
        embed_batch_size=cfg.embedder.embed_batch_size,
        truncate_embed_dim=int(cfg.embedder.embed_dim) if cfg.embedder.embed_dim else None,
        cache_embeddings=bool(OmegaConf.select(cfg, "embedder.cache_embeddings", default=False)),
        embed_cache_dir=OmegaConf.select(cfg, "embedder.embed_cache_dir", default=None),
        cat_qry_template=OmegaConf.select(cfg, "query.cat_qry_template",
                                          default="{pivot_a} ... {pivot_b}?"),
        cat_qry_bar_template=OmegaConf.select(cfg, "query.cat_qry_bar_template",
                                              default="{pivot_b} ... {pivot_a}?"),
        upath_source=OmegaConf.select(cfg, "data.upath_source", default="walk"),
        hit_threshold=_hit_threshold,
    )

    if mode == "per_table":
        from dataset import _discover_table_ids
        table_ids = _discover_table_ids(
            data_dir, require_smp=(_dm_kwargs["upath_source"] == "walk"))
        print(f"[train] per_table mode — {len(table_ids)} tables found")
        scores: list[float] = []
        for i, tid in enumerate(table_ids, 1):
            print(f"\n{'='*60}\n[train] Table {i}/{len(table_ids)}: {tid}\n{'='*60}")
            dm = TableEmbedJePADataModule(data_dir=data_dir, filter_table_id=tid, **_dm_kwargs)
            dm.setup()
            n_samples = len(dm._dataset)
            if n_samples < cfg.training.batch_size:
                print(f"[train] Skipping — only {n_samples} samples "
                      f"(< batch_size {cfg.training.batch_size})")
                continue
            scores.append(_train_one(cfg, dm, out_dir=out_base / tid, run_label=tid))
        return float(sum(scores) / max(len(scores), 1)) if scores else float("-inf")

    dm = TableEmbedJePADataModule(data_dir=data_dir, **_dm_kwargs)
    dm.setup()
    return _train_one(cfg, dm, out_dir=out_base, run_label="table-jepa")


# Load .env (e.g. OLLAMA_IP) so ${oc.env:...} interpolations resolve at config time.
load_dotenv()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()

"""
finetune.py
Supervised fine-tuning of the CTA classifier for column type annotation.

Pipeline
────────
Stage 1 — Column embedding extraction  (offline, done once)
  Load the pretrained TableEmbedJePA encoder.
  For each (table, column) pair in the CTA dataset, collect all U-paths
  whose col_idx_a matches the column, encode them through the JEPA encoder,
  and mean-pool the resulting node_a representations → one [d] column embedding.
  If finetuning.pretrained_ckpt is null, use the raw LLM cell embeddings
  (mean of node_a from CTASMPDataset embed_cache, no encoder pass).

Stage 2 — Classifier training
  Train CTAClassifier (a linear head over the pooled column embedding)
  with BCEWithLogitsLoss (multi-label) and the Freebase type labels from type_vocab.

Run from the UTUEL repo root:
    python CTA/finetune.py

Fine-tune from a pretrained checkpoint:
    python CTA/finetune.py \
        finetuning.pretrained_ckpt=CTA/checkpoints/pretrain/last.ckpt

Override config on the CLI (Hydra):
    python CTA/finetune.py finetuning.epochs=30 classifier.pool_mode=attention

HPO sweep (sweep_optuna.yaml):
    python CTA/finetune.py --config-name sweep_optuna --multirun

Metrics reported on dev and test (multi-label, multi-class):
    Precision (micro / macro)   via torchmetrics MultilabelPrecision
    Recall    (micro / macro)   via torchmetrics MultilabelRecall
    F1        (micro / macro)   via torchmetrics MultilabelF1Score  ← checkpoint monitor
    threshold configured via eval.threshold in config.yaml (default 0.5)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_TRL  = _ROOT / "TRL-model"

for p in (str(_TRL), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Register TRL-model/model as a proper package so relative imports inside it work.
import importlib.util as _ilu

def _ensure_trl_model_pkg() -> None:
    pkg_name = "model"
    if pkg_name in sys.modules:
        return
    pkg_init = _TRL / "model" / "__init__.py"
    spec = _ilu.spec_from_file_location(pkg_name, pkg_init,
               submodule_search_locations=[str(_TRL / "model")])
    pkg  = _ilu.module_from_spec(spec)
    pkg.__path__ = [str(_TRL / "model")]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)

_ensure_trl_model_pkg()

from model  import TableEmbedJePA   # TRL-model/model/
from config import TableEmbedJePAConfig  # TRL-model/config.py

# CTA components
try:
    from CTA.dataset import CTASMPDataset
    from CTA.dataset_utils import load_type_vocab, resolve_data_paths
except ImportError:
    from dataset import CTASMPDataset
    from dataset_utils import load_type_vocab, resolve_data_paths


# ── Column embedding extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_column_embeddings(
    smp_ds: CTASMPDataset,
    type2idx: dict[str, int],
    col_types_per_table: dict[str, list[list[str]]],   # table_id → [n_cols][n_types]
    embed_mode: str = "column",   # 'column' | 'cell'
    include_header_emb: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract per-column (or per-cell) **raw LLM** node embeddings for every
    labelled column.  The JEPA encoder (if any) lives inside CTAClassifier and
    runs online in forward(), so freeze_encoder actually controls gradients.

    embed_mode='column'  (default)
        Mean-pool all raw node reps that belong to a column → one [d_llm] vector
        per (table, col) pair.  The encoder maps this to [d_model] in forward().

    embed_mode='cell'
        Keep each raw node rep separately.  The encoder runs per-cell in forward().
        At evaluation, per-cell logits are averaged (soft majority vote).

    include_header_emb=True
        Also include the column's own header node embedding in the mean-pool.
        In U-paths, the header of col_idx_a is stored at pivot_a (smp_idx[:,0]);
        the header of col_idx_b is stored at pivot_b (smp_idx[:,3]).
        In cell mode the header rep is appended as an extra entry per column.

    Returns
    -------
    embs      [N, d_llm]       N = n_cols (column) or n_cells (cell)
    multi_hot [N, num_classes] float multi-hot label matrix
    col_ids   [N]              int — unique column index each row belongs to
    """
    num_classes = len(type2idx)

    embs_list:      list[torch.Tensor] = []
    multi_hot_list: list[torch.Tensor] = []
    col_ids_list:   list[int]          = []
    col_counter = 0

    n_records = len(smp_ds.records)
    # U-paths only cover pairs (col_idx_a < col_idx_b), so the last column of a
    # table never appears as col_idx_a — it is always col_idx_b.  Build two
    # separate indices so every column is reachable.
    rec_col_a_to_js: dict[tuple[int, int], list[int]] = defaultdict(list)
    rec_col_b_to_js: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j, (rec_idx, up) in enumerate(smp_ds._samples):
        rec_col_a_to_js[(rec_idx, up.col_idx_a)].append(j)
        if up.col_idx_b != up.col_idx_a:
            rec_col_b_to_js[(rec_idx, up.col_idx_b)].append(j)

    skipped = 0
    for rec_idx in range(n_records):
        table_id = smp_ds.records[rec_idx]["table_id"]
        col_types: list[list[str]] = col_types_per_table.get(table_id, [])

        for col_idx, types_for_col in enumerate(col_types):
            if not types_for_col:
                continue

            hot = torch.zeros(num_classes, dtype=torch.float32)
            for t in types_for_col:
                if t in type2idx:
                    hot[type2idx[t]] = 1.0
            if hot.sum() == 0:
                skipped += 1
                continue

            js_a = rec_col_a_to_js.get((rec_idx, col_idx), [])
            js_b = rec_col_b_to_js.get((rec_idx, col_idx), [])
            if not js_a and not js_b:
                skipped += 1
                continue

            # Raw LLM embeddings (pre-encoder):
            # a-side U-paths: this column's cell is node_a → smp_idx[:, 1]
            # b-side U-paths: this column's cell is node_b → smp_idx[:, 2]
            # Header node:    pivot_a → smp_idx[:, 0] (a-side) / pivot_b → smp_idx[:, 3] (b-side)
            reps: list[torch.Tensor] = []
            if js_a:
                idx_a = torch.tensor(js_a, dtype=torch.long)
                reps.append(F.normalize(smp_ds._embed_cache[smp_ds._smp_idx[idx_a, 1]], dim=-1))
            if js_b:
                idx_b = torch.tensor(js_b, dtype=torch.long)
                reps.append(F.normalize(smp_ds._embed_cache[smp_ds._smp_idx[idx_b, 2]], dim=-1))
            cell_reps = torch.cat(reps, dim=0)  # [n_upaths_for_col, d_llm]

            # Header embedding: take from first available U-path (same text for all)
            if include_header_emb:
                if js_a:
                    hdr_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_a[0:1], 0]]
                else:
                    hdr_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_b[0:1], 3]]
                hdr_rep = F.normalize(hdr_raw, dim=-1)  # [1, d_llm]

            if embed_mode == "column":
                col_rep = cell_reps.mean(dim=0)
                if include_header_emb:
                    col_rep = torch.stack([col_rep, hdr_rep[0]]).mean(dim=0)
                embs_list.append(col_rep)
                multi_hot_list.append(hot)
                col_ids_list.append(col_counter)
            else:  # cell — one entry per U-path node rep
                for rep in cell_reps:
                    embs_list.append(rep)
                    multi_hot_list.append(hot)
                    col_ids_list.append(col_counter)
                if include_header_emb:  # header as an extra cell-level entry
                    embs_list.append(hdr_rep[0])
                    multi_hot_list.append(hot)
                    col_ids_list.append(col_counter)
            col_counter += 1

    avg_lpc = sum(h.sum().item() for h in multi_hot_list) / max(col_counter, 1)
    print(
        f"[CTA][extract][{embed_mode}] {col_counter} columns  {len(embs_list)} samples "
        f"({skipped} skipped)  avg_labels/col={avg_lpc:.2f}"
    )
    if not embs_list:
        raise RuntimeError(
            f"[CTA][extract] No labelled columns survived for split. "
            f"{skipped} columns were skipped (no matching U-paths or no types in vocab). "
            "Check: (1) type_vocab covers your data, (2) max_rows_per_table >= 2 so "
            "U-paths can be generated, (3) data.folder points to the correct dataset."
        )
    return (
        torch.stack(embs_list),
        torch.stack(multi_hot_list),
        torch.tensor(col_ids_list, dtype=torch.long),
    )


# ── Multi-label classifier Lightning module ───────────────────────────────────

class CTAClassifier(pl.LightningModule):
    """
    Multi-label, multi-class CTA classifier with optional end-to-end encoder.

    When ``encoder`` is provided the forward pass is:
        raw_llm_emb  →  input_projection  →  [CLS, proj]  →  transformer_encoder
        →  CLS output  →  head  →  logits
    When ``encoder`` is None:
        raw_llm_emb  →  head  →  logits

    ``freeze_encoder=True``  : encoder weights are frozen — only the head trains.
    ``freeze_encoder=False`` : encoder trains jointly with the head.

    Metrics (torchmetrics, threshold-based — threshold from eval.threshold):
        accuracy_micro
        precision_micro, precision_macro
        recall_micro,    recall_macro
        f1_micro,        f1_macro       ← checkpoint monitor
    """

    def __init__(
        self,
        embed_dim: int,                      # d_llm: raw LLM embedding size
        head_dim: int,                       # d_model (encoder out) or embed_dim if no encoder
        num_classes: int,
        intermediate_size: int | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 20,
        label_smoothing: float = 0.0,
        threshold: float = 0.5,
        # ── encoder ──────────────────────────────────────────────────────────
        encoder: "TableEmbedJePA | None" = None,
        encoder_config_dict: dict | None = None,  # saved in hparams for ckpt reload
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        # ── Encoder submodule ─────────────────────────────────────────────────
        # On the first call encoder is the live object; on checkpoint reload
        # encoder=None but encoder_config_dict is restored from hparams so we
        # reconstruct an empty skeleton that load_state_dict fills.
        if encoder is not None:
            self.encoder = encoder
        elif encoder_config_dict is not None:
            _d = dict(encoder_config_dict)
            _ablate = _d.pop("_ablate_proj", False)
            _enc_cfg = TableEmbedJePAConfig(**_d)
            self.encoder = TableEmbedJePA(config=_enc_cfg, ablate_proj=_ablate)
        else:
            self.encoder = None

        if self.encoder is not None and freeze_encoder:
            self.encoder.requires_grad_(False)

        # ── Classification head ───────────────────────────────────────────────
        if intermediate_size:
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(head_dim),
                torch.nn.Linear(head_dim, intermediate_size),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(intermediate_size, num_classes),
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(head_dim),
                torch.nn.Linear(head_dim, num_classes),
            )
        self.bce = torch.nn.BCEWithLogitsLoss()

        def _metrics() -> torch.nn.ModuleDict:
            return torch.nn.ModuleDict({
                "accuracy": MultilabelAccuracy(
                    num_labels=num_classes, threshold=threshold, average="micro"),
                "precision_micro": MultilabelPrecision(
                    num_labels=num_classes, threshold=threshold, average="micro"),
                "precision_macro": MultilabelPrecision(
                    num_labels=num_classes, threshold=threshold, average="macro"),
                "recall_micro": MultilabelRecall(
                    num_labels=num_classes, threshold=threshold, average="micro"),
                "recall_macro": MultilabelRecall(
                    num_labels=num_classes, threshold=threshold, average="macro"),
                "f1_micro": MultilabelF1Score(
                    num_labels=num_classes, threshold=threshold, average="micro"),
                "f1_macro": MultilabelF1Score(
                    num_labels=num_classes, threshold=threshold, average="macro"),
            })

        self.train_metrics = _metrics()
        self.val_metrics   = _metrics()
        self.test_metrics  = _metrics()
        self._split_metrics = {
            "train": self.train_metrics,
            "val":   self.val_metrics,
            "test":  self.test_metrics,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            # x: [B, d_llm]  →  treat as a single-token sequence + CLS
            x = x.unsqueeze(1)                                               # [B, 1, d_llm]
            proj = self.encoder.input_projection(x)                          # [B, 1, d_model]
            cls  = self.encoder.input_projection(
                       self.encoder.cls_token                                # [1, 1, d_llm]
                   ).expand(x.size(0), -1, -1)                               # [B, 1, d_model]
            seq  = torch.cat([cls, proj], dim=1)                             # [B, 2, d_model]
            enc_out, _, _ = self.encoder.transformer_encoder(seq)
            x = enc_out[:, 0, :]                                             # CLS → [B, d_model]
        return self.head(x)

    def _step(self, batch: tuple, split: str) -> torch.Tensor:
        col_embs, multi_hot = batch          # [B, d], [B, C] float
        logits = self(col_embs)              # [B, C]
        loss   = self.bce(logits, multi_hot)
        probs  = torch.sigmoid(logits)       # [B, C]
        target = multi_hot.long()            # [B, C] int

        # Torchmetrics: Precision / Recall / F1
        mdict = self._split_metrics[split]
        for name, metric in mdict.items():
            self.log(
                f"{split}/{name}", metric(probs, target),
                prog_bar=(name in ("f1_micro", "f1_macro") and split == "val"),
                on_step=False, on_epoch=True,
            )

        self.log(f"{split}/loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _):       self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """
    Fine-tune CTA classifier and evaluate on dev and test splits.

    Returns dev F1-micro (Optuna objective, direction=maximize).
    """
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    pl.seed_everything(cfg.finetuning.seed, workers=True)

    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold  = float(OmegaConf.select(cfg, "eval.threshold", default=0.5))
    embed_mode         = str(OmegaConf.select(cfg, "classifier.embed_mode",         default="column"))
    smp_source         = str(OmegaConf.select(cfg, "classifier.smp_source",         default="smp"))
    include_header_emb = bool(OmegaConf.select(cfg, "classifier.include_header_emb", default=False))
    if embed_mode not in ("column", "cell"):
        raise ValueError(f"classifier.embed_mode must be 'column' or 'cell', got {embed_mode!r}")
    if smp_source not in ("smp", "smp_bar", "both"):
        raise ValueError(f"classifier.smp_source must be 'smp', 'smp_bar', or 'both', got {smp_source!r}")

    # ── Type vocabulary ────────────────────────────────────────────────────────
    _paths = resolve_data_paths(cfg.data)
    type2idx, _ = load_type_vocab(_paths["type_vocab"])
    num_classes = len(type2idx)

    # ── Pretrained encoder (optional) ─────────────────────────────────────────
    pretrained_ckpt = OmegaConf.select(cfg, "finetuning.pretrained_ckpt")
    # ── Encoder setup ──────────────────────────────────────────────────────────
    # freeze_encoder=True  + pretrained_ckpt=null  → NO encoder (raw LLM embs → head)
    # freeze_encoder=False + pretrained_ckpt=null  → random-init encoder, trained jointly
    # freeze_encoder=False + pretrained_ckpt=path  → pretrained encoder, trained jointly
    # freeze_encoder=True  + pretrained_ckpt=path  → pretrained encoder, frozen (head only)
    freeze_encoder = bool(OmegaConf.select(cfg, "classifier.freeze_encoder", default=False))
    use_encoder    = pretrained_ckpt or not freeze_encoder   # skip only when null+frozen

    _hs = cfg.model.hidden_size
    _nh = max(1, min(cfg.model.num_heads, _hs // 64))
    while _hs % _nh != 0 and _nh > 1:
        _nh -= 1
    _ablate = bool(OmegaConf.select(cfg, "model.ablate_proj", default=False))
    _enc_cfg = TableEmbedJePAConfig(
        hidden_size=_hs,
        num_hidden_layers=cfg.model.num_layers,
        num_attention_heads=_nh,
        intermediate_size=cfg.model.intermediate_size or (_hs * 4),
        attention_probs_dropout_prob=cfg.model.attention_dropout,
        hidden_dropout_prob=cfg.model.hidden_dropout,
        layer_norm_eps=cfg.model.layer_norm_eps,
        embedding_dim=int(cfg.embedder.embed_dim),
        tempeture=cfg.model.temperature,
        beta=cfg.model.beta,
    )

    encoder: "TableEmbedJePA | None" = None
    if use_encoder:
        if pretrained_ckpt:
            encoder = TableEmbedJePA.load_from_checkpoint(
                pretrained_ckpt, map_location="cpu", config=_enc_cfg
            )
            print(f"[CTA][finetune] loaded pretrained encoder from {pretrained_ckpt}")
        else:
            encoder = TableEmbedJePA(config=_enc_cfg, ablate_proj=_ablate)
            print("[CTA][finetune] random-init encoder (no pretrained_ckpt)")
        if freeze_encoder:
            encoder.requires_grad_(False)
            encoder.eval()

    # ── Build per-split TensorDatasets ────────────────────────────────────────
    # +1 because max_rows_* is user-facing as "data rows" but the header row
    # is stored separately — adding 1 ensures the user's count is fully honoured.
    _MAX_ROWS = {
        "train": (cfg.data.get("max_rows_train") + 1) if cfg.data.get("max_rows_train") is not None else None,
        "dev":   (cfg.data.get("max_rows_dev")   + 1) if cfg.data.get("max_rows_dev")   is not None else None,
        "test":  (cfg.data.get("max_rows_test")  + 1) if cfg.data.get("max_rows_test")  is not None else None,
    }

    def _make_smp_ds(data_path: str, split: str) -> CTASMPDataset:
        return CTASMPDataset(
            data_path=data_path,
            model_type=cfg.embedder.model_type,
            base_url=cfg.embedder.get("base_url"),
            model_name=cfg.embedder.get("model_name"),
            api_key=cfg.embedder.get("api_key"),
            max_records=cfg.data.get("max_records"),
            max_rows_per_table=_MAX_ROWS[split],
            use_graph_walks=cfg.smp.use_graph_walks,
            num_walks=cfg.smp.num_walks,
            chunk_size=cfg.smp.chunk_size,
            precompute=True,
            embed_batch_size=cfg.embedder.embed_batch_size,
            cache_embeddings=cfg.embedder.cache_embeddings,
            embed_cache_dir=cfg.embedder.get("embed_cache_dir"),
            cat_qry_template=cfg.query.cat_qry_template,
            cat_qry_bar_template=cfg.query.cat_qry_bar_template,
        )

    def _load_col_types(data_path: str) -> dict[str, list[list[str]]]:
        records = json.loads(Path(data_path).read_text(encoding="utf-8"))
        max_r = cfg.data.get("max_records")
        if max_r is not None:
            records = records[:max_r]
        # Key by table_id (rec[0]) to avoid positional misalignment with smp_ds.records
        # (CTASMPDataset silently drops records with <2 headers or empty rows)
        return {str(r[0]): r[7] if len(r) > 7 else [] for r in records}

    splits: dict[str, TensorDataset] = {}
    col_ids_map: dict[str, torch.Tensor] = {}
    for split, path_key in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        path = _paths[path_key]
        smp_ds    = _make_smp_ds(path, split)
        col_types = _load_col_types(path)
        embs, multi_hot, col_ids = extract_column_embeddings(
            smp_ds, type2idx, col_types,
            embed_mode=embed_mode,
            include_header_emb=include_header_emb,
        )
        splits[split]   = TensorDataset(embs, multi_hot)
        col_ids_map[split] = col_ids
        n_cols = int(col_ids.max().item()) + 1 if len(col_ids) else 0
        print(
            f"[CTA][finetune][{split}] {n_cols} columns  {len(embs)} samples "
            f"d={embs.shape[1]}  num_classes={num_classes}  embed_mode={embed_mode}"
        )

    # embed_dim = d_llm (raw LLM); head_dim = d_model (encoder output) or d_llm if no encoder
    embed_dim = splits["train"].tensors[0].shape[1]
    head_dim  = cfg.model.hidden_size if encoder is not None else embed_dim

    # Encoder config dict saved into hparams so CTAClassifier.load_from_checkpoint
    # can reconstruct the encoder skeleton and then fill it from the state_dict.
    encoder_config_dict: dict | None = None
    if encoder is not None:
        encoder_config_dict = {
            "hidden_size":                  _hs,
            "num_hidden_layers":            cfg.model.num_layers,
            "num_attention_heads":          _nh,
            "intermediate_size":            cfg.model.intermediate_size or (_hs * 4),
            "attention_probs_dropout_prob": cfg.model.attention_dropout,
            "hidden_dropout_prob":          cfg.model.hidden_dropout,
            "layer_norm_eps":               cfg.model.layer_norm_eps,
            "embedding_dim":                int(cfg.embedder.embed_dim),
            "tempeture":                    cfg.model.temperature,
            "beta":                         cfg.model.beta,
            "_ablate_proj":                 _ablate,
        }

    def _loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            splits[split],
            batch_size=cfg.finetuning.batch_size,
            shuffle=shuffle,
            num_workers=cfg.finetuning.dataloader_num_workers,
            pin_memory=True,
        )

    # ── Classifier ────────────────────────────────────────────────────────────
    model = CTAClassifier(
        embed_dim=embed_dim,
        head_dim=head_dim,
        num_classes=num_classes,
        intermediate_size=OmegaConf.select(cfg, "classifier.intermediate_size") or None,
        lr=cfg.finetuning.lr,
        weight_decay=cfg.finetuning.weight_decay,
        max_epochs=cfg.finetuning.epochs,
        label_smoothing=cfg.finetuning.label_smoothing,
        threshold=threshold,
        encoder=encoder,
        encoder_config_dict=encoder_config_dict,
        freeze_encoder=freeze_encoder,
    )
    enc_mode_str = (
        f"pretrained+{'frozen' if freeze_encoder else 'joint'}"
        if encoder is not None else "none"
    )
    print(
        f"[CTA][finetune] embed_dim={embed_dim}→head_dim={head_dim}  num_classes={num_classes}"
        f"  embed_mode={embed_mode}  include_header_emb={include_header_emb}"
        f"  encoder={enc_mode_str}  threshold={threshold}"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    out_dir = Path(cfg.finetuning.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename="cta-finetune-{epoch:02d}-{val/f1_micro:.4f}",
            monitor="val/f1_micro",
            mode="max",
            save_last=True,
            save_top_k=cfg.finetuning.save_total_limit,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if cfg.finetuning.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/f1_micro",
                patience=cfg.finetuning.early_stopping_patience,
                min_delta=cfg.finetuning.early_stopping_min_delta,
                mode="max",
            )
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.finetuning.epochs,
        precision=(
            "16-mixed" if cfg.finetuning.fp16
            else "bf16-mixed" if cfg.finetuning.bf16
            else "32-true"
        ),
        gradient_clip_val=cfg.finetuning.max_grad_norm,
        log_every_n_steps=cfg.finetuning.logging_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=str(out_dir),
    )

    trainer.fit(
        model,
        train_dataloaders=_loader("train", shuffle=True),
        val_dataloaders=_loader("dev",   shuffle=False),
        ckpt_path=cfg.finetuning.get("ckpt_path") or None,
    )

    # ── Offline evaluation on best checkpoint ─────────────────────────────────
    eval_dir = Path(cfg.eval.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt  = callbacks[0].best_model_path or str(out_dir / "last.ckpt")
    best_model = CTAClassifier.load_from_checkpoint(best_ckpt)
    best_model.eval().to(device)

    metrics: dict = {}

    for split in ("dev", "test"):
        embs, multi_hot = splits[split].tensors
        col_ids = col_ids_map[split]

        with torch.no_grad():
            logits = best_model(embs.to(device)).cpu()

        # Soft majority voting in cell mode: average per-cell logits → column logit
        if embed_mode == "cell":
            unique_cols = torch.unique(col_ids)
            logits    = torch.stack([logits[col_ids == c].mean(0)    for c in unique_cols])
            multi_hot = torch.stack([multi_hot[col_ids == c][0]      for c in unique_cols])

        probs      = torch.sigmoid(logits)
        target_int = multi_hot.long()

        split_metrics: dict = {}

        # Precision / Recall / F1 (micro & macro) via torchmetrics
        split_metrics["accuracy"] = float(
            MultilabelAccuracy(num_labels=num_classes, threshold=threshold, average="micro")
            (probs, target_int)
        )
        for avg in ("micro", "macro"):
            split_metrics[f"precision_{avg}"] = float(
                MultilabelPrecision(num_labels=num_classes, threshold=threshold, average=avg)
                (probs, target_int)
            )
            split_metrics[f"recall_{avg}"] = float(
                MultilabelRecall(num_labels=num_classes, threshold=threshold, average=avg)
                (probs, target_int)
            )
            split_metrics[f"f1_{avg}"] = float(
                MultilabelF1Score(num_labels=num_classes, threshold=threshold, average=avg)
                (probs, target_int)
            )
        split_metrics["threshold"] = threshold

        out_file = eval_dir / f"cta_{split}_metrics.json"
        out_file.write_text(json.dumps(split_metrics, indent=2), encoding="utf-8")
        row = "  ".join(
            f"{k}={v:.4f}" for k, v in split_metrics.items() if k != "threshold"
        )
        print(f"[CTA][finetune][{split}]  {row}  (thr={threshold})  -> {out_file}")
        metrics[split] = split_metrics

    return metrics["dev"].get("f1_micro", 0.0)   # Optuna objective


if __name__ == "__main__":
    main()

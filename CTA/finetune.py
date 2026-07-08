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

import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict
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
    embed_mode: str = "column",   # 'column' | 'cell' | 'cell_header'
    smp_source: str = "smp",     # 'smp' | 'smp_bar' | 'both'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract raw LLM U-path embeddings for every labelled column.

    The JEPA encoder runs **online** inside ``CTAClassifier.forward``, which
    sees the full 4-token U-path [pivot_a, node_a, node_b, pivot_b] and then
    selects output positions based on ``smp_source`` / ``embed_mode``.
    This function only extracts the raw LLM cache; no pooling or selection here.

    smp_source controls which U-paths are eligible for a column:
        'smp'     → column must appear as col_a  (node_a = this col's cell, position [1])
        'smp_bar' → column must appear as col_b  (node_b = this col's cell, position [2])
        'both'    → all U-paths involving this column (a-side and b-side)

    embed_mode='column'
        One sample per (table, column): position-wise mean-pool across eligible U-paths
        → [4, d_llm].

    embed_mode='cell'
        One sample per eligible U-path → [4, d_llm] per path.

    embed_mode='cell_header'
        One sample per eligible U-path (like 'cell'), but the header token is
        used as the column representative: pivot_a when smp, pivot_b when smp_bar.

    The 4 positions are always in canonical U-path order:
        [0] pivot_a  — column header of the a-side column
        [1] node_a   — cell value  of the a-side column  (← this col when smp)
        [2] node_b   — cell value  of the b-side column  (← this col when smp_bar)
        [3] pivot_b  — column header of the b-side column

    Returns
    -------
    smp_embs  [N, 4, d_llm]    raw embeddings: [pivot_a, node_a, node_b, pivot_b]
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

            # Filter paths by smp_source: each source only uses the paths
            # where this column plays the correct role.
            #   smp     → col_a role → js_a only  (node_a at position [1] = this col)
            #   smp_bar → col_b role → js_b only  (node_b at position [2] = this col)
            #   both    → all paths
            if smp_source == "smp":
                js_b = []
            elif smp_source == "smp_bar":
                js_a = []
            # both: keep js_a and js_b as-is

            if not js_a and not js_b:
                skipped += 1
                continue

            # _embed_cache[_smp_idx[j]] → [4, d_llm] for U-path j.
            # Advanced indexing: _embed_cache[_smp_idx[idx]] → [n, 4, d_llm].
            if embed_mode == "column":
                # Collect all U-paths for this column (a-side and b-side) then
                # mean-pool each position independently → [4, d_llm].
                all_paths: list[torch.Tensor] = []
                if js_a:
                    idx_a = torch.tensor(js_a, dtype=torch.long)
                    all_paths.append(smp_ds._embed_cache[smp_ds._smp_idx[idx_a]])  # [n_a, 4, d]
                if js_b:
                    idx_b = torch.tensor(js_b, dtype=torch.long)
                    all_paths.append(smp_ds._embed_cache[smp_ds._smp_idx[idx_b]])  # [n_b, 4, d]
                stacked = torch.cat(all_paths, dim=0)              # [n_paths, 4, d_llm]
                pooled  = F.normalize(stacked.mean(dim=0), dim=-1) # [4, d_llm]
                embs_list.append(pooled)
                multi_hot_list.append(hot)
                col_ids_list.append(col_counter)
            else:  # cell / cell_header — one [4, d_llm] entry per U-path
                if js_a:
                    idx_a  = torch.tensor(js_a, dtype=torch.long)
                    paths_a = F.normalize(
                        smp_ds._embed_cache[smp_ds._smp_idx[idx_a]], dim=-1
                    )  # [n_a, 4, d_llm]
                    for i in range(paths_a.size(0)):
                        embs_list.append(paths_a[i])
                        multi_hot_list.append(hot)
                        col_ids_list.append(col_counter)
                if js_b:
                    idx_b  = torch.tensor(js_b, dtype=torch.long)
                    paths_b = F.normalize(
                        smp_ds._embed_cache[smp_ds._smp_idx[idx_b]], dim=-1
                    )  # [n_b, 4, d_llm]
                    for i in range(paths_b.size(0)):
                        embs_list.append(paths_b[i])
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
        torch.stack(embs_list),       # [N, 4, d_llm]
        torch.stack(multi_hot_list),  # [N, C]
        torch.tensor(col_ids_list, dtype=torch.long),
    )


# ── Multi-label classifier Lightning module ───────────────────────────────────

class CTAClassifier(pl.LightningModule):
    """
    Multi-label, multi-class CTA classifier with optional end-to-end encoder.

    Input to forward(): [B, 4, d_llm] — [pivot_a, node_a, node_b, pivot_b] raw LLM embeddings.

    When ``encoder`` is provided:
        [B, 4, d_llm]  →  input_projection
        →  [CLS, pivot_a, node_a, node_b, pivot_b]  (5 tokens)  →  transformer_encoder
        →  select positions per embed_mode + smp_source  →  head  →  logits
    When ``encoder`` is None:
        select raw LLM positions per embed_mode + smp_source  →  head  →  logits

    Position selection rules (enc_out positions: 0=CLS, 1=pivot_a, 2=node_a, 3=node_b, 4=pivot_b):
        column + smp      → avg(pivot_a, node_a)                → [B, d]  → head → [B, C]
        column + smp_bar  → avg(pivot_b, node_b)                → [B, d]  → head → [B, C]
        column + both     → stack(avg(pa,na), avg(pb,nb))       → [2B, d] → head → [2B, C] → mean → [B, C]
        cell   + smp      → node_a                              → [B, d]  → head → [B, C]
        cell   + smp_bar  → node_b                              → [B, d]  → head → [B, C]
        cell   + both     → stack(node_a, node_b)               → [2B, d] → head → [2B, C] → mean → [B, C]
        cell_header + smp      → pivot_a                         → [B, d]  → head → [B, C]
        cell_header + smp_bar  → pivot_b                         → [B, d]  → head → [B, C]
        cell_header + both     → stack(pivot_a, pivot_b)         → [2B, d] → head → [2B, C] → mean → [B, C]

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
        head_dim: int,                       # d_model (encoder out) or d_llm if no encoder
        num_classes: int,
        embed_mode: str = "column",          # 'column' | 'cell' | 'cell_header'
        smp_source: str = "smp",             # 'smp' | 'smp_bar' | 'both'
        intermediate_size: int | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 20,
        threshold: float = 0.5,
        # ── encoder ──────────────────────────────────────────────────────────
        encoder: "TableEmbedJePA | None" = None,
        encoder_config_dict: dict | None = None,  # saved in hparams for ckpt reload
        freeze_encoder: bool = False,
        # ── loss ─────────────────────────────────────────────────────────────
        pos_weight: "str | float | None" = None,   # None | float | 'inline'
        pos_weight_max: float = 100.0,             # cap for 'inline' per-class weights
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

        # ── BCE positive-class weighting (class imbalance) ────────────────────
        #   None     → no weighting (plain BCE)
        #   float    → fixed scalar weight applied to every positive
        #   'inline' → per-batch per-class weight = (#neg / #pos) computed inline
        #              from the batch target inside _step (capped by pos_weight_max)
        self.pos_weight_cfg = pos_weight
        self.pos_weight_max = float(pos_weight_max)
        self._pos_weight_scalar = (
            float(pos_weight)
            if isinstance(pos_weight, (int, float)) and not isinstance(pos_weight, bool)
            else None
        )

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
        # x: expected [B, 4, d_llm]  —  [pivot_a, node_a, node_b, pivot_b]
        # Backward-compatibility: older extraction code may still yield [B, d_llm].
        if x.dim() == 2:
            if not getattr(self, "_warned_legacy_input", False):
                print(
                    "[CTA][forward] WARNING: received legacy 2D input [B, d]. "
                    "Expanding to [B, 4, d] by repetition for compatibility.",
                    flush=True,
                )
                self._warned_legacy_input = True
            x = x.unsqueeze(1).expand(-1, 4, -1)
        elif x.dim() == 3 and x.size(1) == 1:
            if not getattr(self, "_warned_single_token_input", False):
                print(
                    "[CTA][forward] WARNING: received single-token input [B, 1, d]. "
                    "Expanding to [B, 4, d] by repetition for compatibility.",
                    flush=True,
                )
                self._warned_single_token_input = True
            x = x.expand(-1, 4, -1)
        elif x.dim() != 3 or x.size(1) != 4:
            raise ValueError(
                f"CTAClassifier.forward expected input shape [B, 4, d], got {tuple(x.shape)}"
            )

        if self.encoder is not None:
            enc_device = next(self.encoder.transformer_encoder.parameters()).device
            if x.device != enc_device:
                x = x.to(enc_device)
            B = x.size(0)
            proj = self.encoder.input_projection(x)                     # [B, 4, d_model]
            cls_tok = self.encoder.cls_token.to(device=x.device, dtype=x.dtype)
            cls  = self.encoder.input_projection(
                       cls_tok                                           # [1, 1, d_llm]
                   ).expand(B, -1, -1)                                  # [B, 1, d_model]
            seq     = torch.cat([cls, proj], dim=1)                     # [B, 5, d_model]
            enc_out, _, _ = self.encoder.transformer_encoder(seq)       # [B, 5, d_model]
            # positions: 0=CLS  1=pivot_a  2=node_a  3=node_b  4=pivot_b
            pa, na = enc_out[:, 1, :], enc_out[:, 2, :]
            nb, pb = enc_out[:, 3, :], enc_out[:, 4, :]
        else:
            pa, na = x[:, 0, :], x[:, 1, :]
            nb, pb = x[:, 2, :], x[:, 3, :]

        em  = self.hparams.embed_mode
        src = self.hparams.smp_source
        if em == "column":
            if src == "smp":
                rep = (pa + na) * 0.5                                     # [B, d]
            elif src == "smp_bar":
                rep = (pb + nb) * 0.5                                     # [B, d]
            else:  # both → stack → [2B, d]
                rep = torch.cat([(pa + na) * 0.5, (pb + nb) * 0.5], dim=0)
        elif em == "cell_header":
            # use only the header token (pivot) as the column representative
            if src == "smp":
                rep = pa                                                   # [B, d]
            elif src == "smp_bar":
                rep = pb                                                   # [B, d]
            else:  # both → stack → [2B, d]
                rep = torch.cat([pa, pb], dim=0)
        else:  # cell
            if src == "smp":
                rep = na                                                   # [B, d]
            elif src == "smp_bar":
                rep = nb                                                   # [B, d]
            else:  # both → stack → [2B, d]
                rep = torch.cat([na, nb], dim=0)
        logits = self.head(rep)                                            # [B, C] or [2B, C]
        if src == "both":
            B = x.size(0)
            logits = logits.view(2, B, -1).mean(dim=0)                    # [B, C]
        return logits

    def _batch_pos_weight(self, target: torch.Tensor) -> "torch.Tensor | None":
        """BCE pos_weight computed inline from the batch target.

        Returns None when weighting is disabled.  For 'inline' mode the weight
        for class c is (#negatives / #positives) within this batch; classes
        absent from the batch get a neutral weight of 1.0.
        """
        mode = self.pos_weight_cfg
        if mode is None:
            return None
        if self._pos_weight_scalar is not None:
            return torch.full(
                (target.shape[1],), self._pos_weight_scalar,
                dtype=target.dtype, device=target.device,
            )
        if isinstance(mode, str) and mode.lower() in ("inline", "auto", "batch"):
            pos = target.sum(dim=0)                       # [C] positives in batch
            neg = target.shape[0] - pos                   # [C] negatives in batch
            pw  = (neg / pos.clamp(min=1.0)).clamp(max=self.pos_weight_max)
            pw[pos == 0] = 1.0                            # classes absent in batch → neutral
            return pw
        return None

    def _step(self, batch: tuple, split: str) -> torch.Tensor:
        col_embs, multi_hot = batch          # [B, 4, d_llm], [B, C] float
        logits = self(col_embs)              # [B, C]
        pos_weight = self._batch_pos_weight(multi_hot)
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, multi_hot, pos_weight=pos_weight
            )
        else:
            loss = self.bce(logits, multi_hot)
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
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr
        # )
        # return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        # + add this scheduler
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",        # maximizing F1
            patience=3,
            factor=0.5,
            min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch","monitor": "val/f1_micro"}}


# ── Main ──────────────────────────────────────────────────────────────────────
# Load .env (e.g. OLLAMA_IP) so ${oc.env:...} interpolations resolve at config time.
load_dotenv()

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
    if embed_mode not in ("column", "cell", "cell_header"):
        raise ValueError(f"classifier.embed_mode must be 'column', 'cell' or 'cell_header', got {embed_mode!r}")
    if smp_source not in ("smp", "smp_bar", "both"):
        raise ValueError(f"classifier.smp_source must be 'smp', 'smp_bar', or 'both', got {smp_source!r}")

    # ── Type vocabulary ────────────────────────────────────────────────────────
    _paths = resolve_data_paths(cfg.data)
    type2idx, _ = load_type_vocab(_paths["type_vocab"])
    num_classes = len(type2idx)

    # ── Pretrained encoder (optional) ─────────────────────────────────────────
    pretrained_ckpt = OmegaConf.select(cfg, "finetuning.pretrained_ckpt")
    pretrain_cfg = cfg  # default: use current config

    if pretrained_ckpt:
        # Resolve the checkpoint file and its sibling run_config.yaml.
        # A directory → last.ckpt inside it; a file → run_config.yaml alongside.
        ckpt_path = Path(pretrained_ckpt)
        if ckpt_path.is_dir():
            pretrain_run_cfg = ckpt_path / "run_config.yaml"
            pretrained_ckpt = str(ckpt_path / "last.ckpt")
            print(f"[CTA][finetune] auto-resolved checkpoint folder to {pretrained_ckpt}")
        else:
            pretrain_run_cfg = ckpt_path.parent / "run_config.yaml"

        if pretrain_run_cfg.exists():
            pretrain_cfg = OmegaConf.load(str(pretrain_run_cfg))
            print(f"[CTA][finetune] loaded pretraining config from {pretrain_run_cfg}")

            # The pretrained encoder architecture is fixed by the checkpoint, so
            # its model config OVERRIDES any model.* passed on the CLI / finetune
            # config (e.g. model.num_layers=1 arg vs. 3 in the checkpoint).  We
            # copy the whole pretrain model block into cfg so every downstream
            # consumer stays consistent: the encoder build, the hparams
            # encoder_config_dict used to rebuild the skeleton on reload, the
            # cfg hash, and the re-saved run_config.yaml.
            pretrain_model = OmegaConf.select(pretrain_cfg, "model")
            if pretrain_model is not None:
                with open_dict(cfg):
                    for _k, _v in pretrain_model.items():
                        _old = OmegaConf.select(cfg, f"model.{_k}")
                        if _old != _v:
                            print(
                                f"[CTA][finetune] override model.{_k}: "
                                f"{_old!r} → {_v!r} (from pretrain checkpoint)"
                            )
                        cfg.model[_k] = _v
        else:
            print(
                f"[CTA][finetune] WARNING: no run_config.yaml found next to "
                f"{pretrained_ckpt}; using model.* from the finetune config as-is"
            )
    
    # ── Encoder setup ──────────────────────────────────────────────────────────
    # freeze_encoder=True  + pretrained_ckpt=null  → NO encoder (raw LLM embs → head)
    # freeze_encoder=False + pretrained_ckpt=null  → random-init encoder, trained jointly
    # freeze_encoder=False + pretrained_ckpt=path  → pretrained encoder, trained jointly
    # freeze_encoder=True  + pretrained_ckpt=path  → pretrained encoder, frozen (head only)
    freeze_encoder = bool(OmegaConf.select(cfg, "classifier.freeze_encoder", default=False))
    use_encoder    = pretrained_ckpt or not freeze_encoder   # skip only when null+frozen

    _hs = pretrain_cfg.model.hidden_size
    _nh = max(1, min(pretrain_cfg.model.num_heads, _hs // 64))
    while _hs % _nh != 0 and _nh > 1:
        _nh -= 1
    _ablate = bool(OmegaConf.select(pretrain_cfg, "model.ablate_proj", default=False))
    _enc_cfg = TableEmbedJePAConfig(
        hidden_size=_hs,
        num_hidden_layers=pretrain_cfg.model.num_layers,
        num_attention_heads=_nh,
        intermediate_size=pretrain_cfg.model.intermediate_size or (_hs * 4),
        attention_probs_dropout_prob=pretrain_cfg.model.attention_dropout,
        hidden_dropout_prob=pretrain_cfg.model.hidden_dropout,
        layer_norm_eps=pretrain_cfg.model.layer_norm_eps,
        embedding_dim=int(cfg.embedder.embed_dim),
        tempeture=pretrain_cfg.model.temperature,
        beta=pretrain_cfg.model.beta,
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
    _MAX_ROWS = {
        "train": cfg.data.get("max_rows_train"),
        "dev":   cfg.data.get("max_rows_dev"),
        "test":  cfg.data.get("max_rows_test"),
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
            expected_embed_dim=int(cfg.embedder.embed_dim) if cfg.embedder.get("embed_dim") else None,
            use_global_cache=cfg.embedder.get("use_global_cache", False),
            include_query=cfg.embedder.get("include_query", False),
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
            smp_source=smp_source,
        )
        splits[split]   = TensorDataset(embs, multi_hot)
        col_ids_map[split] = col_ids
        n_cols = int(col_ids.max().item()) + 1 if len(col_ids) else 0
        print(
            f"[CTA][finetune][{split}] {n_cols} columns  {len(embs)} samples "
            f"d={embs.shape[2]}  num_classes={num_classes}  embed_mode={embed_mode}"
        )

    # embed_dim = d_llm; head_dim = d_model (or d_llm if no encoder)
    embed_dim = splits["train"].tensors[0].shape[2]  # [N, 4, d_llm]
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
        embed_mode=embed_mode,
        smp_source=smp_source,
        intermediate_size=OmegaConf.select(cfg, "classifier.intermediate_size") or None,
        lr=cfg.finetuning.lr,
        weight_decay=cfg.finetuning.weight_decay,
        max_epochs=cfg.finetuning.epochs,
        threshold=threshold,
        encoder=encoder,
        encoder_config_dict=encoder_config_dict,
        freeze_encoder=freeze_encoder,
        pos_weight=OmegaConf.select(cfg, "classifier.pos_weight"),
        pos_weight_max=float(OmegaConf.select(cfg, "classifier.pos_weight_max") or 100.0),
    )
    enc_mode_str = (
        f"pretrained+{'frozen' if freeze_encoder else 'joint'}"
        if encoder is not None else "none"
    )
    print(
        f"[CTA][finetune] embed_dim={embed_dim}→head_dim={head_dim}  num_classes={num_classes}"
        f"  embed_mode={embed_mode}  smp_source={smp_source}"
        f"  encoder={enc_mode_str}  threshold={threshold}"
    )

    # ── Output directories: base / model_slug / timestamp_cfghash ───────────
    _model_slug = (
        cfg.embedder.model_name.split("/")[-1]
        .replace(" ", "_").replace(":", "#")
    )
    _run_ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:8]
    _run_suffix = Path(_model_slug) / f"{_run_ts}_{_cfg_hash}"

    # ── Callbacks ─────────────────────────────────────────────────────────────
    out_dir  = Path(cfg.finetuning.output_dir) / _run_suffix
    eval_dir = Path(cfg.eval.output_dir)        / _run_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        _saved_cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "_run": {
                "slug":    _model_slug,
                "ts":      _run_ts,
                "hash":    _cfg_hash,
                "out_dir": str(out_dir),
                "eval_dir": str(eval_dir),
            }
        }))
    OmegaConf.save(_saved_cfg, str(out_dir / "run_config.yaml"))
    print(f"[CTA][finetune] embedder slug  : {_model_slug}")
    print(f"[CTA][finetune] checkpoint dir : {out_dir}")
    print(f"[CTA][finetune] eval dir       : {eval_dir}")

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


    best_ckpt  = callbacks[0].best_model_path or str(out_dir / "last.ckpt")
    best_model = CTAClassifier.load_from_checkpoint(best_ckpt)
    best_model.eval().to(device)

    # Run test through the trainer so metrics appear in TensorBoard
    trainer.test(model=best_model, dataloaders=_loader("test", shuffle=False))

    # Defensive re-sync: trainer.test may alter module placement/lifecycle.
    best_model = best_model.to(device)
    best_model.eval()
    eval_device = next(best_model.parameters()).device

    metrics: dict = {}

    def _metric_dict(preds: torch.Tensor, target_int: torch.Tensor, thr: float) -> dict:
        """Micro/macro accuracy, precision, recall, F1 via torchmetrics.

        ``preds`` are binarised at ``thr`` — pass probabilities with the eval
        threshold, or hard 0/1 vote predictions with thr=0.5.
        """
        d = {"accuracy": float(
            MultilabelAccuracy(num_labels=num_classes, threshold=thr, average="micro")(preds, target_int)
        )}
        for avg in ("micro", "macro"):
            d[f"precision_{avg}"] = float(
                MultilabelPrecision(num_labels=num_classes, threshold=thr, average=avg)(preds, target_int))
            d[f"recall_{avg}"] = float(
                MultilabelRecall(num_labels=num_classes, threshold=thr, average=avg)(preds, target_int))
            d[f"f1_{avg}"] = float(
                MultilabelF1Score(num_labels=num_classes, threshold=thr, average=avg)(preds, target_int))
        return d

    for split in ("dev", "test"):
        embs, multi_hot = splits[split].tensors
        col_ids = col_ids_map[split]

        with torch.no_grad():
            logits = best_model(embs.to(eval_device)).cpu()

        split_metrics: dict = {}

        # Aggregate per-column via two voting schemes for every embed_mode.
        # In 'column' mode each column is already a single row (unique col_id),
        # so both votes reduce to plain thresholding; in 'cell'/'cell_header'
        # mode a column has one row per U-path and the votes actually aggregate.
        unique_cols = torch.unique(col_ids)
        col_target  = torch.stack(
            [multi_hot[col_ids == c][0] for c in unique_cols]
        ).long()

        # (1) Soft vote (current logic): mean per-cell logits → column logit,
        #     then sigmoid + threshold.
        col_logits = torch.stack([logits[col_ids == c].mean(0) for c in unique_cols])
        soft_probs = torch.sigmoid(col_logits)
        soft = _metric_dict(soft_probs, col_target, threshold)

        # (2) Hard majority (plurality) vote over the FULL prediction vector:
        #     threshold each cell first, then within each column count how many
        #     cells produced each distinct binary pattern and keep the most
        #     frequent whole pattern (e.g. [0,1,1,0] x2 vs [1,0,1,0] x1 → keep
        #     [0,1,1,0]).  Ties on count are broken by the larger number of
        #     predicted labels (max row sum); any remaining tie picks the first
        #     such pattern deterministically.
        cell_pred = (torch.sigmoid(logits) >= threshold).float()          # [n_cells, C] per-cell votes

        def _plurality(group: torch.Tensor) -> torch.Tensor:
            # group: [n_g, C] binary rows → winning [C] pattern
            uniq, _, counts = torch.unique(
                group, dim=0, return_inverse=True, return_counts=True
            )
            top  = counts.max()
            cand = counts == top                                          # tied top-count rows
            if int(cand.sum()) > 1:
                sums = uniq.sum(dim=1)
                sums = torch.where(cand, sums, sums.new_full(sums.shape, -1.0))
                win  = int(sums.argmax())                                 # break tie by max #labels
            else:
                win  = int(counts.argmax())
            return uniq[win]

        hard_pred = torch.stack(
            [_plurality(cell_pred[col_ids == c]) for c in unique_cols]
        )                                                                  # [n_cols, C]
        hard = _metric_dict(hard_pred, col_target, 0.5)

        # Unprefixed keys = soft vote (backward-compat + Optuna objective);
        # hard_* reports the majority-vote scheme alongside it.
        split_metrics.update(soft)
        split_metrics.update({f"hard_{k}": v for k, v in hard.items()})

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

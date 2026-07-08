"""
CTAClassifier.py
Column Type Annotation classifier � PyTorch Lightning module.

Architecture
------------
  Input  : pre-pooled column embedding  [B, embed_dim]
           (produced upstream by extract_column_embeddings or an embedder)
  Head   : LayerNorm ? Linear(embed_dim, num_classes)
  Loss   : BCEWithLogitsLoss (multi-label)
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

try:
    from ..config import CTAConfig
except ImportError:
    from config import CTAConfig


class CTAClassifier(pl.LightningModule):
    """
    Lightweight multi-label classifier over pre-pooled column embeddings.

    Parameters
    ----------
    config           : CTAConfig with at least `embedding_dim`, `num_classes`,
                       `layer_norm_eps`
    intermediate_size: read from config.classifier_intermediate_size;
                       if set, inserts a hidden layer
                       LayerNorm → Linear(embed_dim, intermediate_size) → GELU
                       → Dropout → Linear(intermediate_size, num_classes);
                       if None, uses a single Linear(embed_dim, num_classes).
    lr               : learning rate
    weight_decay     : AdamW weight decay
    max_epochs       : used for cosine LR schedule
    """

    def __init__(
        self,
        config: CTAConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        d = config.embedding_dim
        intermediate_size = getattr(config, "classifier_intermediate_size", None)
        if intermediate_size:
            self.head = nn.Sequential(
                nn.LayerNorm(d, eps=config.layer_norm_eps),
                nn.Linear(d, intermediate_size),
                nn.GELU(),
                nn.Dropout(getattr(config, "hidden_dropout_prob", 0.5)),
                nn.Linear(intermediate_size, config.num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d, eps=config.layer_norm_eps),
                nn.Linear(d, config.num_classes),
            )

        self.bce_loss = nn.BCEWithLogitsLoss()

    # -- Forward ---------------------------------------------------------------

    def forward(self, col_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        col_emb : [B, embed_dim]  pre-pooled column embedding

        Returns
        -------
        logits  : [B, num_classes]
        """
        return self.head(col_emb)

    # -- Lightning steps -------------------------------------------------------

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        col_emb = batch["col_emb"]   # [B, embed_dim]
        labels  = batch["labels"]    # [B, num_classes]  float multi-hot

        logits = self(col_emb)
        loss   = self.bce_loss(logits, labels)
        self.log(f"{split}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "test")

    # -- Optimiser -------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

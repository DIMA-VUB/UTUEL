"""
pretrain.py
Self-supervised pretraining for CTA using the TableEmbedJePA model.

The pretraining regime is IDENTICAL to TRL-model/train.py:
  - U-paths (SMP) are generated from each CTA table (reconstructed from
    the .table_col_type entity-link format)
  - TableEmbedJePA is trained with JEPA + InfoNCE losses
  - The pretrained encoder is saved and reused by finetune.py

The only difference from the TRL-model training is the data source:
  CTA .table_col_type files  (instead of WikiSQL JSONL)

Run from the UTUEL repo root:
    python CTA/pretrain.py

Override any config.yaml value on the CLI (Hydra):
    python CTA/pretrain.py pretraining.epochs=50 model.hidden_size=512
    python CTA/pretrain.py data.max_records=1000 smp.use_graph_walks=true

Sweep (sweep_optuna.yaml):
    python CTA/pretrain.py --config-name sweep_optuna --multirun

The pretrained checkpoint is saved to pretraining.output_dir and can
be loaded by finetune.py via finetuning.pretrained_ckpt.
"""

from __future__ import annotations

import datetime
import hashlib
import sys
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # CTA/
_ROOT = _HERE.parent                   # UTUEL/
_TRL  = _ROOT / "TRL-model"

# Insert TRL-model dir first so bare `import config` / `import smp` resolve there.
for p in (str(_TRL), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Register TRL-model/model as a proper package so relative imports inside it work.
import importlib.util as _ilu, types as _types

def _ensure_trl_model_pkg() -> None:
    """Register TRL-model/model in sys.modules under the 'model' package name."""
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

from config import TableEmbedJePAConfig  # TRL-model/config.py  (now first on sys.path)
from model  import TableEmbedJePA        # TRL-model/model/

# CTA-specific dataset / datamodule
try:
    from CTA.dataset import CTASMPDataModule
except ImportError:
    from dataset import CTASMPDataModule


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig, embed_dim_in: int) -> TableEmbedJePA:
    """
    Instantiate a TableEmbedJePA model from the CTA config.

    The model architecture and all loss weights are taken from config.yaml
    (model:, loss:) — identical to TRL-model/train.py::build_model.
    """
    hidden_size = cfg.model.hidden_size
    num_heads   = max(1, min(cfg.model.num_heads, hidden_size // 64))
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
        lr=cfg.pretraining.lr,
        weight_decay=cfg.pretraining.weight_decay,
        max_epochs=cfg.pretraining.epochs,
        ema_decay=cfg.model.ema_decay,
        jepa_weight=cfg.loss.jepa,
        jepa_bar_weight=cfg.loss.jepa_bar,
        local_weight=cfg.loss.local,
        global_weight=cfg.loss["global"],
        ablate_proj=ablate_proj,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """
    Pretrain TableEmbedJePA on CTA .table_col_type data.

    Returns the final val train_loss (negated for Optuna direction=maximize).
    """
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    pl.seed_everything(cfg.pretraining.seed, workers=True)

    print(OmegaConf.to_yaml(cfg))

    # ── Data ──────────────────────────────────────────────────────────────────
    t_setup = time.perf_counter()
    print("[CTA][pretrain] starting datamodule setup …", flush=True)
    dm = CTASMPDataModule(cfg)
    dm.setup("fit")
    print(
        f"[CTA][pretrain] datamodule setup finished in {time.perf_counter() - t_setup:.1f}s",
        flush=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # When ablate_proj=true the input_projection is Identity, so hidden_size MUST
    # match the embedder dim.  Sync it from the data (e.g. 768) so the saved
    # run_config is correct and finetune rebuilds the encoder with the right dim.
    if bool(OmegaConf.select(cfg, "model.ablate_proj", default=False)) \
            and cfg.model.hidden_size != dm.embed_dim:
        with open_dict(cfg):
            print(
                f"[CTA][pretrain] ablate_proj=true → overriding hidden_size "
                f"{cfg.model.hidden_size} → embed_dim {dm.embed_dim}"
            )
            cfg.model.hidden_size = dm.embed_dim
    model = build_model(cfg, embed_dim_in=dm.embed_dim)
    print(
        f"[CTA][pretrain] dataset={cfg.data.train_path}"
        f"  tables={len(dm.train_ds.records)}"
        f"  u-paths={len(dm.train_ds)}"
        f"  embed_dim={dm.embed_dim}"
        f"  hidden_size={cfg.model.hidden_size}"
    )

    # ── Output directory: base / model_slug / timestamp_cfghash ─────────────
    _model_slug = (
        cfg.embedder.model_name.split("/")[-1]
        .replace(" ", "_").replace(":", "#")
    )
    _run_ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:8]
    out_dir = Path(cfg.pretraining.output_dir) / _model_slug / f"{_run_ts}_{_cfg_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        _saved_cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "_run": {
                "slug":    _model_slug,
                "ts":      _run_ts,
                "hash":    _cfg_hash,
                "out_dir": str(out_dir),
            }
        }))
    OmegaConf.save(_saved_cfg, str(out_dir / "run_config.yaml"))
    print(f"[CTA][pretrain] embedder slug  : {_model_slug}")
    print(f"[CTA][pretrain] checkpoint dir : {out_dir}")

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename="cta-pretrain-{epoch:02d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_last=True,
            save_top_k=cfg.pretraining.save_total_limit,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if cfg.pretraining.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                patience=cfg.pretraining.early_stopping_patience,
                min_delta=cfg.pretraining.early_stopping_min_delta,
                mode="min",
            )
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.pretraining.epochs,
        precision=(
            "16-mixed" if cfg.pretraining.fp16
            else "bf16-mixed" if cfg.pretraining.bf16
            else "32-true"
        ),
        gradient_clip_val=cfg.pretraining.max_grad_norm,
        log_every_n_steps=cfg.pretraining.logging_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=str(out_dir),
    )

    print("[CTA][pretrain] starting trainer.fit …", flush=True)
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=cfg.pretraining.get("ckpt_path") or None,
    )

    train_loss = float(
        trainer.callback_metrics.get("train_loss", torch.tensor(float("inf")))
    )
    print(f"[CTA][pretrain] finished  train_loss={train_loss:.4f}")
    return -train_loss   # negated → Optuna direction=maximize


if __name__ == "__main__":
    main()

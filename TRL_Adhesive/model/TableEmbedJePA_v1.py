"""
table_jepa.py
TableJEPA — Table Embedding JEPA (Joint-Embedding Predictive Architecture).

Architecture:
  Input  : pre-computed LLM cell embeddings  [B, seq_len, hidden_size]
  Encoder: stack of transformer blocks with a semantic-matrix attention mask
  Losses :
    1. Cell-level InfoNCE    — aligns cell embedding with its column/row headers
    2. U-path-level InfoNCE  — separates U-path embeddings across table contexts
    3. Query-level cosine    — aligns SMP-induced queries with answer-cell anchors
                               (activated after epoch `when_to_include_kl_loss`)
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig
from transformers.activations import GELUActivation


# ── Model output dataclass ────────────────────────────────────────────────────

@dataclass
class TableEmbedJePAOutput:
    """Return type for TableEmbedJePA.forward()."""
    loss:          torch.Tensor
    logits:        torch.Tensor
    attentions:    Optional[torch.Tensor] = None
    query_logits:  Optional[torch.Tensor] = None   # populated during query inference
    jepa_loss:     Optional[torch.Tensor] = None   # JEPA prediction loss (SMP)
    jepa_bar_loss: Optional[torch.Tensor] = None   # JEPA prediction loss (SMP_bar)
    local_loss:    Optional[torch.Tensor] = None   # local InfoNCE loss
    global_loss:   Optional[torch.Tensor] = None   # global SMP-level InfoNCE loss


# ── Activation registry ───────────────────────────────────────────────────────

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier({"gelu": GELUActivation})


# ── Auxiliary loss functions ──────────────────────────────────────────────────

def isotropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Encourage the covariance of embeddings to be close to identity."""
    losses = []
    for i in range(x.shape[0]):
        cov = torch.cov(x[i].T)
        identity = torch.eye(cov.shape[0], device=x.device)
        losses.append(torch.norm(cov - identity))
    return torch.stack(losses).mean()


def angular_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    anchor   = F.normalize(anchor,   p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    pos_sim  = (anchor * positive).sum(dim=1)
    neg_sim  = (anchor * negative).sum(dim=1)
    return F.relu(margin + neg_sim - pos_sim).mean()


# ── Sub-modules ───────────────────────────────────────────────────────────────

class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if encoder_hidden_states is not None:
            key_layer   = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer   = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        scores = torch.clamp(scores, max=1e4, min=-1e4)

        if attention_mask is not None:
            # Additive key-padding mask, broadcast over heads + query positions:
            # shape [B, 1, 1, L] with 0 for valid keys and a large negative for pad.
            scores = scores + attention_mask

        scores = torch.clamp(scores, max=1e5)
        probs  = nn.functional.softmax(scores, dim=-1)
        probs  = self.dropout(probs)

        context = torch.matmul(probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        return (context, scores) if output_attentions else (context,)


class RobertaSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.dense     = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        query: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        # Residual + LayerNorm (post-norm)
        return input_tensor + self.LayerNorm(hidden_states)


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig, position_embedding_type=None):
        super().__init__()
        self.self   = RobertaSelfAttention(config, position_embedding_type)
        self.output = RobertaSelfOutput(config.hidden_size, config.layer_norm_eps,
                                        config.hidden_dropout_prob)
        self.pruned_heads: set = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions: bool = False,
        query: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_out = self.self(
            hidden_states, attention_mask, head_mask,
            encoder_hidden_states, encoder_attention_mask,
            past_key_value, output_attentions,
        )
        attn_output = self.output(self_out[0], hidden_states, query=query)
        return (attn_output,) + self_out[1:]


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act   = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) \
                     else config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x))


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense     = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        query: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return input_tensor + self.LayerNorm(hidden_states)


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention   = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output       = RobertaOutput(config)
        self.is_decoder   = config.is_decoder
        self._query_flag  = False  # propagated from Encoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions: bool = False,
        query: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self._query_flag = query
        self_attn_past_kv = past_key_value[:2] if past_key_value is not None else None
        attn_outputs = self.attention(
            hidden_states, attention_mask, head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_kv,
            query=query,
        )
        attn_output = attn_outputs[0]
        outputs     = attn_outputs[1:]

        layer_output = self._ffn_chunk(attn_output)
        return (layer_output,) + outputs

    def _ffn_chunk(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.output(self.intermediate(attn_output), attn_output,
                           query=self._query_flag)


class Encoder(nn.Module):
    """Stack of N RobertaLayer blocks."""

    def __init__(self, num_layers: int, config: RobertaConfig, with_residual_norm: bool = True):
        super().__init__()
        layer = RobertaLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, list, list]:
        output = src
        hidden_layers, attention_outputs = [], []
        for layer in self.layers:
            out, attn = layer(output, attention_mask=mask, output_attentions=True, query=query)
            output = out
            hidden_layers.append(output)
            attention_outputs.append(attn.unsqueeze(0) if attn.dim() != 4 else attn)
        return output, hidden_layers, attention_outputs


# ── Predictor ────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Maps the online encoder's node_a hidden state to the target encoder's
    CLS embedding space.

    Input : h_node_a  [B, d] — node_a output of the online encoder
    Output: predicted [B, d] — predicted representation in target space
    """

    def __init__(self, hidden_size: int, pred_hidden_mult: int = 4):
        super().__init__()
        h = hidden_size * pred_hidden_mult
        self.net = nn.Sequential(
            nn.Linear(hidden_size, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class NonSquareIdentity(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x[..., :self.out_features]

# ── TableEmbedJePA ───────────────────────────────────────────────────────────

class TableEmbedJePA(pl.LightningModule):
    """
    Table Embedding JEPA (Joint-Embedding Predictive Architecture).

    Self-supervised table representation model that refines pre-computed LLM
    node embeddings via a transformer encoder trained with three objectives:

      1. JEPA prediction loss — predictor(online_enc(SMP)[node]) ≈
                                 target_enc(query)[CLS]  (cosine distance)
      2. Local InfoNCE        — the node aligned with its SMP header/pivot slots
      3. Global SMP InfoNCE   — CLS[SMP[i]] positive with CLS[SMP_bar[i]]

    The target encoder is an EMA of the online encoder (updated each batch).

    Input sequence layout (variable length, role-masked — set at runtime):
      SMP      : [CLS, a_h1, …, node, …, b_h1]   — length L+1, node slot varies
      SMP_bar  : [CLS, b_h1, …, node, …, a_h1]   — reversed walk
      Query    : [concat(pivot_a, node, pivot_b)] — length 1, no CLS

    Because tables nest headers, the node's position is not fixed; the model
    locates it via the per-slot role mask (``smp_role`` — 1 at the node, 0 at
    headers) and ignores padded slots via ``smp_pad_mask``.

    Usage::

        model = TableEmbedJePA(config, lr=1e-4, max_epochs=20)
        trainer = pl.Trainer(max_epochs=20)
        trainer.fit(model, datamodule=dm)

    Inference::

        out = model(query_inference=q_seq)   # [N, 3, d] sequences
        cls_embeds = out.query_logits        # [N, d] — use for kNN retrieval
    """

    def __init__(
        self,
        config: RobertaConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 20,
        ema_decay: float = 0.996,
        jepa_weight: float = 1.0,
        jepa_bar_weight: float = 1.0,
        local_weight: float = 1.0,
        global_weight: float = 1.0,
        ablate_proj: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.ema_decay = ema_decay

        # Loss weights — 0.0 disables that term from total_loss
        self.jepa_weight     = jepa_weight
        self.jepa_bar_weight = jepa_bar_weight
        self.local_weight    = local_weight
        self.global_weight   = global_weight

        self.tempeture = config.tempeture

        # ── Architecture ───────────────────────────────────────────────────────
        # Input projection: LLM embed_dim_in → transformer hidden_size (embed_dim_out).
        # When ablate_proj=True, replaced with Identity; the transformer then
        # operates at embedding_dim so we patch config.hidden_size accordingly.
        if ablate_proj:
            # self.input_projection = nn.Identity()
            self.input_projection = NonSquareIdentity(config.embedding_dim, config.hidden_size)
            config = copy.copy(config)           # don't mutate the caller's object
            # config.hidden_size = config.embedding_dim
        else:
            self.input_projection = nn.Linear(config.embedding_dim, config.hidden_size)

        # CLS token: same dimension as projection input
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Online encoder: updated by gradient descent
        self.transformer_encoder = Encoder(config.num_hidden_layers, config=config)

        # Target encoder: EMA of online encoder, not updated by gradients
        self.target_encoder = copy.deepcopy(self.transformer_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Predictor: maps node_a hidden state to target encoder's CLS space
        self.predictor = Predictor(config.hidden_size)

        self.save_hyperparameters(ignore=["config"])

    # ── EMA update ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """Update target encoder as an EMA of the online encoder."""
        decay = self.ema_decay
        for p_on, p_tgt in zip(self.transformer_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.lerp_(p_on.data, 1.0 - decay)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._update_target_encoder()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        smp_embeds: Optional[torch.Tensor] = None,
        smp_bar_embeds: Optional[torch.Tensor] = None,
        smp_role: Optional[torch.Tensor] = None,
        smp_bar_role: Optional[torch.Tensor] = None,
        smp_pad_mask: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None,
        query_bar_embeds: Optional[torch.Tensor] = None,
        query_inference: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TableEmbedJePAOutput:
        """
        Forward pass.

        Training mode (smp_embeds, smp_bar_embeds, query_embeds all provided):
          Computes three losses:
            1. JEPA prediction loss — predictor(h_node) ≈ target_enc(query)[CLS]
            2. Local InfoNCE        — node ↔ its header/pivot slots
            3. Global SMP InfoNCE   — CLS[SMP[i]] ↔ CLS[SMP_bar[i]] positive pairs

        Inference mode (query_inference provided):
          Embeds query sequence through the target encoder; returns mean-pooled
          output in query_logits for nearest-neighbour retrieval.

        Args:
            smp_embeds       [B, L, d_in]: ordered walk [a_h…, node, …b_h] (padded)
            smp_bar_embeds   [B, L, d_in]: reversed walk (padded)
            smp_role         [B, L]      : 1 at the node slot, 0 at headers/pad
            smp_bar_role     [B, L]      : role mask aligned with the reversed walk
            smp_pad_mask     [B, L]      : 1 for valid slots, 0 for padding
            query_embeds     [B, 1, d_in]: LLM embed of concat(pivot_a, node, pivot_b)
            query_bar_embeds [B, 1, d_in]: symmetric target for SMP_bar
            query_inference  [N, 1, d_in]: query sequences for inference mode
        """
        ref = smp_embeds if smp_embeds is not None else query_inference
        device = ref.device

        # ── Inference mode ────────────────────────────────────────────────────
        if query_inference is not None:
            N = query_inference.shape[0]
            q_proj = self.input_projection(query_inference)            # [N, 1, d_out]
            # Single LLM embedding per query — pass through target encoder, squeeze
            with torch.no_grad():
                z_q, _, _ = self.target_encoder(q_proj)               # [N, 1, d_out]
            q_pool = F.normalize(z_q[:, 0, :], dim=-1)               # [N, d_out]
            dummy  = torch.zeros(1, device=device)
            return TableEmbedJePAOutput(loss=dummy, logits=dummy, query_logits=q_pool)

        # ── Training mode ─────────────────────────────────────────────────────
        B, L, _ = smp_embeds.shape
        NEG = -1e9   # additive attention-mask fill for padded key positions

        if smp_pad_mask is None:
            smp_pad_mask = torch.ones(B, L, device=device)
        if smp_role is None:
            smp_role = torch.zeros(B, L, device=device)
        if smp_bar_role is None:
            smp_bar_role = smp_role

        # Project raw LLM embeddings from embed_dim_in → hidden_size (embed_dim_out)
        smp_proj = self.input_projection(smp_embeds)                  # [B, L, d_out]
        bar_proj = self.input_projection(smp_bar_embeds)              # [B, L, d_out]
        qry_proj = self.input_projection(query_embeds)                # [B, 1, d_out]
        qry_bar_proj = self.input_projection(query_bar_embeds)        # [B, 1, d_out]
        cls_proj = self.input_projection(self.cls_token)              # [1, 1, d_out]

        cls = cls_proj.expand(B, -1, -1)                             # [B, 1, d_out]

        # Prepend CLS to SMP sequences (query is a single concatenated LLM embedding)
        smp_in     = torch.cat([cls, smp_proj], dim=1)               # [B, L+1, d_out]
        smp_bar_in = torch.cat([cls, bar_proj], dim=1)               # [B, L+1, d_out]
        qry_in     = qry_proj                                        # [B, 1, d_out]

        # Full-length masks that include the always-valid CLS slot at index 0.
        cls_ones  = torch.ones(B, 1, device=device)
        cls_zeros = torch.zeros(B, 1, device=device)
        pad_full      = torch.cat([cls_ones,  smp_pad_mask], dim=1)   # [B, L+1] 1=valid
        role_full     = torch.cat([cls_zeros, smp_role],     dim=1)   # [B, L+1] 1=node
        role_bar_full = torch.cat([cls_zeros, smp_bar_role], dim=1)   # [B, L+1]

        # Additive key-padding attention mask [2B, 1, 1, L+1] (0 valid / NEG pad).
        attn_add = (1.0 - pad_full)[:, None, None, :] * NEG          # [B, 1, 1, L+1]
        attn_add2 = torch.cat([attn_add, attn_add], dim=0)          # [2B, 1, 1, L+1]

        # Online encoder: process SMP + SMP_bar in one fused batch
        all_smp = torch.cat([smp_in, smp_bar_in], dim=0)             # [2B, L+1, d]
        enc_out, _, attn_outs = self.transformer_encoder(all_smp, mask=attn_add2)
        enc_smp = enc_out[:B]                                        # [B, L+1, d]
        enc_bar = enc_out[B:]                                        # [B, L+1, d]

        # Target encoder: process query and query_bar (no grad; EMA weights)
        qry_bar_in = qry_bar_proj                                    # [B, 1, d_out]
        with torch.no_grad():
            z_qry,     _, _ = self.target_encoder(qry_in)           # [B, 1, d]
            z_qry_bar, _, _ = self.target_encoder(qry_bar_in)       # [B, 1, d]
        z_target     = F.normalize(z_qry[:, 0, :],     dim=-1)      # [B, d]
        z_target_bar = F.normalize(z_qry_bar[:, 0, :], dim=-1)      # [B, d]

        zero = torch.tensor(0.0, device=device)
        total_loss = zero.clone()

        def _node_vec(h: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
            """Masked mean over the node slot(s) — exactly one node per U-path."""
            r = role.unsqueeze(-1)                                   # [B, L+1, 1]
            return (h * r).sum(1) / r.sum(1).clamp_min(1e-6)         # [B, d]

        # ── Loss 1: JEPA prediction loss (SMP) ────────────────────────────────
        # predictor maps the online encoder's node hidden state → target CLS space
        h_node = _node_vec(enc_smp, role_full)                       # [B, d]
        p_node = F.normalize(self.predictor(h_node), dim=-1)         # [B, d]
        jepa_loss = (2 - 2 * (p_node * z_target).sum(-1)).mean()
        if self.jepa_weight > 0:
            total_loss = total_loss + self.jepa_weight * jepa_loss

        # ── Loss 1b: JEPA prediction loss (SMP_bar) ───────────────────────────
        h_node_bar = _node_vec(enc_bar, role_bar_full)               # [B, d]
        p_node_bar = F.normalize(self.predictor(h_node_bar), dim=-1) # [B, d]
        jepa_bar_loss = (2 - 2 * (p_node_bar * z_target_bar).sum(-1)).mean()
        if self.jepa_bar_weight > 0:
            total_loss = total_loss + self.jepa_bar_weight * jepa_bar_loss

        # ── Loss 2: Local alignment (node ↔ its header/pivot slots) ────────────
        # anchor    : the node slot (via role mask)
        # positives : every valid header/pivot slot (pad_full · ¬role, excl. CLS)
        # Loss = mean cosine distance of the node from its header slots.
        h_norm  = F.normalize(enc_smp, dim=-1)                       # [B, L+1, d]
        h_na    = F.normalize(_node_vec(enc_smp, role_full), dim=-1) # [B, d]
        hdr_mask = pad_full * (1.0 - role_full)                      # [B, L+1] valid, non-node
        hdr_mask = hdr_mask.clone()
        hdr_mask[:, 0] = 0.0                                         # exclude CLS
        sim = (h_na.unsqueeze(1) * h_norm).sum(-1)                   # [B, L+1] cos(node, slot)
        local_loss = ((1 - sim) * hdr_mask).sum() / hdr_mask.sum().clamp_min(1e-6)
        if self.local_weight > 0:
            total_loss = total_loss + self.local_weight * local_loss

        # ── Loss 3: Global SMP-level InfoNCE ─────────────────────────────────
        # CLS[SMP[i]] is positive with CLS[SMP_bar[i]] (same U-path, reversed)
        cls_smp = F.normalize(enc_smp[:, 0, :], dim=-1)             # [B, d]
        cls_bar = F.normalize(enc_bar[:, 0, :], dim=-1)             # [B, d]
        all_cls = torch.cat([cls_smp, cls_bar], dim=0)             # [2B, d]
        sim_mat = (all_cls @ all_cls.T) / self.tempeture           # [2B, 2B]
        sim_mat = sim_mat.masked_fill(
            torch.eye(2 * B, dtype=torch.bool, device=device), float("-inf"))
        labels_g = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])
        global_loss = F.cross_entropy(sim_mat, labels_g)
        if self.global_weight > 0:
            total_loss = total_loss + self.global_weight * global_loss

        return TableEmbedJePAOutput(
            loss=total_loss,
            logits=enc_smp,
            attentions=attn_outs[-1] if attn_outs else None,
            jepa_loss=jepa_loss,
            jepa_bar_loss=jepa_bar_loss,
            local_loss=local_loss,
            global_loss=global_loss,
        )

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> Optional[torch.Tensor]:
        out = self(
            smp_embeds=batch["smp_embeds"],
            smp_bar_embeds=batch["smp_bar_embeds"],
            smp_role=batch.get("smp_role"),
            smp_bar_role=batch.get("smp_bar_role"),
            smp_pad_mask=batch.get("smp_pad_mask"),
            query_embeds=batch["query_embeds"],
            query_bar_embeds=batch["query_bar_embeds"],
        )

        # Guard against NaN/Inf loss (can occur on degenerate per-table batches
        # where all U-paths share identical pivot embeddings and the InfoNCE
        # gradient is amplified by 1/temperature before clipping takes effect).
        if not torch.isfinite(out.loss):
            print(f"\n[train] WARNING: loss={out.loss.item():.4g} at batch {batch_idx} "
                  f"(jepa={out.jepa_loss.item():.4g}, "
                  f"jepa_bar={out.jepa_bar_loss.item():.4g}, "
                  f"local={out.local_loss.item():.4g}, "
                  f"global={out.global_loss.item():.4g}) — step skipped")
            return None   # Lightning 2.0: skip without updating weights

        B  = batch["smp_embeds"].shape[0]
        kw = dict(on_step=True, on_epoch=True, batch_size=B)
        self.log("train_loss",  out.loss,        prog_bar=True,  **kw)
        # Raw (unweighted) individual losses — always logged for monitoring
        self.log("jepa_loss",     out.jepa_loss,     prog_bar=False, **kw)
        self.log("jepa_bar_loss", out.jepa_bar_loss, prog_bar=False, **kw)
        self.log("local_loss",    out.local_loss,    prog_bar=False, **kw)
        self.log("global_loss",   out.global_loss,   prog_bar=False, **kw)
        return out.loss

    # ── optimizers ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

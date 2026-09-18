"""Learned token embedding front end for TableEmbedJePA."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
import warnings


class ScratchEmbedder(nn.Module):
    """Embed padded cell token ids using masked mean or CLS cross-attention pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_id: int,
        pretrained_model_name: str | None = None,
        pooling: str = "mean",
        cls_token_id: int | None = None,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls_attention"}:
            raise ValueError("pooling must be 'mean' or 'cls_attention'.")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pooling = pooling
        self.cls_token_id = cls_token_id
        self._warned_missing_cls = False
        if pooling == "cls_attention":
            num_heads = max(1, min(8, embed_dim // 64))
            while embed_dim % num_heads != 0:
                num_heads -= 1
            self.cls_attention = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True,
            )
        if pretrained_model_name:
            pretrained_embedding = AutoModel.from_pretrained(
                pretrained_model_name).get_input_embeddings().weight
            if tuple(pretrained_embedding.shape) != (vocab_size, embed_dim):
                raise ValueError(
                    "Pretrained embedding shape must match tokenizer vocabulary and embed_dim: "
                    f"expected {(vocab_size, embed_dim)}, got {tuple(pretrained_embedding.shape)}."
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        has_cls = (
            self.cls_token_id is not None
            and bool(input_ids[..., 0].eq(self.cls_token_id).all())
        )
        if self.pooling == "cls_attention" and has_cls:
            # Flatten leading dimensions so every SMP element self-attends over
            # its own token sequence, then retain the tokenizer's first CLS token.
            leading_shape = token_embeddings.shape[:-2]
            sequence_length = token_embeddings.shape[-2]
            flat_tokens = token_embeddings.reshape(-1, sequence_length, token_embeddings.shape[-1])
            flat_mask = attention_mask.reshape(-1, sequence_length)
            attended, _ = self.cls_attention(
                flat_tokens,
                flat_tokens,
                flat_tokens,
                key_padding_mask=flat_mask.eq(0),
                need_weights=False,
            )
            return attended[:, 0, :].reshape(*leading_shape, token_embeddings.shape[-1])
        if self.pooling == "cls_attention" and not self._warned_missing_cls:
            warnings.warn(
                "scratch_pooling='cls_attention' requires the tokenizer to place its CLS token "
                "at position 0. Falling back to masked mean pooling.",
                stacklevel=2,
            )
            self._warned_missing_cls = True
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        token_counts = mask.sum(dim=-2).clamp_min(1.0)
        return (token_embeddings * mask).sum(dim=-2) / token_counts
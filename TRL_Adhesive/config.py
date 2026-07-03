"""
config.py
TableEmbedJePAConfig — extends RobertaConfig with UTUEL-specific hyperparameters.
"""

from transformers import RobertaConfig


class TableEmbedJePAConfig(RobertaConfig):
    """
    Configuration for the TableEmbedJePA table representation learning model.

    All standard RobertaConfig fields are inherited (hidden_size,
    num_hidden_layers, num_attention_heads, …).  UTUEL-specific fields:

        tempeture           – softmax temperature for contrastive losses
        beta                – reserved weighting scalar
        embedding_dim       – dimensionality of the raw LLM input embeddings
                              (embed_dim_in).  A learned linear projection maps
                              this to hidden_size (embed_dim_out) inside the model,
                              so the two may differ.
    """

    model_type = "table_embed_jepa"

    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 1,
        is_decoder: bool = False,
        add_cross_attention: bool = False,
        chunk_size_feed_forward: int = 0,
        # ── UTUEL-specific ──────────────────────────────────────────────────
        tempeture: float = 0.07,
        beta: float = 0.5,
        embedding_dim: int = 768,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            is_decoder=is_decoder,
            add_cross_attention=add_cross_attention,
            chunk_size_feed_forward=chunk_size_feed_forward,
            **kwargs,
        )
        self.tempeture = tempeture
        self.beta = beta
        self.embedding_dim = embedding_dim

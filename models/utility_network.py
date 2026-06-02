"""
Memory Utility Network (MUN)
============================
The core neural architecture for predicting memory utility scores.

A memory item's utility u(m, c, t) is a scalar in [0,1] that estimates how
likely memory m will be needed given current context c at future time t.

Architecture:
  - Memory encoder: transformer with sinusoidal + temporal-decay positional encoding
  - Context encoder: transformer with CLS pooling
  - Cross-attention fusion: memories attend over context
  - Multi-layer utility prediction head (MLP)
  - Auxiliary contrastive projection head
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional / Temporal Encoding ────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: cos needs same number of terms as sin
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalDecayEncoding(nn.Module):
    """
    Learned temporal decay encoding.

    Encodes relative age of a memory as a continuous feature vector, allowing
    the model to learn how utility decays with time in a task-dependent way.
    """

    def __init__(self, d_model: int, num_timescales: int = 16):
        super().__init__()
        self.num_timescales = num_timescales
        self.d_model = d_model
        # Learned log-timescales initialised at geometric spacing
        self.log_timescales = nn.Parameter(
            torch.linspace(0.0, math.log(1e4), num_timescales)
        )
        self.proj = nn.Linear(num_timescales * 2, d_model)

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ages: (B, N) float tensor of memory ages (normalised to [0,1])
        Returns:
            (B, N, d_model) temporal encodings
        """
        timescales = torch.exp(self.log_timescales)                          # (T,)
        scaled = ages.unsqueeze(-1) / (timescales.unsqueeze(0).unsqueeze(0) + 1e-8)  # (B, N, T)
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)     # (B, N, 2T)
        return self.proj(enc)                                                # (B, N, d_model)


# ── Encoder Blocks ─────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Single pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + x


class MemoryEncoder(nn.Module):
    """
    Encodes a batch of memory items with temporal awareness.

    Input:  pre-embedded memory vectors (from sentence-transformers or similar).
    Output: contextualised memory representations (B, N, d_model).
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.temporal_enc = TemporalDecayEncoding(d_model)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, d_model * 4, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        embeddings: torch.Tensor,                      # (B, N, input_dim)
        ages: Optional[torch.Tensor] = None,           # (B, N) normalised ages
        padding_mask: Optional[torch.Tensor] = None,   # (B, N) True=pad
    ) -> torch.Tensor:
        """Returns (B, N, d_model)."""
        x = self.input_proj(embeddings)                # (B, N, d_model)
        x = self.pos_enc(x)
        if ages is not None:
            x = x + self.temporal_enc(ages)
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return self.norm(x)


class ContextEncoder(nn.Module):
    """
    Encodes agent context history (recent observations/steps) into a fixed-size
    context vector and a full sequence for cross-attention.
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, d_model * 4, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        # Learnable CLS token for global context pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.d_model = d_model

    def forward(
        self,
        embeddings: torch.Tensor,                      # (B, L, input_dim)
        padding_mask: Optional[torch.Tensor] = None,   # (B, L) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context_vec: (B, d_model) — CLS-pooled global context
            context_seq: (B, L, d_model) — per-token sequence (without CLS)
        """
        B = embeddings.size(0)
        x = self.input_proj(embeddings)                # (B, L, d_model)
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                 # (B, L+1, d_model)

        # Extend padding mask for CLS (CLS is never padding)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask_ext = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)
        else:
            padding_mask_ext = None

        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask_ext)

        x = self.norm(x)
        context_vec = x[:, 0]     # CLS token → (B, d_model)
        context_seq = x[:, 1:]    # token sequence → (B, L, d_model)
        return context_vec, context_seq


# ── Cross-Attention Fusion ──────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses memory representations with context via cross-attention.

    Each memory position attends over the full context sequence to obtain
    a relevance-weighted representation for downstream utility prediction.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        memory_seq: torch.Tensor,                       # (B, N, d_model)
        context_seq: torch.Tensor,                      # (B, L, d_model)
        memory_mask: Optional[torch.Tensor] = None,     # (B, N) True=pad
        context_mask: Optional[torch.Tensor] = None,   # (B, L) True=pad
    ) -> torch.Tensor:
        """Returns (B, N, d_model) fused memory representations."""
        # Pre-norm cross-attention: memories query context
        residual = memory_seq
        x = self.norm1(memory_seq)
        x, _ = self.cross_attn(
            query=x,
            key=context_seq,
            value=context_seq,
            key_padding_mask=context_mask,
        )
        x = residual + x
        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + x


# ── Utility Prediction Head ────────────────────────────────────────────────────

class UtilityHead(nn.Module):
    """
    MLP that predicts a utility score in (0, 1) for each memory.

    Input: concatenation of fused memory repr and global context vector.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        # Learned temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

        # Build MLP: input is [fused_memory ‖ context_vec] → d_model * 2
        dims = [d_model * 2] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        fused_memory: torch.Tensor,   # (B, N, d_model)
        context_vec: torch.Tensor,    # (B, d_model)
    ) -> torch.Tensor:
        """Returns (B, N) utility scores in (0, 1)."""
        B, N, D = fused_memory.shape
        ctx_expanded = context_vec.unsqueeze(1).expand(-1, N, -1)    # (B, N, d_model)
        x = torch.cat([fused_memory, ctx_expanded], dim=-1)           # (B, N, 2*d_model)
        logits = self.mlp(x).squeeze(-1)                              # (B, N)
        temp = self.temperature.clamp(min=0.01)
        return torch.sigmoid(logits / temp)


# ── Contrastive Projection Head ─────────────────────────────────────────────────

class ContrastiveHead(nn.Module):
    """Projects memory representations into a unit-normalised contrastive space."""

    def __init__(self, d_model: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model) → (B, N, proj_dim), L2-normalised
        return F.normalize(self.proj(x), dim=-1)


# ── Full Memory Utility Network ─────────────────────────────────────────────────

class MemoryUtilityNetwork(nn.Module):
    """
    Memory Utility Network (MUN).

    Forward pass:
      1. Encode memory items with sinusoidal + temporal-decay positional encoding
      2. Encode agent context history; extract CLS vector and token sequence
      3. Fuse via cross-attention (memories attend over context)
      4. Predict per-memory utility scores via MLP head
      5. Return contrastive projections for auxiliary InfoNCE loss

    All padded positions are masked throughout; their output scores are zeroed.

    Usage::

        model = MemoryUtilityNetwork(input_dim=384)
        out = model(memory_embs, context_embs, memory_ages)
        scores = out["utility_scores"]   # (B, N) ∈ (0, 1)
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 512,
        nhead: int = 8,
        num_memory_layers: int = 3,
        num_context_layers: int = 2,
        utility_hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
        contrastive_proj_dim: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        self.memory_encoder = MemoryEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_memory_layers,
            dropout=dropout,
        )
        self.context_encoder = ContextEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_context_layers,   # FIX: was always using num_memory_layers
            dropout=dropout,
        )
        self.fusion = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout)
        self.utility_head = UtilityHead(
            d_model=d_model,
            hidden_dims=utility_hidden_dims,
            dropout=dropout,
            temperature=temperature,
        )
        self.contrastive_head = ContrastiveHead(d_model, contrastive_proj_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        memory_embeddings: torch.Tensor,                        # (B, N, input_dim)
        context_embeddings: torch.Tensor,                       # (B, L, input_dim)
        memory_ages: Optional[torch.Tensor] = None,             # (B, N)
        memory_padding_mask: Optional[torch.Tensor] = None,     # (B, N) True=pad
        context_padding_mask: Optional[torch.Tensor] = None,    # (B, L) True=pad
        return_representations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            memory_embeddings:    (B, N, input_dim) — N candidate memory items
            context_embeddings:   (B, L, input_dim) — L context steps
            memory_ages:          (B, N) — normalised ages ∈ [0, 1]
            memory_padding_mask:  (B, N) — True for padded slots
            context_padding_mask: (B, L) — True for padded context steps
            return_representations: if True, include intermediate tensors in output

        Returns dict with keys:
            ``utility_scores``   (B, N) ∈ (0, 1) — higher = more useful to retain
            ``contrastive_proj`` (B, N, proj_dim) — for InfoNCE auxiliary loss
            ``memory_repr``      (B, N, d_model)  — if return_representations
            ``fused_repr``       (B, N, d_model)  — if return_representations
            ``context_vec``      (B, d_model)      — if return_representations
            ``context_seq``      (B, L, d_model)   — if return_representations
        """
        # 1. Encode memories
        memory_repr = self.memory_encoder(
            memory_embeddings,
            ages=memory_ages,
            padding_mask=memory_padding_mask,
        )  # (B, N, d_model)

        # 2. Encode context
        context_vec, context_seq = self.context_encoder(
            context_embeddings,
            padding_mask=context_padding_mask,
        )  # (B, d_model), (B, L, d_model)

        # 3. Cross-attention fusion
        fused = self.fusion(
            memory_seq=memory_repr,
            context_seq=context_seq,
            memory_mask=memory_padding_mask,
            context_mask=context_padding_mask,
        )  # (B, N, d_model)

        # 4. Utility scores
        utility_scores = self.utility_head(fused, context_vec)  # (B, N) ∈ (0,1)

        # Zero out padded memory positions
        if memory_padding_mask is not None:
            utility_scores = utility_scores.masked_fill(memory_padding_mask, 0.0)

        # 5. Contrastive projections
        contrastive_proj = self.contrastive_head(fused)  # (B, N, proj_dim)

        output: Dict[str, torch.Tensor] = {
            "utility_scores": utility_scores,
            "contrastive_proj": contrastive_proj,
        }
        if return_representations:
            output["memory_repr"] = memory_repr
            output["fused_repr"] = fused
            output["context_vec"] = context_vec
            output["context_seq"] = context_seq

        return output

    @torch.no_grad()
    def predict_utility(
        self,
        memory_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        memory_ages: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference-only utility prediction. Returns (B, N) scores."""
        self.eval()
        out = self.forward(
            memory_embeddings,
            context_embeddings,
            memory_ages=memory_ages,
            memory_padding_mask=memory_padding_mask,
            context_padding_mask=context_padding_mask,
        )
        return out["utility_scores"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_params = self.num_parameters() / 1e6
        return (
            f"MemoryUtilityNetwork(\n"
            f"  input_dim={self.input_dim},\n"
            f"  d_model={self.d_model},\n"
            f"  params={n_params:.2f}M\n"
            f")"
        )


# ── Model Factory ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> MemoryUtilityNetwork:
    """Construct a MemoryUtilityNetwork from a config dict."""
    m = config.get("model", config)
    return MemoryUtilityNetwork(
        input_dim=m.get("embedding_dim", 384),
        d_model=m.get("hidden_dim", 512),
        nhead=m.get("num_attention_heads", 8),
        num_memory_layers=m.get("num_transformer_layers", 3),
        num_context_layers=m.get("num_context_layers", 2),
        utility_hidden_dims=tuple(m.get("utility_hidden_dims", [256, 128, 64])),
        dropout=m.get("dropout", 0.1),
        contrastive_proj_dim=m.get("contrastive_proj_dim", 128),
        temperature=m.get("utility_head", {}).get("temperature", 1.0),
    )

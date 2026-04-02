"""
Tiny Transformer for modular arithmetic.

Architecture:
  - Token embedding (p tokens → d_model dimensions)
  - Positional embedding (2 positions for the two-token input [a, b])
  - 1 or 2 Transformer encoder layers
  - Linear classification head → p classes

Total parameters kept under 150k so training runs on CPU in minutes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with multi-head self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # Layer norms (pre-norm architecture for stability)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Storage for attention weights (populated by hooks)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, S, D = x.shape

        # Pre-norm self-attention
        normed = self.ln1(x)
        Q = self.W_q(normed).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(normed).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(normed).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()  # Cache for visualization

        attn_out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.dropout(self.W_o(attn_out))

        # Pre-norm FFN
        normed = self.ln2(x)
        x = x + self.dropout(self.ffn(normed))

        return x


class GrokTransformer(nn.Module):
    """
    Minimal Transformer for (a + b) mod p classification.

    The model:
      1. Embeds two input tokens a and b.
      2. Adds learned positional embeddings for positions 0 and 1.
      3. Passes through n_layers TransformerBlocks.
      4. Reads the output at position 0 (like a [CLS] token).
      5. Projects to p classes via a linear head.
    """

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 200,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding layers
        self.token_embedding = nn.Embedding(p, d_model)
        self.position_embedding = nn.Embedding(2, d_model)  # Only 2 positions

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm + classification head
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) — token indices for [a, b]
        Returns:
            (batch, p) — logits over residue classes
        """
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)

        # Embed tokens + positions
        h = self.token_embedding(x) + self.position_embedding(positions)

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h)

        # Read from position 0 and classify
        h = self.ln_final(h[:, 0, :])  # (batch, d_model)
        logits = self.head(h)          # (batch, p)

        return logits

    def get_token_embeddings(self) -> torch.Tensor:
        """Return the raw token embedding matrix (p, d_model) for PCA visualization."""
        return self.token_embedding.weight.detach().clone()

    def get_attention_weights(self) -> list:
        """Return cached attention weights from the last forward pass."""
        return [block.last_attn_weights for block in self.blocks]

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Verify architecture and parameter count
    model = GrokTransformer(p=97, d_model=128, n_heads=4, d_ff=200, n_layers=1)
    print(f"Model architecture:\n{model}\n")
    print(f"Total trainable parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randint(0, 97, (8, 2))
    logits = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")  # (8, 97)
    print(f"Embeddings shape: {model.get_token_embeddings().shape}")  # (97, 128)

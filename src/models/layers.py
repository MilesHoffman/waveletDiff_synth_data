"""
Basic building blocks for wavelet diffusion transformers.

This module contains fundamental layers used across the wavelet diffusion model:
- Time embedding for diffusion timesteps
- Positional encoding for sequence modeling
- Adaptive layer normalization for conditioning
- Transformer blocks with time conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    A simplified normalization that only scales by RMS (no mean subtraction).
    Used for QK-Norm in attention to bound logit magnitudes.
    Compile-safe: no control flow, all operations are differentiable.
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class QKNormAttention(nn.Module):
    """Multi-Head Attention with Query-Key Normalization for training stability.
    
    Applies RMSNorm to Q and K after projection to bound attention logits,
    preventing numerical instability in bf16 and compiled modes.
    
    Uses F.scaled_dot_product_attention for hardware acceleration (Flash Attention).
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 batch_first: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout_p = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # QK-Norm: RMSNorm applied per-head to Q and K
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scale factor for attention
        self.register_buffer('scale', torch.tensor(self.head_dim ** -0.5))
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query, key, value: [batch, seq_len, embed_dim] if batch_first
            attn_mask: Optional attention mask
            need_weights: If True, return attention weights (for visualization)
            average_attn_weights: If True, average weights across heads
        
        Returns:
            output: Same shape as query
            attn_weights: Optional attention weights
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        
        # Apply QK-Norm (normalize along head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use PyTorch's SDPA (Flash Attention when available)
        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale.item()
        )
        
        # Reshape back: [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Handle batch_first for output
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Attention weights (for visualization, e.g., CrossLevelAttention)
        attn_weights = None
        if need_weights:
            with torch.no_grad():
                attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if attn_mask is not None:
                    attn_logits = attn_logits + attn_mask
                attn_weights = F.softmax(attn_logits, dim=-1)
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)
        
        return output, attn_weights


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create sinusoidal position encoding for time
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
        # MLP to process time embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, t):
        """
        Args:
            t: Normalized time values [0, 1], shape (batch_size,)
        
        Returns:
            Time embeddings of shape (batch_size, embed_dim)
        """
        # Scale to appropriate range and ensure float
        t = t.float() * 1000.0
        t = t.unsqueeze(-1)  # (batch_size, 1)
        
        # Create sinusoidal embeddings
        emb = t * self.emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, embed_dim)
        
        # Process through MLP
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""
    
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on time embedding."""
    
    def __init__(self, embed_dim, time_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        
        # Adaptive parameters predicted from time embedding
        self.ada_lin = nn.Linear(time_embed_dim, 2 * embed_dim)
        
        # Initialize to standard layer norm behavior
        with torch.no_grad():
            self.ada_lin.weight.zero_()
            self.ada_lin.bias.zero_()
            # Set bias to have scale=1, shift=0 initially
            self.ada_lin.bias[:embed_dim] = 1.0
    
    def forward(self, x, time_embed):
        """
        Args:
            x: Input tensor of shape (..., embed_dim)
            time_embed: Time embedding of shape (batch_size, time_embed_dim)
        
        Returns:
            Normalized tensor with adaptive scaling and shifting
        """
        # Normalize input
        x_norm = self.norm(x)
        
        # Predict adaptive parameters
        ada_params = self.ada_lin(time_embed)  # (batch_size, 2 * embed_dim)
        
        # Split into scale and shift
        scale, shift = ada_params.chunk(2, dim=-1)  # Each: (batch_size, embed_dim)
        
        # Expand to match input dimensions
        while scale.dim() < x_norm.dim():
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        
        # Apply adaptive transformation
        return scale * x_norm + shift


class WaveletTransformerBlock(nn.Module):
    """Transformer block with time conditioning for wavelet coefficient processing."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, time_embed_dim=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention
        self.norm1 = AdaLayerNorm(dim, time_embed_dim)
        self.attn = QKNormAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.norm2 = AdaLayerNorm(dim, time_embed_dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_embed, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            time_embed: Time embedding of shape (batch_size, time_embed_dim)
            mask: Optional attention mask
        
        Returns:
            Transformed tensor of same shape as input
        """
        # Self-attention with time conditioning
        x_norm1 = self.norm1(x, time_embed)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with time conditioning
        x_norm2 = self.norm2(x, time_embed)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        
        return x


class WaveletLevelTransformer(nn.Module):
    """Transformer for processing a specific wavelet level.
    
    Each wavelet coefficient in the level is treated as a separate token in the sequence,
    allowing the transformer to learn relationships between different coefficients within
    the same level. Positional encoding is added to provide spatial information about
    the coefficient positions.
    """
    
    def __init__(self, level_dim, num_features, embed_dim=128, num_heads=8, num_layers=4, 
                 time_embed_dim=64, dropout=0.1):
        super().__init__()
        self.level_dim = level_dim
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(num_features, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=level_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(embed_dim, num_heads, dropout=dropout, time_embed_dim=time_embed_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_features)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_embed):
        """
        Args:
            x: Wavelet coefficients of shape (batch_size, level_dim, num_features)
            time_embed: Time embedding of shape (batch_size, time_embed_dim)
        
        Returns:
            Processed coefficients of shape (batch_size, level_dim, num_features)
        """
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch_size, level_dim, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, time_embed)
        
        # Project back to coefficient space
        x = self.output_proj(x)  # (batch_size, level_dim, num_features)
        
        return x
    
    def get_embeddings(self, x, time_embed):
        """Get intermediate embeddings before final projection.
        
        Used for cross-level attention processing.
        
        Args:
            x: Wavelet coefficients of shape (batch_size, level_dim, num_features)
            time_embed: Time embedding of shape (batch_size, time_embed_dim)
        
        Returns:
            Embeddings of shape (batch_size, level_dim, embed_dim)
        """
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch_size, level_dim, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, time_embed)
        
        return x  # (batch_size, level_dim, embed_dim)
    
    def final_projection(self, x_embed):
        """Apply final projection to embeddings.
        
        Args:
            x_embed: Embeddings of shape (batch_size, level_dim, embed_dim)
        
        Returns:
            Coefficients of shape (batch_size, level_dim, num_features)
        """
        x = self.output_proj(x_embed)  # (batch_size, level_dim, num_features)
        return x




"""
Cross-level attention mechanisms for wavelet coefficients.

This module implements level-to-level attention where each wavelet level (as a whole)
attends to other levels, rather than individual coefficients attending to each other.
This creates a more balanced attention mechanism where levels with more coefficients
don't have disproportionate influence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import AdaLayerNorm, QKNormAttention


class CrossLevelAttention(nn.Module):
    """Cross-level attention mechanism for wavelet coefficients.
    
    This module implements level-to-level attention where each wavelet level (as a whole)
    attends to other levels, rather than individual coefficients attending to each other.
    This creates a more balanced attention mechanism where levels with more coefficients
    don't have disproportionate influence.
    """
    
    def __init__(self, level_embed_dims, common_dim=None, num_heads=8, dropout=0.1, 
                 time_embed_dim=64, attention_mode="all_to_all"):
        super().__init__()
        self.level_embed_dims = level_embed_dims
        self.num_levels = len(level_embed_dims)
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_mode = attention_mode  # "all_to_all" or "cross_only"
        
        # Use the maximum embedding dimension as common dimension if not specified
        self.common_dim = common_dim if common_dim is not None else max(level_embed_dims)
        
        
        # Project aggregated level representations to common dimension
        self.level_projections = nn.ModuleList()
        for embed_dim in level_embed_dims:
            projector = nn.Linear(embed_dim, self.common_dim)
            self.level_projections.append(projector)
        
        # Cross-level attention layers (now operating on level representations)
        if self.attention_mode == "all_to_all":
            # Standard multi-head attention (each level attends to all levels including itself)
            self.cross_attention = QKNormAttention(
                self.common_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:  # cross_only
            # Custom attention that prevents self-attention
            self.cross_attention_layers = nn.ModuleList()
            for i in range(self.num_levels):
                # Each level gets its own attention layer for attending to other levels
                attention_layer = QKNormAttention(
                    self.common_dim, num_heads, dropout=dropout, batch_first=True
                )
                self.cross_attention_layers.append(attention_layer)
        
        # Level expansion layers - distribute level information back to coefficients
        self.level_expanders = nn.ModuleList()
        for embed_dim in level_embed_dims:
            expander = nn.Linear(self.common_dim, embed_dim)
            self.level_expanders.append(expander)
        
        # Adaptive layer norm for cross-level attention
        self.cross_norm = nn.ModuleList()
        for embed_dim in level_embed_dims:
            norm = AdaLayerNorm(embed_dim, time_embed_dim)
            self.cross_norm.append(norm)
        
        # Learnable level position encodings to distinguish between levels
        self.level_position_embeddings = nn.Parameter(torch.randn(self.num_levels, self.common_dim))
        # Initialize level position encodings with more diverse values to help distinguish levels
        with torch.no_grad():
            self.level_position_embeddings.normal_(0, 0.1)
            # Ensure level position encodings are somewhat orthogonal
            for i in range(self.num_levels):
                self.level_position_embeddings[i] = F.normalize(self.level_position_embeddings[i], dim=0)
        
        # Gate to control how much cross-level information to use
        self.cross_level_gates = nn.ModuleList()
        for embed_dim in level_embed_dims:
            gate = nn.Sequential(
                nn.Linear(embed_dim * 2 + time_embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.cross_level_gates.append(gate)
    
    def get_cross_level_attention_weights(self, level_embeddings, time_embed):
        """
        Extract cross-level attention weights for visualization.
        This method mirrors the forward pass logic to ensure consistency.
        
        Args:
            level_embeddings: List of tensors, each of shape [batch_size, level_seq_len, level_embed_dim]
            time_embed: [batch_size, time_embed_dim]
        
        Returns:
            Attention weights tensor of shape [num_levels, num_levels]
        """
        batch_size = level_embeddings[0].shape[0]
        attention_matrix = torch.zeros(self.num_levels, self.num_levels, device=level_embeddings[0].device)
        
        # Step 1: Base representations (project to common_dim and add positional encoding)
        projected_levels = []
        for i, (level_emb, projector) in enumerate(zip(level_embeddings, self.level_projections)):
            proj_emb = projector(level_emb)
            proj_emb = proj_emb + self.level_position_embeddings[i].unsqueeze(0).unsqueeze(1)
            projected_levels.append(proj_emb)
            
        # Step 2 & 3: Pyramidal alignment and cross-attention
        for i in range(self.num_levels):
            target_seq_len = projected_levels[i].shape[1]
            query = projected_levels[i]
            
            keys_values = []
            source_indices = []
            
            for j in range(self.num_levels):
                if self.attention_mode == "cross_only" and i == j:
                    continue
                    
                source = projected_levels[j]
                if source.shape[1] != target_seq_len:
                    source_t = source.transpose(1, 2)
                    aligned_t = F.interpolate(source_t, size=target_seq_len, mode='linear', align_corners=False)
                    aligned = aligned_t.transpose(1, 2)
                else:
                    aligned = source
                keys_values.append(aligned)
                source_indices.append(j)
                
            if not keys_values:
                continue
                
            kv_tensor = torch.cat(keys_values, dim=1)
            
            if self.attention_mode == "all_to_all":
                _, attn_weights = self.cross_attention(query, kv_tensor, kv_tensor, average_attn_weights=True, need_weights=True)
            else:
                _, attn_weights = self.cross_attention_layers[i](query, kv_tensor, kv_tensor, average_attn_weights=True, need_weights=True)
                
            # attn_weights shape (if averaged over heads): [batch_size, target_seq_len, num_sources * target_seq_len]
            attn_weights_avg = attn_weights.mean(dim=0)
            
            # Split back into individual source blocks along key sequence dimension
            chunks = torch.chunk(attn_weights_avg, len(source_indices), dim=-1)
            
            for chunk, j in zip(chunks, source_indices):
                # chunk shape: [target_seq_len, target_seq_len]
                attention_matrix[i, j] = chunk.mean()
                
        return attention_matrix

    def forward(self, level_embeddings, time_embed):
        """
        Args:
            level_embeddings: List of tensors, each of shape [batch_size, level_seq_len, level_embed_dim]
            time_embed: [batch_size, time_embed_dim]
        
        Returns:
            List of tensors with same shapes as input, but with cross-level attention applied
        """
        batch_size = level_embeddings[0].shape[0]
        
        # Original embeddings are used for residual connections
        # No clone needed since we don't modify them in-place
        original_embeddings = level_embeddings
        
        # Step 1: Base representations (project to common_dim and add positional encoding)
        projected_levels = []
        for i, (level_emb, projector) in enumerate(zip(level_embeddings, self.level_projections)):
            proj_emb = projector(level_emb)
            proj_emb = proj_emb + self.level_position_embeddings[i].unsqueeze(0).unsqueeze(1)
            projected_levels.append(proj_emb)
            
        # Step 2 & 3: Pyramidal alignment and cross-attention
        cross_attended_levels = []
        for i in range(self.num_levels):
            target_seq_len = projected_levels[i].shape[1]
            query = projected_levels[i]
            
            keys_values = []
            for j in range(self.num_levels):
                if self.attention_mode == "cross_only" and i == j:
                    continue
                    
                source = projected_levels[j]
                if source.shape[1] != target_seq_len:
                    source_t = source.transpose(1, 2)
                    aligned_t = F.interpolate(source_t, size=target_seq_len, mode='linear', align_corners=False)
                    aligned = aligned_t.transpose(1, 2)
                else:
                    aligned = source
                keys_values.append(aligned)
                
            if not keys_values:
                cross_attended_levels.append(query)
                continue
                
            kv_tensor = torch.cat(keys_values, dim=1)
            
            if self.attention_mode == "all_to_all":
                out, _ = self.cross_attention(query, kv_tensor, kv_tensor)
            else:
                out, _ = self.cross_attention_layers[i](query, kv_tensor, kv_tensor)
                
            cross_attended_levels.append(out)
            
        # Step 4: Expand and Gated Residual
        output_embeddings = []
        for i, (cross_attn_out, expander, original_emb) in enumerate(
            zip(cross_attended_levels, self.level_expanders, original_embeddings)
        ):
            expanded_level = expander(cross_attn_out) # [batch, seq_len_i, embed_dim]
            
            # Apply adaptive layer norm
            cross_output_norm = self.cross_norm[i](expanded_level, time_embed)
            
            # Compute gate to control cross-level information
            time_embed_expanded = time_embed.unsqueeze(1).expand(-1, original_emb.shape[1], -1)
            # GLU-Style Gate evaluates incoming cross_output_norm along with original and time embeddings
            gate_input = torch.cat([original_emb, cross_output_norm, time_embed_expanded], dim=-1)
            gate = self.cross_level_gates[i](gate_input)
            
            # Apply gated residual connection
            output = original_emb + gate * cross_output_norm
            output_embeddings.append(output)
        
        return output_embeddings

"""
Geodesic Attention
==================

Distance-based attention mechanism in curved manifold space.

States that are geodesically "closer" attend to each other more strongly.

Physics Motivation:
- Locality in curved space determines interaction strength
- Geodesic distance is the natural metric in Riemannian geometry
- Similar to kernelized attention with learned metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicAttention(nn.Module):
    """
    Attention mechanism based on geodesic distances in manifold.
    
    Computes pairwise distances and uses exp(-dist/T) as attention weights.
    States closer in the manifold geometry attend more strongly.
    
    Args:
        dim: State dimension
        temperature: Attention temperature (default: 1.0)
        distance_metric: 'euclidean' or 'riemannian' (default: 'euclidean')
            - 'euclidean': Simple L2 distance (fast)
            - 'riemannian': Learned metric via Q/K/V projections (slower, more expressive)
    
    Example:
        >>> attention = GeodesicAttention(dim=128, temperature=0.5, distance_metric='riemannian')
        >>> x_agg, v_agg, attn_matrix = attention(x_seq, v_seq)
        >>> # attn_matrix[i, j] = attention from token i to token j
    """
    
    def __init__(self, dim, temperature=1.0, distance_metric='euclidean'):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.distance_metric = distance_metric
        
        if distance_metric == 'riemannian':
            # Learnable metric for geodesic distance (like transformer Q/K/V)
            self.query_proj = nn.Linear(dim, dim)
            self.key_proj = nn.Linear(dim, dim)
            self.value_proj = nn.Linear(dim, dim)
    
    def euclidean_distance(self, x1, x2):
        """
        Compute pairwise Euclidean distances.
        
        Args:
            x1: [B, L, dim]
            x2: [B, L, dim]
        
        Returns:
            dist: [B, L, L] pairwise distances
        """
        # Expand for broadcasting
        x1_exp = x1.unsqueeze(2)  # [B, L, 1, dim]
        x2_exp = x2.unsqueeze(1)  # [B, 1, L, dim]
        
        # Squared Euclidean distance
        dist_sq = ((x1_exp - x2_exp) ** 2).sum(dim=-1)  # [B, L, L]
        dist = torch.sqrt(dist_sq + 1e-8)  # Add epsilon for numerical stability
        
        return dist
    
    def riemannian_distance(self, x):
        """
        Approximate Riemannian distance using learned metric.
        
        Uses Q/K projections similar to transformers to define metric.
        
        Args:
            x: [B, L, dim]
        
        Returns:
            dist: [B, L, L] approximate geodesic distances
        """
        # Project to metric space
        Q = self.query_proj(x)  # [B, L, dim]
        K = self.key_proj(x)    # [B, L, dim]
        
        # Similarity (negative distance proxy)
        # High Q·K means states are close in learned metric
        similarity = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)  # [B, L, L]
        
        # Convert to distance (high similarity = low distance)
        dist = -similarity
        
        return dist
    
    def forward(self, x_seq, v_seq, aggregate_v=True):
        """
        Aggregate states using geodesic distance attention.
        
        Args:
            x_seq: Position sequence [B, L, dim]
            v_seq: Velocity sequence [B, L, dim]
            aggregate_v: If True, also aggregate velocity (default: True)
        
        Returns:
            x_agg: Aggregated position [B, dim]
            v_agg: Aggregated velocity [B, dim] or None
            attn_weights: Attention weights [B, L, L] for interpretability
        """
        B, L, dim = x_seq.shape
        
        # Compute pairwise distances
        if self.distance_metric == 'euclidean':
            dist = self.euclidean_distance(x_seq, x_seq)  # [B, L, L]
        else:  # 'riemannian'
            dist = self.riemannian_distance(x_seq)  # [B, L, L]
        
        # Attention weights (closer = higher weight)
        # Use negative distance: exp(-dist/T) gives higher weight to closer points
        attn_weights = F.softmax(-dist / self.temperature, dim=-1)  # [B, L, L]
        
        # Aggregate positions weighted by attention
        if self.distance_metric == 'riemannian':
            # Use learned value projection
            V = self.value_proj(x_seq)  # [B, L, dim]
            x_attended = torch.bmm(attn_weights, V)  # [B, L, dim]
            # Use last token's aggregated state
            x_agg = x_attended[:, -1]  # [B, dim]
        else:
            # Simple weighted average
            x_attended = torch.bmm(attn_weights, x_seq)  # [B, L, dim]
            x_agg = x_attended[:, -1]  # [B, dim]
        
        # Aggregate velocity if requested
        if aggregate_v:
            v_attended = torch.bmm(attn_weights, v_seq)  # [B, L, dim]
            v_agg = v_attended[:, -1]  # [B, dim]
        else:
            v_agg = None
        
        return x_agg, v_agg, attn_weights

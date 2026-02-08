"""
Geodesic Aggregation Components
=================================

Physical mechanisms for aggregating information across sequence states
in geodesic flow.

Three approaches:
1. **Hamiltonian Pooling** - Energy-weighted aggregation
2. **Geodesic Attention** - Distance-based attention in manifold
3. **Momentum Accumulation** - Trajectory-based integration

Usage:
    >>> from gfn.aggregation import HamiltonianPooling, GeodesicAttention, MomentumAccumulation
    >>> 
    >>> # Or use factory
    >>> from gfn.aggregation import create_aggregation
    >>> pooling = create_aggregation('hamiltonian', dim=128, temperature=0.5)
"""

from .hamiltonian_pooling import HamiltonianPooling
from .geodesic_attention import GeodesicAttention
from .momentum_accumulation import MomentumAccumulation


def create_aggregation(agg_type, dim, **kwargs):
    """
    Factory function to create aggregation module.
    
    Args:
        agg_type: 'hamiltonian', 'geodesic', or 'momentum'
        dim: State dimension
        **kwargs: Additional arguments for specific aggregator
    
    Returns:
        aggregator: nn.Module instance
    
    Example:
        >>> agg = create_aggregation('hamiltonian', dim=128, temperature=0.5)
        >>> x_agg, v_agg, info = agg(x_seq, v_seq)
    """
    agg_map = {
        'hamiltonian': HamiltonianPooling,
        'geodesic': GeodesicAttention,
        'momentum': MomentumAccumulation
    }
    
    if agg_type not in agg_map:
        raise ValueError(
            f"Unknown aggregation type: {agg_type}. "
            f"Choose from: {list(agg_map.keys())}"
        )
    
    return agg_map[agg_type](dim, **kwargs)


__all__ = [
    'HamiltonianPooling',
    'GeodesicAttention',
    'MomentumAccumulation',
    'create_aggregation'
]

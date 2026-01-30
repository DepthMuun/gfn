"""
Neural Embeddings
=================

Advanced embedding strategies using implicit neural fields and SIREN.

Available Embeddings:
    - ImplicitEmbedding: Learnable coordinates + SIREN network
    - FunctionalEmbedding: Pure functional (zero-lookup) embedding
    - SineLayer: SIREN layer for building custom embeddings

Theory:
    Instead of storing E[i] for every token i, we store low-rank coordinate c[i]
    and learn continuous function f(c) → R^D.
    
    Benefits:
        1. Infinite vocabulary (via hashing)
        2. Smooth interpolation (metric topology)
        3. Massive parameter reduction (O(1) vs O(V))
"""

from .siren import SineLayer
from .implicit import ImplicitEmbedding
from .functional import FunctionalEmbedding

__all__ = [
    'SineLayer',
    'ImplicitEmbedding',
    'FunctionalEmbedding',
]

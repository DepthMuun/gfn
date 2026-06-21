"""
GeometryFactory — GFN V5
Creates geometry instances from PhysicsConfig.
Supports: euclidean, torus, low_rank, reactive, adaptive, hyperbolic, holographic.
"""

from typing import Optional
from ..config.schema import PhysicsConfig
from ..registry import GEOMETRY_REGISTRY
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
import logging

logger = logging.getLogger(__name__)

_GEOMETRIES_REGISTERED = False

def _register_all_geometries():
    """Importa los submódulos explícitamente para registrar las geometrías."""
    global _GEOMETRIES_REGISTERED
    if _GEOMETRIES_REGISTERED:
        return
    from . import euclidean
    from . import torus
    from . import low_rank
    from . import adaptive
    from . import reactive
    from . import hyperbolic
    from . import holographic
    from . import hierarchical
    from . import spherical
    _GEOMETRIES_REGISTERED = True

class GeometryFactory:
    """
    Creates manifold geometries from configuration.

    Primary key: topology.type  ('euclidean', 'torus', 'hyperbolic', ...)
    Secondary key: topology.riemannian_type  ('low_rank', 'reactive', 'adaptive', ...)

    riemannian_type overrides topology.type when explicitly set and registered.
    """

    @staticmethod
    def _lookup_key(config: PhysicsConfig) -> str:
        _register_all_geometries()
        topo_type = config.topology.type.lower()
        riem_type = getattr(config.topology, 'riemannian_type', 'reactive').lower()
        available = GEOMETRY_REGISTRY.list_keys()
        explicit_keys = set(getattr(config, '_explicit_keys', set()))

        riem_explicit = any(
            key in explicit_keys
            for key in (
                'riemannian_type',
                'topology.riemannian_type',
                'physics.topology.riemannian_type',
                'topology_riemannian_type',
            )
        )
        
        # Priority Logic:
        # 1. Specialized analytical topologies win by default.
        #    Learned riemannian geometries only override them when the user
        #    explicitly requested riemannian_type.
        learned_types = {'low_rank', 'reactive', 'adaptive', 'low_rank_paper'}
        if topo_type in available and topo_type != TOPOLOGY_EUCLIDEAN:
            if riem_explicit and riem_type in learned_types and riem_type in available:
                return riem_type
            return topo_type

        # 2. For Euclidean/default spaces, prefer learned geometries when available.
        if riem_type in learned_types and riem_type in available:
            return riem_type

        # 3. Fallback to riem_type or topo_type
        if riem_type in available:
            return riem_type
            
        return topo_type

    @staticmethod
    def create(config: PhysicsConfig):
        """
        Create geometry using default dim from config.
        Looks for 'dim' in topology config or falls back to 64.
        """
        lookup_key = GeometryFactory._lookup_key(config)
        available = GEOMETRY_REGISTRY.list_keys()

        if lookup_key in available:
            geometry_cls = GEOMETRY_REGISTRY.get(lookup_key)
            try:
                dim = getattr(config, 'dim', 64)
                rank = getattr(config.topology, 'riemannian_rank', 16)
                return geometry_cls(dim=dim, rank=rank, config=config)
            except TypeError:
                try:
                    return geometry_cls(config=config)
                except TypeError:
                    return geometry_cls()

        logger.warning(f"Geometry '{lookup_key}' not found. Using EuclideanGeometry.")
        from .euclidean import EuclideanGeometry
        return EuclideanGeometry(config=config)

    @staticmethod
    def create_with_dim(dim: int, rank: int, num_heads: int, config: PhysicsConfig):
        """
        Create geometry with explicit dim and rank.
        Used by ModelFactory to pass head_dim (not total dim) to the geometry,
        since geometry operates on per-head tensors [B, H, HD].
        """
        lookup_key = GeometryFactory._lookup_key(config)
        available = GEOMETRY_REGISTRY.list_keys()

        if lookup_key in available:
            geometry_cls = GEOMETRY_REGISTRY.get(lookup_key)
            try:
                return geometry_cls(dim=dim, rank=rank, num_heads=num_heads, config=config)
            except TypeError:
                try:
                    return geometry_cls(dim=dim, rank=rank, config=config)
                except TypeError:
                    try:
                         return geometry_cls(config=config)
                    except TypeError:
                         return geometry_cls()

        logger.warning(f"Geometry '{lookup_key}' not found. Using EuclideanGeometry.")
        from .euclidean import EuclideanGeometry
        try:
             return EuclideanGeometry(dim=dim, num_heads=num_heads, config=config)
        except TypeError:
             return EuclideanGeometry(config=config)

# gfn/geometry/__init__.py
# Centralized geometry API for the GFN framework.

try:
    from ..realizations.gssm.geometry import (
        ToroidalRiemannianGeometry,
        FlatToroidalRiemannianGeometry,
        ReactiveRiemannianGeometry,
        HyperRiemannianGeometry,
        AdaptiveRiemannianGeometry,
        LowRankRiemannianGeometry,
        PaperLowRankRiemannianGeometry,
        HolographicRiemannianGeometry,
        SphericalGeometry,
        HierarchicalGeometry,
        BaseGeometry,
        GeometryFactory
    )
except ImportError:
    pass

__all__ = [
    "BaseGeometry",
    "GeometryFactory",
    "ToroidalRiemannianGeometry",
    "FlatToroidalRiemannianGeometry",
    "ReactiveRiemannianGeometry",
    "HyperRiemannianGeometry",
    "AdaptiveRiemannianGeometry",
    "LowRankRiemannianGeometry",
    "PaperLowRankRiemannianGeometry",
    "HolographicRiemannianGeometry",
    "SphericalGeometry",
    "HierarchicalGeometry",
]

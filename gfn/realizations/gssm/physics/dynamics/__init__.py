"""
gfn/physics/dynamics/__init__.py — GFN V5
Ported from: gfn_old/nn/layers/flow/dynamics/

Dynamics System: 5 state update modes on the manifold.
"""
from .base import BaseDynamics
from .direct import DirectDynamics
from .residual import ResidualDynamics
from .mix import MixDynamics
from .gated import GatedDynamics
from .stochastic import StochasticDynamics
from typing import Optional
import torch.nn as nn
from ...constants import TOPOLOGY_EUCLIDEAN

DYNAMICS_REGISTRY = {
    'direct':     DirectDynamics,
    'residual':   ResidualDynamics,
    'mix':        MixDynamics,
    'gated':      GatedDynamics,
    'stochastic': StochasticDynamics,
}


def get_dynamics(dynamics_type: str, dim: int,
                 norm_layer: Optional[nn.Module] = None,
                 topology: str = TOPOLOGY_EUCLIDEAN, **kwargs) -> BaseDynamics:
    """
    Dynamics Factory.

    For POSITION tensors: pass topology=self.topology
    For VELOCITY tensors: always pass topology=TOPOLOGY_EUCLIDEAN (tangent space)
    """
    dynamics_type = dynamics_type.lower()
    dynamics_cls = DYNAMICS_REGISTRY.get(dynamics_type)
    if dynamics_cls is None:
        raise ValueError(
            f"Unknown dynamics type: '{dynamics_type}'. "
            f"Available: {list(DYNAMICS_REGISTRY.keys())}"
        )
    return dynamics_cls(dim, norm_layer, topology=topology, **kwargs)


__all__ = [
    'BaseDynamics', 'DirectDynamics', 'ResidualDynamics', 'MixDynamics',
    'GatedDynamics', 'StochasticDynamics', 'get_dynamics', 'DYNAMICS_REGISTRY',
]

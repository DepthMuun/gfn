"""
gfn/physics/dynamics/__init__.py — GFN V5
Portado desde: gfn_old/nn/layers/flow/dynamics/

Sistema de Dynamics: 5 modos de actualización de estado en el manifold.
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
    Factory de Dynamics.

    Para tensores de POSICIÓN: pasar topology=self.topology
    Para tensores de VELOCIDAD: pasar topology=TOPOLOGY_EUCLIDEAN siempre (espacio tangente)
    """
    dynamics_type = dynamics_type.lower()
    dynamics_cls = DYNAMICS_REGISTRY.get(dynamics_type)
    if dynamics_cls is None:
        raise ValueError(
            f"Tipo de dynamics desconocido: '{dynamics_type}'. "
            f"Disponibles: {list(DYNAMICS_REGISTRY.keys())}"
        )
    return dynamics_cls(dim, norm_layer, topology=topology, **kwargs)


__all__ = [
    'BaseDynamics', 'DirectDynamics', 'ResidualDynamics', 'MixDynamics',
    'GatedDynamics', 'StochasticDynamics', 'get_dynamics', 'DYNAMICS_REGISTRY',
]

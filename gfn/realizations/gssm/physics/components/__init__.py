"""
Physics Components — GFN V5
============================
Modular physics components for singularity handling, hysteresis dynamics, and friction.
"""

from .singularities import (
    SingularityGate,
    SingularityDetector,
    SingularityRegistry,
)

from .hysteresis import (
    HysteresisModule,
    HysteresisState,
    HysteresisRegistry,
)

from .friction import (
    FrictionGate,
    AdaptiveFriction,
    FrictionRegistry,
)

from .stochasticity import (
    BrownianForce,
    OUDynamicsForce,
)

from .curiosity import (
    GeometricCuriosityForce,
)

__all__ = [
    # Singularity components
    'SingularityGate',
    'SingularityDetector', 
    'SingularityRegistry',
    
    # Hysteresis components
    'HysteresisModule',
    'HysteresisState',
    'HysteresisRegistry',
    
    # Friction components
    'FrictionGate',
    'AdaptiveFriction',
    'FrictionRegistry',

    # Stochastic / Exploration components
    'BrownianForce',
    'OUDynamicsForce',
    'GeometricCuriosityForce',
]

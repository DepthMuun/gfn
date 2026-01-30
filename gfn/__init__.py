"""
GFN: Geodesic Flow Networks
===========================

A novel neural architecture that models sequences as flows on Riemannian manifolds.

Available Integrators:
    - HeunIntegrator: 2nd order, fast (RECOMMENDED)
    - RK4Integrator: 4th order, accurate but slower
    - SymplecticIntegrator: Velocity Verlet, energy-preserving
    - LeapfrogIntegrator: Störmer-Verlet, best symplectic

Usage:
    from gfn import GFN
    model = GFN(vocab_size=16, dim=512, depth=12, rank=128, integrator_type='heun')
    
    # Physics-informed training:
    from gfn import GFNLoss, RiemannianAdam
    criterion = GFNLoss(lambda_h=0.01)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
"""

__version__ = "2.5.0"
__author__ = "Manifold Laboratory (Joaquín Stürtz)"

# Core Models (from core/ package)
from .core import Manifold as GFN  # Alias for backward compatibility
from .core import Manifold, AdjointManifold
# Legacy aliases
AdjointGFN = AdjointManifold

# Model Components (from model/ package)
from .model.state import ManifoldState
from .model.fusion import CUDAFusionManager

# Layers
from .layers import MLayer as GLayer  # Alias
from .layers import MLayer, ParallelMLayer, RiemannianGating

# Geometry
from .geometry import LowRankChristoffel

# Integrators
from .integrators import (
    HeunIntegrator,
    RK4Integrator,
    SymplecticIntegrator,
    LeapfrogIntegrator,
    YoshidaIntegrator,
    DormandPrinceIntegrator,
    EulerIntegrator,
    ForestRuthIntegrator,
    OmelyanIntegrator,
    CouplingFlowIntegrator,
    NeuralIntegrator,
)

# Readouts
from .readouts import ImplicitReadout

# Loss Functions
from .losses import (
    hamiltonian_loss,
    geodesic_regularization,
    GFNLoss,
)

# Optimizers
from .optimizers import (
    RiemannianAdam,
    ManifoldSGD,
)

# Datasets
from .datasets import MathDataset, MixedHFDataset

# Utilities
from .utils import parallel_scan, GPUMonitor

# Registry
# Registry
INTEGRATORS = {
    'euler': EulerIntegrator,
    'heun': HeunIntegrator,
    'rk4': RK4Integrator,
    'rk45': DormandPrinceIntegrator, # Alias for DP
    'symplectic': SymplecticIntegrator,
    'leapfrog': LeapfrogIntegrator,
    'yoshida': YoshidaIntegrator,
    'forest_ruth': ForestRuthIntegrator,
    'omelyan': OmelyanIntegrator,
    'coupling': CouplingFlowIntegrator,
    'neural': NeuralIntegrator,
}

__all__ = [
    "GFN",
    "Manifold",  # Export base class
    "ManifoldState",  # State management
    "CUDAFusionManager",  # CUDA fusion
    "GLayer", "RiemannianGating",
    "LowRankChristoffel", 
    "HeunIntegrator", "RK4Integrator", "SymplecticIntegrator", "LeapfrogIntegrator", 
    "YoshidaIntegrator", "DormandPrinceIntegrator", "EulerIntegrator",
    "ForestRuthIntegrator", "OmelyanIntegrator", "CouplingFlowIntegrator", "NeuralIntegrator",
    "INTEGRATORS",
    "ImplicitReadout",
    "hamiltonian_loss", "geodesic_regularization", "GFNLoss",
    "RiemannianAdam", "ManifoldSGD",
    "MathDataset", "MixedHFDataset",
    "GPUMonitor",
]

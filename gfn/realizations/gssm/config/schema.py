# schema.py — GFN V5
# Configuration class definitions (Schema)
# SEPARATION: Default values go to defaults.py, physical constants to constants.py

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

# Import correct physical constants
from ..constants import (
    EPSILON_STANDARD,
    TOPOLOGY_TORUS,
    MIN_DT,
    MAX_DT,
    CURVATURE_CLAMP,
    SINGULARITY_THRESHOLD,
    BLACK_HOLE_STRENGTH,
    DEFAULT_DT,
    DEFAULT_FRICTION,
    DEFAULT_PLASTICITY,
    MAX_VELOCITY,
)


@dataclass
class TopologyConfig:
    """Manifold topology configuration."""
    type: str = TOPOLOGY_TORUS 
    R: float = 2.0           # Major radius of torus (default)
    r: float = 1.0           # Minor radius of torus (default)
    curvature: float = 0.0
    riemannian_type: str = 'reactive' 
    riemannian_rank: int = 16
    riemannian_class: Optional[str] = None
    geometry_scope: str = 'local'  # 'local' (dim/heads) or 'global' (full dim)
    # NEW: Learnable parameters
    learnable_R: bool = True   # Make R learnable (as per paper)
    learnable_r: bool = True   # Make r learnable (as per paper)


@dataclass
class StabilityConfig:
    """Numerical stability configuration."""
    base_dt: float = DEFAULT_DT 
    adaptive: bool = True
    dt_min: float = MIN_DT
    dt_max: float = MAX_DT
    enable_trace_normalization: bool = True
    wrap_x: bool = True
    friction: float = DEFAULT_FRICTION 
    velocity_friction_scale: float = 0.0
    velocity_saturation: float = 0.0  # P2.3: 0 = disabled, >0 = clamp magnitude via tanh
    curvature_clamp: float = CURVATURE_CLAMP
    friction_mode: str = 'static'  # 'static' or 'lif'
    integrator_type: str = 'leapfrog'
    toroidal_curvature_scale: float = 0.01  # scale for torus Christoffel contribution


@dataclass
class DynamicTimeConfig:
    enabled: bool = False
    type: str = 'riemannian'


@dataclass
class HysteresisConfig:
    enabled: bool = False
    ghost_force: bool = True
    hyst_decay: float = 0.1
    hyst_update_w: float = 1.0
    hyst_update_b: float = 0.0
    hyst_readout_w: float = 1.0
    hyst_readout_b: float = 0.0


@dataclass
class ActiveInferenceConfig:
    enabled: bool = False
    holographic_geometry: bool = False 
    thermodynamic_geometry: bool = False
    plasticity: float = DEFAULT_PLASTICITY
    dynamic_time: DynamicTimeConfig = field(default_factory=DynamicTimeConfig)
    reactive_curvature: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, 
        "plasticity": 0.0
    })
    geodesic_lensing: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    
    # Exploration / Noise
    stochasticity: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, 
        "type": "brownian", 
        "sigma": 0.01, 
        "theta": 0.15, 
        "mu": 0.0
    })
    curiosity: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, 
        "strength": 0.1, 
        "decay": 0.99
    })


@dataclass
class EmbeddingConfig:
    type: str = 'standard'
    mode: str = 'linear'
    coord_dim: int = 16
    impulse_scale: float = 1.0
    omega_0: float = 30.0


@dataclass
class ReadoutConfig:
    type: str = 'standard'


@dataclass
class MixtureConfig:
    coupler_mode: str = 'mean_field'


@dataclass
class DynamicsConfig:
    type: str = 'direct'


@dataclass
class FractalConfig:
    enabled: bool = False
    threshold: float = 0.5
    alpha: float = 0.2


@dataclass
class SingularityConfig:
    enabled: bool = False
    epsilon: float = EPSILON_STANDARD
    strength: float = BLACK_HOLE_STRENGTH
    threshold: float = SINGULARITY_THRESHOLD


@dataclass
class PhysicsConfig:
    """Complete physics configuration."""
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    active_inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    singularities: SingularityConfig = field(default_factory=SingularityConfig)
    trajectory_mode: str = 'partition'
    lensing: Dict[str, Any] = field(default_factory=lambda: {'enabled': False})
    checkpointing: Dict[str, Any] = field(default_factory=lambda: {'enabled': False})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    optimizer: str = 'adamw'
    max_lr: Optional[float] = None
    total_steps: Optional[int] = None
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'lambda_g': 0.001,
        'lambda_h': 0.0,
        'geodesic_mode': 'magnitude'
    })


@dataclass
class ManifoldConfig:
    """Main Manifold model configuration."""
    vocab_size: int
    dim: int = 512
    depth: int = 4
    heads: int = 4
    rank: int = 32
    integrator: str = 'leapfrog'
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    adjoint_enabled: bool = False
    adjoint_rtol: float = 1e-4
    adjoint_atol: float = 1e-4
    holographic: bool = False
    impulse_scale: float = 1.0
    dynamics_type: str = 'direct'
    mixer_type: str = 'low_rank'
    trajectory_mode: str = 'partition'
    coupler_mode: str = 'mean_field'
    initial_spread: float = 0.1  # 0.1 recommended for better gradient flow

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

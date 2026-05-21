"""
Configuration validation — GFN V5
Verifies parameter compatibility before building components.
Merged from utils/validation.py and original config/validator.py.
"""

from typing import Dict, Any, List, Optional
from .schema import ManifoldConfig, PhysicsConfig
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN, TOPOLOGY_SPHERE

class ConfigValidationError(Exception):
    """Critical configuration validation error."""
    pass

class ConfigValidator:
    """Central validator for GFN configurations."""
    
    @staticmethod
    def validate_physics(cfg: PhysicsConfig, dim: Optional[int] = None, heads: Optional[int] = None):
        """
        Validate physical and architectural consistency of PhysicsConfig.
        Raises ConfigValidationError if strict topology/stability rules are violated.
        """
        # 1. Topology checks
        if cfg.topology.type == TOPOLOGY_TORUS:
            if dim is not None and heads is not None:
                head_dim = dim // heads
                if head_dim % 2 != 0:
                    raise ConfigValidationError(
                        f"Toroid geometry requires head_dim (dim//heads) to be even. "
                        f"Found {dim}//{heads}={head_dim}"
                    )
            
        if cfg.topology.type == TOPOLOGY_SPHERE and cfg.topology.curvature <= 0:
             raise ConfigValidationError("Spherical topology requires positive curvature.")

        # 2. Stability checks
        if cfg.stability.base_dt <= 0:
            raise ConfigValidationError("base_dt must be positive.")
        if cfg.stability.friction < 0:
            raise ConfigValidationError("friction cannot be negative.")
        
        # 3. Mode Compatibility
        if cfg.trajectory_mode == 'ensemble' and heads is not None and heads <= 1:
            raise ConfigValidationError("Ensemble trajectory mode requires more than 1 head.")


def validate_manifold_config(config: ManifoldConfig) -> List[str]:
    """
    Validates a complete ManifoldConfig and its nested PhysicsConfig.
    Returns list of warnings (empty if everything OK).
    Raises ConfigValidationError on critical errors or compatibility issues.
    """
    warnings = []

    # Critical validations (Raise exceptions)
    if config.dim % config.heads != 0:
        raise ConfigValidationError(
            f"dim={config.dim} is not divisible by heads={config.heads}. "
            f"head_dim={config.dim/config.heads:.1f} is not an integer."
        )

    if config.vocab_size <= 0:
        raise ConfigValidationError(f"vocab_size={config.vocab_size} must be > 0.")
        
    if config.depth <= 0:
        raise ConfigValidationError(f"depth={config.depth} must be > 0.")

    # Validate Physics properties via centralized method
    ConfigValidator.validate_physics(config.physics, config.dim, config.heads)

    # Soft validations (Warnings)
    head_dim = config.dim // config.heads
    topo_type = config.physics.topology.type.lower()
    
    if topo_type == TOPOLOGY_TORUS and head_dim % 2 != 0:
        warnings.append(
            f"[WARN] For toroidal geometry, head_dim={head_dim} should be even "
            f"for sin/cos representations. Consider using heads={config.dim // (head_dim + 1)} or similar."
        )

    if config.rank > config.dim:
        warnings.append(
            f"[WARN] rank={config.rank} > dim={config.dim}. "
            f"Decomposition is not low-rank. Intentional?"
        )

    dt = config.physics.stability.base_dt
    if dt > 1.0:
        warnings.append(f"[WARN] base_dt={dt} > 1.0 may cause numerical instability.")
    if dt < 1e-5:
        warnings.append(f"[WARN] base_dt={dt} < 1e-5 may slow convergence.")

    return warnings


def validate_and_print(config: ManifoldConfig) -> bool:
    """
    Validates the configuration and prints warnings.
    Returns True if valid, False if there were errors.
    """
    try:
        warnings = validate_manifold_config(config)
        for w in warnings:
            print(w)
        print(f"[OK] Configuration valid. {len(warnings)} warning(s).")
        return True
    except ConfigValidationError as e:
        print(f"[ERROR] Validation failed: {e}")
        return False

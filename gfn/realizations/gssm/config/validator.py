"""
Validación de configuraciones — GFN V5
Verifica la compatibilidad de parámetros antes de construir componentes.
Fusionado de utils/validation.py y config/validator.py original.
"""

from typing import Dict, Any, List, Optional
from .schema import ManifoldConfig, PhysicsConfig
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN, TOPOLOGY_SPHERE

class ConfigValidationError(Exception):
    """Error de validación de configuración crítica."""
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
    Valida un ManifoldConfig completo y su PhysicsConfig anidado.
    Retorna lista de warnings (vacía si todo está OK).
    Lanza ConfigValidationError en errores críticos o de compatibilidad.
    """
    warnings = []

    # Validaciones críticas (Raise exceptions)
    if config.dim % config.heads != 0:
        raise ConfigValidationError(
            f"dim={config.dim} no es divisible por heads={config.heads}. "
            f"head_dim={config.dim/config.heads:.1f} no es entero."
        )

    if config.vocab_size <= 0:
        raise ConfigValidationError(f"vocab_size={config.vocab_size} debe ser > 0.")
        
    if config.depth <= 0:
        raise ConfigValidationError(f"depth={config.depth} debe ser > 0.")

    # Validate Physics properties via centralized method
    ConfigValidator.validate_physics(config.physics, config.dim, config.heads)

    # Validaciones suaves (Warnings)
    head_dim = config.dim // config.heads
    topo_type = config.physics.topology.type.lower()
    
    if topo_type == TOPOLOGY_TORUS and head_dim % 2 != 0:
        warnings.append(
            f"[WARN] Para geometría toroidal, head_dim={head_dim} debería ser par "
            f"para representaciones sin/cos. Considera usar heads={config.dim // (head_dim + 1)} o similar."
        )

    if config.rank > config.dim:
        warnings.append(
            f"[WARN] rank={config.rank} > dim={config.dim}. "
            f"La descomposición no es de rango bajo. ¿Intencional?"
        )

    dt = config.physics.stability.base_dt
    if dt > 1.0:
        warnings.append(f"[WARN] base_dt={dt} > 1.0 puede causar inestabilidad numérica.")
    if dt < 1e-5:
        warnings.append(f"[WARN] base_dt={dt} < 1e-5 puede ralentizar convergencia.")

    return warnings


def validate_and_print(config: ManifoldConfig) -> bool:
    """
    Valida la configuración e imprime warnings.
    Retorna True si es válida, False si hubo errores.
    """
    try:
        warnings = validate_manifold_config(config)
        for w in warnings:
            print(w)
        return True
    except ConfigValidationError as e:
        print(f"[CONFIG ERROR] {e}")
        return False

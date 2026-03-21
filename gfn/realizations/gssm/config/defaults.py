"""
config/defaults.py — GFN V5
Valores por defecto centralizados para todas las configuraciones.
Elimina hardcodes dispersos en implementaciones.
"""

from typing import Dict, Any
from ..constants import (
    DEFAULT_DT, DEFAULT_FRICTION, DEFAULT_PLASTICITY,
    MAX_VELOCITY, CURVATURE_CLAMP, VELOCITY_FRICTION_SCALE,
    SINGULARITY_THRESHOLD, BLACK_HOLE_STRENGTH, EPSILON_STANDARD, 
    TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
)

# ─── Física ─────────────────────────────────────────────────────────────────
PHYSICS_DEFAULTS: Dict[str, Any] = {
    # Topología
    'topology_type': TOPOLOGY_EUCLIDEAN,
    'riemannian_type': 'low_rank',
    'major_radius_R': 2.0,
    'minor_radius_r': 1.0,

    # Estabilidad — referencias a constants.py (una sola fuente de verdad)
    'base_dt': DEFAULT_DT,
    'adaptive_dt': True,
    'friction': DEFAULT_FRICTION,
    'velocity_clamp': MAX_VELOCITY,
    'curvature_clamp': CURVATURE_CLAMP,
    'enable_trace_normalization': True,
    'velocity_friction_scale': VELOCITY_FRICTION_SCALE,
    'integrator_type': 'leapfrog',
    'friction_mode': 'static',

    # Inferencia activa
    'active_inference_enabled': True,
    'holographic_geometry': False,
    'plasticity': DEFAULT_PLASTICITY,

    # Singularities
    'singularity_enabled': False,
    'singularity_threshold': SINGULARITY_THRESHOLD,
    'singularity_strength': BLACK_HOLE_STRENGTH,
    'singularity_epsilon': EPSILON_STANDARD,

    # Hysteresis
    'hysteresis_enabled': False,
    'hysteresis_decay': 0.95,
    'hysteresis_ghost_force': True,

    # Stochasticity / Curiosity
    'stochasticity_enabled': False,
    'stochasticity_type': 'brownian',
    'stochasticity_sigma': 0.01,
    'curiosity_enabled': False,
    'curiosity_strength': 0.1,
}

# ─── Modelo ──────────────────────────────────────────────────────────────────
MODEL_DEFAULTS: Dict[str, Any] = {
    'dim': 64,
    'heads': 4,
    'depth': 2,
    'rank': 16,
    'vocab_size': 256,
    'holographic': False,
    'pooling_type': None,
    'initial_spread': 1e-3,
    'n_trajectories': 1,
}

# ─── Entrenamiento ───────────────────────────────────────────────────────────
TRAINING_DEFAULTS: Dict[str, Any] = {
    'lr': 1e-3,
    'optimizer_type': 'adam',
    'weight_decay': 0.0,
    'grad_clip': 1.0,
    'epochs': 10,
    'batch_size': 32,
    'scheduler_type': 'cosine_warmup',
    'warmup_steps': 100,
    'min_lr': 1e-6,
    'task': 'lm',
}

# ─── Pérdidas ────────────────────────────────────────────────────────────────
LOSS_DEFAULTS: Dict[str, Any] = {
    'type': 'generative',
    'mode': 'nll',
    'entropy_coef': 0.0,
    'label_smoothing': 0.0,

    # Physics-informed
    'lambda_physics': 0.01,
    'lambda_geo': 0.001,
    'lambda_ham': 0.0,
    'lambda_kin': 0.0,
}


def get_default(section: str, key: str, fallback=None):
    """
    Obtiene un valor por defecto desde la sección correspondiente.
    Uso: get_default('physics', 'base_dt') -> 0.1
    """
    mapping = {
        'physics': PHYSICS_DEFAULTS,
        'model': MODEL_DEFAULTS,
        'training': TRAINING_DEFAULTS,
        'loss': LOSS_DEFAULTS,
    }
    return mapping.get(section, {}).get(key, fallback)

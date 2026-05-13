"""
config/loader.py — GFN V5
Conversión de dicts de configuración a PhysicsConfig tipado.
Soporte para overrides anidados sobre configs existentes.
"""
from typing import Dict, Any, Optional
from .schema import (
    PhysicsConfig, TopologyConfig, StabilityConfig, DynamicsConfig,
    ActiveInferenceConfig, DynamicTimeConfig, HysteresisConfig,
    EmbeddingConfig, FractalConfig, SingularityConfig,
)


def dict_to_physics_config(d: Dict[str, Any]) -> PhysicsConfig:
    """
    Convierte un dict anidado en un PhysicsConfig tipado.

    Soporta todos los sub-campos de PhysicsConfig. Los campos no presentes
    en el dict mantienen sus valores default del schema.
    Si `d` ya es PhysicsConfig, lo devuelve intacto.
    """
    if isinstance(d, PhysicsConfig):
        return d

    cfg = PhysicsConfig()
    _apply_dict_to_physics_config(cfg, d)
    return cfg


def apply_physics_overrides(cfg: PhysicsConfig, overrides: Dict[str, Any]) -> PhysicsConfig:
    """
    Aplica un dict de overrides sobre un PhysicsConfig EXISTENTE (in-place).

    A diferencia de dict_to_physics_config(), esta función NO parte de defaults
    sino que modifica solo los campos presentes en el dict, dejando el resto intacto.
    Es la función que usa ModelFactory cuando se combina preset + physics kwarg.

    Args:
        cfg:       PhysicsConfig existente (ej. resultado de get_preset())
        overrides: Dict anidado con los campos a sobreescribir

    Returns:
        El mismo cfg modificado in-place (también retornado para encadenamiento).
    """
    if not overrides:
        return cfg
    _apply_dict_to_physics_config(cfg, overrides)
    return cfg


def _apply_dict_to_physics_config(cfg: PhysicsConfig, d: Dict[str, Any]) -> None:
    """Función interna — aplica los campos del dict sobre cfg in-place."""

    # ── Topology ──────────────────────────────────────────────────────────────
    t_d = d.get('topology', d.get('topology_config', {}))
    if isinstance(t_d, dict) and t_d:
        _apply(cfg.topology, t_d, [
            'type', 'R', 'r', 'curvature',
            'riemannian_type', 'riemannian_rank', 'riemannian_class',
            'geometry_scope'
        ])
        if 'major_radius' in t_d: cfg.topology.R = t_d['major_radius']
        if 'minor_radius' in t_d: cfg.topology.r = t_d['minor_radius']

    # ── Stability ─────────────────────────────────────────────────────────────
    s_d = d.get('stability', d.get('stability_config', {}))
    if isinstance(s_d, dict) and s_d:
        _apply(cfg.stability, s_d, [
            'base_dt', 'adaptive', 'dt_min', 'dt_max',
            'enable_trace_normalization', 'wrap_x',
            'friction', 'velocity_friction_scale',
            'curvature_clamp', 'friction_mode',
            'integrator_type',
            # alias legacy
            'velocity_saturation',   # → ignorado, no existe en StabilityConfig
        ])
        # Alias de nombres legacy
        if 'toroidal_curvature_scale' in s_d:
            cfg.stability.curvature_clamp = s_d['toroidal_curvature_scale']

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dyn_d = d.get('dynamics', d.get('dynamics_config', {}))
    if isinstance(dyn_d, dict) and dyn_d:
        if 'type' in dyn_d:
            cfg.dynamics.type = dyn_d['type']

    # ── Active Inference ──────────────────────────────────────────────────────
    ai_d = d.get('active_inference', d.get('active_inference_config', {}))
    if isinstance(ai_d, dict) and ai_d:
        _apply(cfg.active_inference, ai_d, [
            'enabled', 'holographic_geometry',
            'thermodynamic_geometry', 'plasticity',
        ])
        # Dynamic time
        dt_d = ai_d.get('dynamic_time', {})
        if isinstance(dt_d, dict) and dt_d:
            _apply(cfg.active_inference.dynamic_time, dt_d, ['enabled', 'type'])
        # Reactive curvature — es un dict interno
        rc_d = ai_d.get('reactive_curvature', {})
        if isinstance(rc_d, dict) and rc_d:
            cfg.active_inference.reactive_curvature.update(rc_d)
        # Stochasticity — es un dict interno
        st_d = ai_d.get('stochasticity', {})
        if isinstance(st_d, dict) and st_d:
            cfg.active_inference.stochasticity.update(st_d)
        # Curiosity — es un dict interno
        cu_d = ai_d.get('curiosity', {})
        if isinstance(cu_d, dict) and cu_d:
            cfg.active_inference.curiosity.update(cu_d)
    # ── Hysteresis (pueden estar en raíz O dentro de active_inference) ────────
    hyst_src = d.get('hysteresis', ai_d.get('hysteresis', {}) if isinstance(ai_d, dict) else {})
    if isinstance(hyst_src, dict) and hyst_src:
        _apply(cfg.hysteresis, hyst_src, [
            'enabled', 'ghost_force', 'hyst_decay',
            'hyst_update_w', 'hyst_update_b',
            'hyst_readout_w', 'hyst_readout_b',
        ])

    # ── Singularities (pueden estar en raíz O dentro de active_inference) ─────
    sing_src = d.get('singularities', ai_d.get('singularities', {}) if isinstance(ai_d, dict) else {})
    if isinstance(sing_src, dict) and sing_src:
        _apply(cfg.singularities, sing_src, [
            'enabled', 'epsilon', 'strength', 'threshold'
        ])

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb_d = d.get('embedding', d.get('embedding_config', {}))
    if isinstance(emb_d, dict) and emb_d:
        _apply(cfg.embedding, emb_d, [
            'type', 'mode', 'coord_dim', 'impulse_scale', 'omega_0'
        ])

    # ── Readout ───────────────────────────────────────────────────────────────
    read_d = d.get('readout', d.get('readout_config', {}))
    if isinstance(read_d, dict) and read_d:
        _apply(cfg.readout, read_d, ['type'])

    # ── Mixture ───────────────────────────────────────────────────────────────
    mix_d = d.get('mixture', d.get('mixture_config', {}))
    if isinstance(mix_d, dict) and mix_d:
        _apply(cfg.mixture, mix_d, ['coupler_mode'])

    # ── Fractal ───────────────────────────────────────────────────────────────
    frac_d = d.get('fractal', {})
    if isinstance(frac_d, dict) and frac_d:
        _apply(cfg.fractal, frac_d, ['enabled', 'threshold', 'alpha'])

    # ── Top-level trajectory_mode ─────────────────────────────────────────────
    if 'trajectory_mode' in d:
        cfg.trajectory_mode = d['trajectory_mode']

    # ── Attention/mixer alias (legacy ECG configs) ────────────────────────────
    # 'attention': {'mixer_type': 'low_rank'} — se ignora acá, aplica en ManifoldConfig


def _apply(target, source: dict, keys: list) -> None:
    """Copia las claves presentes en source hacia target (setattr)."""
    for k in keys:
        if k in source:
            try:
                setattr(target, k, source[k])
            except AttributeError:
                pass  # clave no existe en el dataclass — ignorar silenciosamente

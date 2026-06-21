"""
config/loader.py — GFN V5
Conversion of configuration dicts to typed PhysicsConfig.
Support for nested overrides on existing configs.
"""
from typing import Dict, Any, Optional
from .schema import (
    PhysicsConfig, TopologyConfig, StabilityConfig, DynamicsConfig,
    ActiveInferenceConfig, DynamicTimeConfig, HysteresisConfig,
    EmbeddingConfig, FractalConfig, SingularityConfig,
)


def dict_to_physics_config(d: Dict[str, Any]) -> PhysicsConfig:
    """
    Converts a nested dict into a typed PhysicsConfig.

    Supports all PhysicsConfig sub-fields. Fields not present
    in the dict maintain their default values from the schema.
    If `d` is already PhysicsConfig, returns it unchanged.
    """
    if isinstance(d, PhysicsConfig):
        return d

    cfg = PhysicsConfig()
    _apply_dict_to_physics_config(cfg, d)
    return cfg


def apply_physics_overrides(cfg: PhysicsConfig, overrides: Dict[str, Any]) -> PhysicsConfig:
    """
    Applies a dict of overrides on an EXISTING PhysicsConfig (in-place).

    Unlike dict_to_physics_config(), this function does NOT start from defaults
    but only modifies the fields present in the dict, leaving the rest intact.
    This is the function that ModelFactory uses when combining preset + physics kwarg.

    Args:
        cfg:       Existing PhysicsConfig (e.g., result of get_preset())
        overrides: Nested dict with fields to overwrite

    Returns:
        The same cfg modified in-place (also returned for chaining).
    """
    if not overrides:
        return cfg
    _apply_dict_to_physics_config(cfg, overrides)
    return cfg


def _apply_dict_to_physics_config(cfg: PhysicsConfig, d: Dict[str, Any]) -> None:
    """Internal function — applies dict fields to cfg in-place."""

    # ── Topology ──────────────────────────────────────────────────────────────
    t_d = d.get('topology', d.get('topology_config', {}))
    if isinstance(t_d, dict) and t_d:
        _apply(cfg.topology, t_d, [
            'type', 'R', 'r', 'curvature',
            'riemannian_type', 'riemannian_rank', 'riemannian_class',
            'geometry_scope', 'learnable_R', 'learnable_r'
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
            # P2.3: velocity_saturation uses tanh-based differentiable clamping
            'velocity_saturation',
            # AdaptiveIntegrator knobs
            'adaptive_alpha',
            'base_solver',
            'toroidal_curvature_scale',
        ])

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
        # Reactive curvature — internal dict
        rc_d = ai_d.get('reactive_curvature', {})
        if isinstance(rc_d, dict) and rc_d:
            cfg.active_inference.reactive_curvature.update(rc_d)
        # Stochasticity — internal dict
        st_d = ai_d.get('stochasticity', {})
        if isinstance(st_d, dict) and st_d:
            cfg.active_inference.stochasticity.update(st_d)
        # Curiosity — internal dict
        cu_d = ai_d.get('curiosity', {})
        if isinstance(cu_d, dict) and cu_d:
            cfg.active_inference.curiosity.update(cu_d)
    # ── Hysteresis (can be at root OR inside active_inference) ────────
    hyst_src = d.get('hysteresis', ai_d.get('hysteresis', {}) if isinstance(ai_d, dict) else {})
    if isinstance(hyst_src, dict) and hyst_src:
        _apply(cfg.hysteresis, hyst_src, [
            'enabled', 'ghost_force', 'hyst_decay',
            'hyst_update_w', 'hyst_update_b',
            'hyst_readout_w', 'hyst_readout_b',
        ])

    # ── Singularities (can be at root OR inside active_inference) ─────
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
        _apply(cfg.readout, read_d, ['type', 'out_dim', 'hidden_dim'])

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
    # 'attention': {'mixer_type': 'low_rank'} — ignored here, applied in ManifoldConfig


def _apply(target, source: dict, keys: list) -> None:
    """Copies keys present in source to target (setattr)."""
    for k in keys:
        if k in source:
            try:
                setattr(target, k, source[k])
            except AttributeError:
                pass  # key does not exist in dataclass — silently ignore

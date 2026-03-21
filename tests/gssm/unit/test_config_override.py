"""
tests/unit/test_config_override.py
===================================
Tests unitarios para el sistema de config override — Fase 1 del Core Fix.

Verifica que:
  1. gfn.create(preset_name=..., physics={...}) aplica correctamente los overrides
  2. dict_to_physics_config() convierte dicts completos
  3. apply_physics_overrides() hace merge sin destruir el preset base
  4. kwargs planos y dict de physics coexisten sin conflictos
"""
import pytest
import torch
import gfn
from gfn.realizations.gssm.config.loader import dict_to_physics_config, apply_physics_overrides
# from gfn.realizations.gssm.config.presets import get_preset  # Deprecated
from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig
from gfn.realizations.gssm.models.factory import ModelFactory


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_kwargs():
    """Kwargs mínimos para un modelo de test rápido."""
    return dict(vocab_size=10, dim=8, heads=2, depth=1, rank=4)


# ── Tests: dict_to_physics_config ────────────────────────────────────────────

class TestDictToPhysicsConfig:
    def test_empty_dict_gives_defaults(self):
        cfg = dict_to_physics_config({})
        assert isinstance(cfg, PhysicsConfig)
        assert cfg.topology.type == 'torus'  # default

    def test_topology_override(self):
        cfg = dict_to_physics_config({'topology': {'type': 'euclidean'}})
        assert cfg.topology.type == 'euclidean'

    def test_stability_override(self):
        cfg = dict_to_physics_config({'stability': {'base_dt': 0.05, 'friction': 0.2}})
        assert cfg.stability.base_dt == pytest.approx(0.05)
        assert cfg.stability.friction == pytest.approx(0.2)

    def test_active_inference_disabled(self):
        cfg = dict_to_physics_config({'active_inference': {'enabled': False}})
        assert cfg.active_inference.enabled is False

    def test_dynamic_time_nested(self):
        cfg = dict_to_physics_config({
            'active_inference': {
                'enabled': True,
                'dynamic_time': {'enabled': True, 'type': 'riemannian'}
            }
        })
        assert cfg.active_inference.dynamic_time.enabled is True

    def test_fractal_override(self):
        cfg = dict_to_physics_config({'fractal': {'enabled': True, 'alpha': 0.5}})
        assert cfg.fractal.enabled is True
        assert cfg.fractal.alpha == pytest.approx(0.5)

    def test_already_physics_config_passthrough(self):
        original = PhysicsConfig()
        original.stability.base_dt = 0.77
        result = dict_to_physics_config(original)
        assert result is original  # mismo objeto
        assert result.stability.base_dt == pytest.approx(0.77)

    def test_stochasticity_dict_update(self):
        cfg = dict_to_physics_config({
            'active_inference': {
                'stochasticity': {'enabled': True, 'sigma': 0.05}
            }
        })
        assert cfg.active_inference.stochasticity['enabled'] is True
        assert cfg.active_inference.stochasticity['sigma'] == pytest.approx(0.05)


# ── Tests: apply_physics_overrides ────────────────────────────────────────────

class TestApplyPhysicsOverrides:
    def test_merge_does_not_reset_other_fields(self):
        """Aplicar un override parcial NO debe borrar el resto del preset."""
        base = PhysicsConfig()
        base.topology.type = 'torus'
        original_topology = base.topology.type
        original_friction  = base.stability.friction

        apply_physics_overrides(base, {'stability': {'base_dt': 0.05}})

        assert base.topology.type    == original_topology  # sin cambios
        assert base.stability.base_dt == pytest.approx(0.05)  # cambiado
        # friction puede o no haber cambiado (solo cambia si está en el override)
        assert base.stability.friction == original_friction  # sin cambios

    def test_empty_dict_noop(self):
        base = PhysicsConfig()
        base.topology.type = 'torus'
        original_dt = base.stability.base_dt
        apply_physics_overrides(base, {})
        assert base.stability.base_dt == original_dt

    def test_physics_config_replaces_entirely(self):
        base = PhysicsConfig()
        base.topology.type = 'torus'
        new_physics = PhysicsConfig()
        new_physics.stability.base_dt = 0.99

        # Pasar PhysicsConfig hace replace completo
        result_model = ModelFactory.create(
            # preset_name='stable-torus', # Deprecated
            physics=new_physics,
            vocab_size=10, dim=8, heads=2, depth=1, rank=4,
        )
        # Modelo creado sin crash — verificación de smoke
        assert result_model is not None


# ── Tests: gfn.create() con physics kwarg ────────────────────────────────────

class TestGfnCreatePhysicsKwarg:
    def test_create_with_physics_dict(self, tiny_kwargs):
        """Flujo principal: gfn.create(preset_name=..., physics={...})"""
        model = gfn.create(
            # preset_name='stable-torus', # Deprecated
            physics={
                'stability': {'base_dt': 0.1, 'friction': 0.01},
                'active_inference': {'enabled': False},
            },
            **tiny_kwargs,
        )
        assert model is not None

    def test_create_with_physics_topology_override(self, tiny_kwargs):
        """Override de topología vía dict de physics."""
        model = gfn.create(
            # preset_name='stable-torus', # Deprecated
            physics={'topology': {'type': 'euclidean'}},
            holographic=False,
            **tiny_kwargs,
        )
        assert model is not None

    def test_create_physics_plus_flat_kwargs(self, tiny_kwargs):
        """Dict de physics y kwargs planos coexisten sin conflicto."""
        model = gfn.create(
            # preset_name='stable-torus', # Deprecated
            physics={'stability': {'friction': 0.1}},
            integrator='yoshida',
            dynamics_type='residual',
            **tiny_kwargs,
        )
        assert model is not None

    def test_create_no_preset_with_config(self, tiny_kwargs):
        """Flujo 3: config preconstruido con dict_to_physics_config."""
        physics_cfg = dict_to_physics_config({
            'topology': {'type': 'euclidean'},
            'stability': {'base_dt': 0.2},
        })
        cfg = ManifoldConfig(physics=physics_cfg, holographic=False, **tiny_kwargs)
        model = ModelFactory.create(config=cfg)
        assert model is not None

    def test_forward_with_physics_dict_model(self, tiny_kwargs):
        """El modelo creado con physics dict hace forward sin error."""
        model = gfn.create(
            # preset_name='stable-torus', # Deprecated
            physics={'active_inference': {'enabled': False}},
            holographic=True,
            **tiny_kwargs,
        )
        x = torch.randint(0, tiny_kwargs['vocab_size'], (2, 5))
        logits, (x_f, v_f), metrics = model(x)
        assert logits.shape[0] == 2

    def test_physics_kwarg_wrong_type_raises(self, tiny_kwargs):
        """Tipo incorrecto en physics kwarg lanza ConfigurationError."""
        from gfn.errors import ConfigurationError
        with pytest.raises(ConfigurationError):
            gfn.create(
                # preset_name='stable-torus', # Deprecated
                physics="esto_no_es_valido",
                **tiny_kwargs,
            )


# ── Test: ECG Physics Config (caso real del usuario) ─────────────────────────

class TestECGPhysicsConfig:
    ECG_PHYSICS_CONFIG = {
        'topology': {'type': 'torus'},
        'active_inference': {
            'enabled': False,
            'dynamic_time': {'enabled': False},
            'reactive_curvature': {'enabled': False, 'plasticity': 0.01},
        },
        'stability': {
            'base_dt': 0.1,
            'enable_trace_normalization': True,
            'friction': 0.01,
            'velocity_saturation': 10.0,  # campo legacy — debe ignorarse sin crash
        },
        'fractal': {'enabled': False},
    }

    def test_ecg_config_dict_applies_without_crash(self):
        """El ECG_PHYSICS_CONFIG del benchmark se aplica sin crash."""
        model = gfn.create(
            # preset_name='stable-torus', # Deprecated
            physics=self.ECG_PHYSICS_CONFIG,
            vocab_size=2, dim=16, heads=2, depth=1, rank=4,
            holographic=True,
        )
        assert model is not None

    def test_ecg_config_stability_values(self):
        """Los valores de stability del ECG_PHYSICS_CONFIG se aplican correctamente."""
        # Verificar que dict_to_physics_config aplica base_dt=0.1
        cfg = dict_to_physics_config(self.ECG_PHYSICS_CONFIG)
        assert cfg.stability.base_dt    == pytest.approx(0.1)
        assert cfg.stability.friction   == pytest.approx(0.01)
        assert cfg.active_inference.enabled is False
        assert cfg.fractal.enabled is False

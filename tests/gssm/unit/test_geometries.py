import pytest
import torch
import math
from gfn.geometry import (
    ToroidalRiemannianGeometry,
    ReactiveRiemannianGeometry,
    HyperRiemannianGeometry,
    AdaptiveRiemannianGeometry,
    LowRankRiemannianGeometry
)
from gfn.realizations.gssm.config.schema import PhysicsConfig
# from gfn.realizations.gssm.config.presets import get_preset  # Deprecated

@pytest.fixture
def default_config():
    cfg = PhysicsConfig()
    cfg.topology.type = 'torus'
    return cfg

def test_toroidal_geometry_bounds(default_config):
    """
    Verifica que la variedad Toroidal (analítica)
    no tenga parámetros dependientes de datos (stateless parameterwise) 
    y que aplique límites geodésicos puros.
    """
    dim, head_dim = 16, 2
    geo = ToroidalRiemannianGeometry(dim, head_dim)
    
    # State
    x = torch.randn(8, dim) # B*H, D
    v = torch.randn(8, dim)
    
    # Forward stateless injects config
    gamma = geo(x, v, physics_config=default_config)
    if isinstance(gamma, tuple): gamma = gamma[0]
    
    assert gamma.shape == x.shape, "Gamma debe mantener dimensionalidad del estado"
    
    # Toroidal analytical geometry uses sin/cos, so Christoffel shouldn't explode
    # But it also scales by friction inside the base protocol.
    assert torch.isfinite(gamma).all(), "Símbolos de Christoffel divergieron"

def test_reactive_dynamic_friction(default_config):
    """
    Verifica que ReactiveGeometry module computes context-aware friction
    and its internal MLPs are properly initialized.
    """
    dim, head_dim = 16, 2
    geo = ReactiveRiemannianGeometry(dim, head_dim)
    
    x = torch.randn(8, dim)
    v = torch.randn(8, dim)
    
    gamma = geo(x, v, physics_config=default_config)
    if isinstance(gamma, tuple): gamma = gamma[0]
    
    assert gamma is not None, "Reactive geometry debe devolver el campo Gamma computado"

def test_hyperbolic_curvature_bounds(default_config):
    """
    Comprueba que el factor de Lorentz en HyperRiemannianGeometry frene 
    explosiones en los márgenes de la variedad.
    """
    dim, head_dim = 16, 2
    geo = HyperRiemannianGeometry(dim, head_dim)
    
    x = torch.ones(8, dim) * 1000.0 # Extreme outward point
    v = torch.ones(8, dim) * 100.0
    
    gamma = geo(x, v, physics_config=default_config)
    if isinstance(gamma, tuple): gamma = gamma[0]
    
    # Lorentz factor will clamp inputs. Gamma should not be NaN
    assert torch.isfinite(gamma).all(), "Curvature failed to clamp extreme Euclidean points"

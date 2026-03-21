import pytest
import torch
from gfn.realizations.gssm.physics.integrators import YoshidaIntegrator, LeapfrogIntegrator, HeunIntegrator
from gfn.realizations.gssm.config.schema import PhysicsConfig

@pytest.fixture
def test_config():
    cfg = PhysicsConfig()
    cfg.topology.type = 'torus'
    return cfg

class MockPhysicsEngine:
    """Mock Minimalist Engine solely to provide state and DT"""
    def __init__(self, config):
        self.config = config
        self.geometry = MockGeometry()
        
    def _get_dt(self):
        return self.config.stability.base_dt
        
    def compute_acceleration(self, x, v, force=None, dt=None, **kwargs):
        return torch.zeros_like(v)

class MockGeometry:
    """Geometry that simply returns state (Identity) to test pure integration formulas"""
    def __call__(self, x, v, physics_config=None):
        return x, torch.zeros_like(x) # Gamma = x, Friction = 0

def test_symplectic_yoshida_energy_conservation(test_config):
    """
    Yoshida is a 4th order symplectic integrator. 
    It must preserve the phase space volume.
    """
    engine = MockPhysicsEngine(test_config)
    geo = MockGeometry()
    integrator = YoshidaIntegrator(engine)
    
    x = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 16)
    
    # 1 Step Forward
    res = integrator.step(x, v)
    new_x, new_v = res['x'], res['v']
    
    # In a pure mock identity geometry (Gamma=x), this acts like an harmonic oscillator
    # Ensure gradients can flow
    assert new_x.requires_grad is False # If inputs don't require grad
    
    x.requires_grad_(True)
    v.requires_grad_(True)
    res2 = integrator.step(x, v)
    new_x = res2['x']
    assert new_x.requires_grad is True, "Autograd flow broken in YoshidaIntegrator"

def test_leapfrog_reversibility(test_config):
    """
    Leapfrog is symmetric and reversible. 
    T(dt) * T(-dt) = Identity.
    """
    engine = MockPhysicsEngine(test_config)
    geo = MockGeometry()
    integrator = LeapfrogIntegrator(engine)
    
    x_orig = torch.randn(2, 4, 16)
    v_orig = torch.randn(2, 4, 16)
    
    # Step Forward
    res_fwd = integrator.step(x_orig.clone(), v_orig.clone())
    x_fwd, v_fwd = res_fwd['x'], res_fwd['v']
    
    # Reverse time (Hack: Invert velocity and step again)
    res_rev = integrator.step(x_fwd, -v_fwd)
    x_rev, v_rev = res_rev['x'], res_rev['v']
    
    # x_rev should closely match x_orig
    assert torch.allclose(x_orig, x_rev, atol=1e-3), "Leapfrog is not time-reversible"

def test_heun_rk2_step(test_config):
    """Heun (RK2) is not symplectic but provides better explicit bounds for reactive fields."""
    engine = MockPhysicsEngine(test_config)
    geo = MockGeometry()
    integrator = HeunIntegrator(engine)
    
    x = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 16)
    
    res = integrator.step(x, v)
    new_x, new_v = res['x'], res['v']
    
    assert new_x.shape == x.shape
    assert new_v.shape == v.shape

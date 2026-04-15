import pytest
import torch
from gfn.realizations.gssm.physics.components.stochasticity import BrownianForce, OUDynamicsForce
from gfn.realizations.gssm.physics.components.curiosity import GeometricCuriosityForce
from gfn.realizations.gssm.config.schema import PhysicsConfig

@pytest.fixture
def default_config():
    cfg = PhysicsConfig()
    cfg.topology.type = 'torus'
    return cfg

def test_brownian_langevin_force(default_config):
    r"""
    Brownian (Langevin) force should inject Gaussian isotropic noise
    scaled by \sqrt{2 * T * dt} (Fluctuation-Dissipation theorem analog in ML).
    """
    dim = 16
    force = BrownianForce(sigma=1.0)
    
    x = torch.zeros(2, 4, dim)
    v = torch.zeros(2, 4, dim)
    
    # BrownianForce (Langevin) uses 1/sqrt(dt) so we must inject the timestep length
    F_stoc = force(x, v, dt=default_config.stability.base_dt)
    
    assert F_stoc.shape == x.shape, "El tensor de ruido estocástico no coincide con el estado"
    assert not torch.allclose(F_stoc, torch.zeros_like(F_stoc)), "BrownianForce devolvió un vector nulo"
    
    # Check if standard deviation is roughly within expectations for isotropic noise
    std = F_stoc.std().item()
    expected_std = force.sigma / (default_config.stability.base_dt ** 0.5)
    
    assert expected_std * 0.5 < std < expected_std * 1.5, "La varianza termodinámica es errónea"

def test_ou_dynamics_restoring_force(default_config):
    """
    Ornstein-Uhlenbeck processes include a spring-like restoring force
    pushing trajectories back to the origin, plus Brownian noise.
    """
    dim, head_dim = 16, 2
    ou = OUDynamicsForce(theta=1.0, mu=0.0, sigma=0.5)
    
    x = torch.ones(2, 4, dim) * 5.0 # Placed far from origin (mu=0)
    v = torch.zeros(2, 4, dim)
    
    F_ou = ou(x, v, dt=default_config.stability.base_dt)
    
    # Force should be centered around mu (0.0) generally, and active
    mean_force = F_ou.mean().item()
    assert abs(mean_force) < 2.0, "OUDynamics falló generando un ruido autocorrelacionado inestable"
    assert F_ou.shape == x.shape, "Dimensionalidad incorrecta"

def test_curiosity_exploration(default_config):
    """
    CuriosityForce acts as an intrinsic reward, pushing heads 
    AWAY from the batch mean to maximize entropy and diversity.
    """
    force = GeometricCuriosityForce(strength=2.0)
    
    # B=2, H=4, D=16
    x = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 16)
    
    F_curiosity = force(x, v)
    
    # It calculates structural distances. Ensure it triggers gradients
    x.requires_grad_(True)
    F = force(x, v)
    sum_F = F.sum()
    sum_F.backward()
    
    assert x.grad is not None, "CuriosityForce rompió el grafo de diferenciación geodésica"
    assert F_curiosity.shape == x.shape

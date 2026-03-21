import pytest
import torch
import torch.nn as nn
from gfn.realizations.gssm.models.factory import ModelFactory
from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig

@pytest.fixture
def stable_torus_config():
    """Provides a Toroidal Configuration that guarantees Active Inference and Friction."""
    phys_cfg = PhysicsConfig()
    phys_cfg.topology.type = 'torus'
    
    # Overriding specific values for a lightweight mock test
    config = ManifoldConfig(
        vocab_size=100,
        dim=32,
        heads=4,
        depth=2,
        rank=16,
        physics=phys_cfg
    )
    
    # Stability
    config.physics.stability.adaptive = True
    config.physics.stability.base_dt = 0.1
    
    # Disable Holographic mode so ModelFactory forces CategoricalReadout
    config.physics.active_inference.holographic_geometry = False
    
    # Enable Stochastic Forces (Langevin / OUDynamics) to ensure they hook correctly
    config.physics.active_inference.stochasticity['enabled'] = True
    config.physics.active_inference.stochasticity['type'] = 'brownian'
    
    return config

def test_manifold_pipeline_forward_backward(stable_torus_config):
    """
    Integration Test: Mocks a full Transformer-like sequence block
    and ensures that the backward pass successfully propagates through
    the symplectic integrators, geometry, and structural stochasticity.
    """
    # 1. Instantiate End-to-End Model via Factory
    model = ModelFactory.create(stable_torus_config)
    
    # Simulated Batch of tokens (e.g. Language Modeling or XOR task)
    B, Seq = 2, 8
    x = torch.randint(0, stable_torus_config.vocab_size, (B, Seq))
    
    # 2. Forward Pass
    logits, state, telemetry = model(x)
    
    # Ensure logits are properly shaped for CrossEntropy [B, Seq, Vocab]
    assert logits.shape == (B, Seq, stable_torus_config.vocab_size), "Readout falló al restaurar dimensionalidad semántica"
    
    # Ensure telemetry hooks captured internal trajectory physics
    assert 'v_seq' in telemetry, "El Pipeline omitió métricas de telemetría de la velocidad"
    assert 'x_seq' in telemetry, "El Pipeline omitió métricas del espacio latente"
    
    # 3. Backward Pass (End-to-end topological gradients)
    # Using a dummy target to compute a CrossEntropy loss
    target = torch.randint(0, stable_torus_config.vocab_size, (B, Seq))
    criterion = nn.CrossEntropyLoss()
    
    # Flatten for CrossEntropy
    loss = criterion(logits.view(-1, stable_torus_config.vocab_size), target.view(-1))
    loss.backward()
    
    # Ensure Model Parameters received gradients (especially those deep in the geometry or attention)
    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_found = True
            break
            
    assert grad_found, "Graphic detatched: No se encontraron gradientes tras la inyección integradora. La topología aisló el backpropagation."

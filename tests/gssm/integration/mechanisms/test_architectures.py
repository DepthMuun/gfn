import pytest
import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from gfn.realizations.gssm.models.factory import ModelFactory
from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig
from tests.utils.telemetry import TelemetryAnalyzer

# Constants for comparisons
DYNAMICS_MODES = ['direct', 'residual']
MIXER_MODES = ['low_rank', 'attention']

@pytest.fixture(scope="module")
def mechanism_telemetry():
    return TelemetryAnalyzer(output_dir="tests/results/mechanisms")

@pytest.mark.parametrize("dyn_type", DYNAMICS_MODES)
@pytest.mark.parametrize("mix_type", MIXER_MODES)
def test_architecture_mechanisms(dyn_type, mix_type, mechanism_telemetry):
    """
    Compares convergence of different architectural mechanisms 
    using a simplified XOR task.
    """
    # 1. Setup Config
    phys_cfg = PhysicsConfig()
    phys_cfg.topology.type = 'torus'
    config = ManifoldConfig(
        vocab_size=2,
        dim=32,
        heads=2,
        depth=2,
        rank=8,
        physics=phys_cfg,
        dynamics_type=dyn_type,
        mixer_type=mix_type,
        holographic=True
    )
    
    # Ensure impulse_scale is sufficient for the task
    config.impulse_scale = 80.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create(config).to(device)
    
    # 2. Mini-XOR Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Using the standard ToroidalLoss since we're in holographic mode
    from gfn.losses.toroidal import ToroidalLoss
    criterion = ToroidalLoss()
    
    B, L = 16, 12
    history = []
    
    print(f"\nEvaluating: Dynamics={dyn_type} | Mixer={mix_type}")
    
    model.train()
    for step in range(50):
        # Generate Parity Data
        x = torch.randint(0, 2, (B, L), device=device)
        y_int = torch.cumsum(x, dim=1) % 2
        y_angle = (y_int.float() * 2.0 - 1.0) * (3.14159 * 0.5)
        
        optimizer.zero_grad()
        logits, state, telemetry = model(x)
        
        # In holographic mode, logits are the raw (x, v) states
        y_exp = y_angle.view(y_angle.shape + (1,) * (logits.ndim - y_angle.ndim))
        y_exp = y_exp.expand_as(logits)
        
        loss = criterion(logits, y_exp)
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
    # 3. Validation and Metrics
    final_loss = sum(history[-5:]) / 5.0
    print(f"Final Loss (avg last 5): {final_loss:.4f}")
    
    # 4. Telemetry Visualization
    # Save the loss curve for comparison
    plot_prefix = f"mech_{dyn_type}_{mix_type}"
    mechanism_telemetry.save_metric(f"{plot_prefix}_final_loss", final_loss)
    
    # Plot last trajectory
    x_seq = telemetry.get('x_seq')
    v_seq = telemetry.get('v_seq')
    if x_seq is not None and v_seq is not None:
        mechanism_telemetry.plot_trajectories(
            x_seq, v_seq, 
            title=f"Dyn: {dyn_type} | Mix: {mix_type}", 
            prefix=plot_prefix
        )
        
    assert final_loss < 20.0 # Basic sanity check that it didn't explode

import pytest
import torch
import torch.nn as nn
from gfn.realizations.gssm.models.factory import ModelFactory
from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig
from gfn.constants import (
    TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
)
from tests.utils.telemetry import TelemetryAnalyzer

# Define the exhaustive grid of components we want to guarantee compatibility for
TOPOLOGIES = [TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN]
INTEGRATORS = ['leapfrog', 'yoshida', 'verlet', 'rk4']
STOCHASTICITIES = [False, True]
MIXERS = ['low_rank', 'attention']

@pytest.fixture(scope="module")
def telemetry():
    return TelemetryAnalyzer(output_dir="tests/results/combinations")

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("stochasticity", STOCHASTICITIES)
@pytest.mark.parametrize("mixer", MIXERS)
def test_manifold_combinations(topology, integrator, stochasticity, mixer, telemetry):
    """
    Exhaustive Combinatorial Test Suite.
    Ensures every logical intersection of GFN V5 connects and propagates gradients.
    """
    phys_cfg = PhysicsConfig()
    phys_cfg.topology.type = topology
    
    # Overriding the core components to test the specific combinatorial branch
    phys_cfg.topology.type = topology
    phys_cfg.active_inference.stochasticity['enabled'] = stochasticity
    if stochasticity:
        phys_cfg.active_inference.stochasticity['type'] = 'brownian'
        
    config = ManifoldConfig(
        vocab_size=50,
        dim=32,
        heads=4,
        depth=1,  # Keep depth = 1 to speed up combinatorial permutations
        rank=8,
        physics=phys_cfg
    )
    
    # Override global assembler settings
    config.integrator = integrator
    config.mixer_type = mixer
    config.physics.active_inference.holographic_geometry = False
    
    # 1. Instantiate
    try:
        model = ModelFactory.create(config)
    except Exception as e:
        pytest.fail(f"Falla de instanciacion con {topology}/{integrator}/{mixer}: {str(e)}")
        
    B, Seq = 2, 8
    x = torch.randint(0, config.vocab_size, (B, Seq))
    target = torch.randint(0, config.vocab_size, (B, Seq))
    criterion = nn.CrossEntropyLoss()
    
    # 2. Forward
    try:
        logits, state, metrics = model(x)
    except Exception as e:
        pytest.fail(f"Falla de Forward con {topology}/{integrator}/{mixer}: {str(e)}")
        
    # 3. Backward
    try:
        loss = criterion(logits.view(-1, config.vocab_size), target.view(-1))
        loss.backward()
    except Exception as e:
        pytest.fail(f"Falla de Backward con {topology}/{integrator}/{mixer}: {str(e)}")
        
    # 4. Extract Telemetry Graph (only for a subset to prevent exploding PNGs)
    x_seq, v_seq = metrics.get('x_seq'), metrics.get('v_seq')
    # Generate plot only if Leapfrog and Torus (to save test time and avoid ~32 PNGs).
    if x_seq is not None and v_seq is not None and integrator == 'leapfrog' and topology == TOPOLOGY_TORUS:
        plot_name = f"combo_{topology}_{integrator}_{mixer}_stoch_{stochasticity}"
        telemetry.plot_trajectories(x_seq, v_seq, title=f"Topologia: {topology.capitalize()} | Mix: {mixer}", prefix=plot_name)
        
    # Validation assertion that everything worked
    assert logits.shape == (B, Seq, config.vocab_size)
    assert loss.item() > 0

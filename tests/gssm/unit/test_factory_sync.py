import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

def test_sync():
    # 1. Test flat kwarg sync
    model = gfn.create(vocab_size=2, integrator='yoshida', impulse_scale=80.0)
    
    # Check if integrator is Yoshida
    integrator = model.layers[0].integrator
    print(f"Integrator type: {type(integrator).__name__}")
    assert "Yoshida" in type(integrator).__name__
    
    # Check if embedding impulse_scale is 80.0
    print(f"Embedding impulse_scale: {model.embedding.impulse_scale}")
    assert model.embedding.impulse_scale == 80.0
    
    # 2. Test physics dict sync
    physics_config = {
        'embedding': {'impulse_scale': 123.0},
        'stability': {'integrator_type': 'verlet'}
    }
    model2 = gfn.create(vocab_size=2, physics=physics_config)
    
    print(f"Model2 Integrator: {type(model2.layers[0].integrator).__name__}")
    assert "Verlet" in type(model2.layers[0].integrator).__name__
    print(f"Model2 Embedding impulse_scale: {model2.embedding.impulse_scale}")
    assert model2.embedding.impulse_scale == 123.0
    
    print("Verification SUCCESSFUL!")

if __name__ == "__main__":
    test_sync()

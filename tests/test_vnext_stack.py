
import torch
from gfn import Manifold

def test_vnext_full_stack():
    print("Starting vNext Full Stack Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 128
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 50.0, 'threshold': 0.8},
            'hysteresis': {'enabled': True, 'strength': 1.0}
        },
        'hierarchical_curvature': {'enabled': True, 'ranks': [8, 16, 32]},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.05}
    }
    
    # Initialize model
    model = Manifold(vocab_size=2, dim=dim, depth=2, heads=4, 
                    integrator_type='leapfrog', physics_config=physics_config).to(device)
    
    # Mock input
    x = torch.randint(0, 2, (4, 10)).to(device)
    
    print("Running forward pass...")
    try:
        output = model(x)
        print(f"Forward pass SUCCESS. Output shape: {output[0].shape}")
        
        loss = output[0].sum()
        print("Running backward pass...")
        loss.backward()
        print("Backward pass SUCCESS.")
        
        # Check for NaNs
        if torch.isnan(output[0]).any():
            print("FAILED: NaNs detected in output")
        else:
            print("SUCCESS: No NaNs detected")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vnext_full_stack()

"""
Quick test to verify CUDA fusion is active.
Run this before the benchmark to confirm optimizations are working.
"""

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.core.manifold import Manifold

def test_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("  CUDA FUSION VERIFICATION TEST")
    print("="*60 + "\n")
    
    # Test configuration matching benchmark
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.4}
    }
    
    print("Testing Leapfrog integrator...")
    model_leapfrog = Manifold(
        vocab_size=2, 
        dim=128, 
        depth=6, 
        heads=4, 
        integrator_type='leapfrog', 
        physics_config=physics_config, 
        impulse_scale=80.0, 
        holographic=True
    ).to(device)
    
    # Check fusion capability
    can_fuse = model_leapfrog.fusion_manager.can_fuse(collect_christ=False)
    print(f"\nâœ“ Fusion available: {can_fuse}")
    
    if can_fuse:
        params = model_leapfrog.fusion_manager.prepare_parameters()
        print(f"âœ“ Parameters prepared: {params is not None}")
        
        if params:
            print(f"  - Topology: {'Torus' if params['topology_id'] == 1 else 'Euclidean'}")
            print(f"  - Layers: {model_leapfrog.depth}")
            print(f"  - Heads: {model_leapfrog.heads}")
            print(f"  - Plasticity: {params['plasticity']}")
    
    # Run a forward pass to trigger fusion
    print("\nRunning forward pass...")
    x = torch.randint(0, 2, (4, 20), device=device)
    
    with torch.no_grad():
        output = model_leapfrog(x, collect_christ=False)
    
    print(f"âœ“ Forward pass completed")
    print(f"  - Output shape: {output[0].shape}")
    
    # Test Heun for comparison
    print("\n" + "-"*60)
    print("Testing Heun integrator for comparison...")
    model_heun = Manifold(
        vocab_size=2, 
        dim=128, 
        depth=6, 
        heads=4, 
        integrator_type='heun', 
        physics_config=physics_config, 
        impulse_scale=80.0, 
        holographic=True
    ).to(device)
    
    can_fuse_heun = model_heun.fusion_manager.can_fuse(collect_christ=False)
    print(f"âœ“ Heun fusion available: {can_fuse_heun}")
    
    with torch.no_grad():
        output_heun = model_heun(x, collect_christ=False)
    
    print(f"âœ“ Heun forward pass completed")
    
    print("\n" + "="*60)
    print("  VERIFICATION COMPLETE")
    print("="*60)
    
    if can_fuse:
        print("\nâœ… CUDA FUSION IS ACTIVE FOR LEAPFROG!")
        print("   Benchmark will use optimized CUDA kernels.")
    else:
        print("\nâš ï¸  CUDA FUSION NOT AVAILABLE")
        print("   Check CUDA compilation and device availability.")
    
    print()

if __name__ == "__main__":
    test_fusion()


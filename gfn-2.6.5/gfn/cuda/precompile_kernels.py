"""
Precompile CUDA Kernels
========================
This script forces PyTorch to compile and cache the CUDA kernels.
Run this ONCE with MSVC in PATH, then the kernels will be cached.
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("="*70)
print("  PRECOMPILING CUDA KERNELS")
print("="*70)

# Force compilation by importing
try:
    from gfn.cuda.ops import christoffel_fused, leapfrog_fused, CUDA_AVAILABLE
    
    if CUDA_AVAILABLE:
        print("\n✅ CUDA kernels loaded and cached!")
        print("   Kernels are now compiled and will be reused automatically.")
        
        # Test them to ensure they work
        device = torch.device('cuda')
        batch, dim, rank = 2, 32, 8
        
        v = torch.randn(batch, dim, device=device)
        U = torch.randn(dim, rank, device=device)
        W = torch.randn(dim, rank, device=device)
        
        gamma = christoffel_fused(v, U, W)
        print(f"\n🧪 Test passed: Christoffel output shape = {gamma.shape}")
        
        x = torch.randn(batch, dim, device=device)
        f = torch.randn(batch, dim, device=device)
        x_new, v_new = leapfrog_fused(x, v, f, U, W, 0.1, 1.0)
        print(f"🧪 Test passed: Leapfrog output shapes = {x_new.shape}, {v_new.shape}")
        
        print("\n" + "="*70)
        print("  SUCCESS: Kernels compiled and cached!")
        print("  You can now use them without recompiling.")
        print("="*70)
        
    else:
        print("\n❌ CUDA kernels failed to compile")
        print("   Make sure MSVC is in PATH before running this script")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

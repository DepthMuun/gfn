"""
Quick test to verify CUDA kernels are working.
"""

import torch
import sys
import os

print("=" * 70)
print("GFN CUDA Kernel Test")
print("=" * 70)

# Test 1: Import CUDA module
print("\n[Test 1] Importing CUDA module...")
try:
    import gfn_cuda
    print(f"✓ gfn_cuda module imported successfully!")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test Christoffel kernel
print("\n[Test 2] Testing Christoffel kernel...")
try:
    batch_size = 4
    dim = 64
    rank = 16
    
    # Create test tensors
    v = torch.randn(batch_size, dim, device='cuda')
    U = torch.randn(dim, rank, device='cuda')
    W = torch.randn(dim, rank, device='cuda')
    x = torch.randn(batch_size, dim, device='cuda')
    V_w = torch.empty(0, device='cuda')
    
    # Call kernel directly
    gamma = gfn_cuda.lowrank_christoffel_fused(
        v, U, W, x, V_w,
        0.1,  # plasticity
        0.8,  # sing_thresh
        10.0, # sing_strength
        0,    # topology (Euclidean)
        2.0,  # R
        1.0   # r
    )
    
    print(f"✓ Christoffel output shape: {gamma.shape}")
    print(f"✓ Output range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    print(f"✓ Output mean: {gamma.mean():.4f}")
        
except Exception as e:
    print(f"✗ Christoffel test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test Leapfrog integrator
print("\n[Test 3] Testing Leapfrog integrator...")
try:
    batch_size = 4
    dim = 64
    rank = 16
    
    # Create test tensors
    x = torch.randn(batch_size, dim, device='cuda')
    v = torch.randn(batch_size, dim, device='cuda')
    force = torch.randn(batch_size, dim, device='cuda')
    U = torch.randn(dim, rank, device='cuda')
    W = torch.randn(dim, rank, device='cuda')
    Wf = torch.empty(0, device='cuda')
    bf = torch.empty(0, device='cuda')
    
    # Call kernel directly
    x_out, v_out = gfn_cuda.leapfrog_fused(
        x, v, force, U, W,
        0.1,   # dt
        1.0,   # dt_scale
        10,    # steps
        0,     # topology
        Wf, bf,
        0.0,   # plasticity
        2.0,   # R
        1.0    # r
    )
    
    print(f"✓ Leapfrog x_out shape: {x_out.shape}")
    print(f"✓ Leapfrog v_out shape: {v_out.shape}")
    print(f"✓ Position change: {(x_out - x).abs().mean():.4f}")
    print(f"✓ Velocity change: {(v_out - v).abs().mean():.4f}")
        
except Exception as e:
    print(f"✗ Leapfrog test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test Heun integrator
print("\n[Test 4] Testing Heun integrator...")
try:
    batch_size = 4
    dim = 64
    rank = 16
    
    # Create test tensors
    x = torch.randn(batch_size, dim, device='cuda')
    v = torch.randn(batch_size, dim, device='cuda')
    force = torch.randn(batch_size, dim, device='cuda')
    U = torch.randn(dim, rank, device='cuda')
    W = torch.randn(dim, rank, device='cuda')
    
    # Call kernel directly
    x_out, v_out = gfn_cuda.heun_fused(
        x, v, force, U, W,
        0.1,   # dt
        1.0,   # dt_scale
        10,    # steps
        0,     # topology
        2.0,   # R
        1.0    # r
    )
    
    print(f"✓ Heun x_out shape: {x_out.shape}")
    print(f"✓ Heun v_out shape: {v_out.shape}")
    print(f"✓ Position change: {(x_out - x).abs().mean():.4f}")
    print(f"✓ Velocity change: {(v_out - v).abs().mean():.4f}")
        
except Exception as e:
    print(f"✗ Heun test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)


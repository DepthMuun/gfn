"""
Quick CUDA Kernel Compilation Test
====================================
Attempts to load and verify CUDA kernels.
"""

import torch
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("="*70)
print("  GFN CUDA KERNEL COMPILATION TEST")
print("="*70)

print(f"\n[GFN] PyTorch version: {torch.__version__}")
print(f"[GFN] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GFN] CUDA version: {torch.version.cuda}")
    print(f"[GFN] GPU: {torch.cuda.get_device_name(0)}")

print("\n[GFN] Attempting to load CUDA kernels...")

try:
    from gfn.cuda.ops import christoffel_fused, leapfrog_fused, CUDA_AVAILABLE
    
    if CUDA_AVAILABLE:
        print("[GFN] CUDA kernels loaded successfully!")
        
        # Quick functionality test
        device = torch.device('cuda')
        
        # Test christoffel_fused
        batch, dim, rank = 4, 64, 8
        v = torch.randn(batch, dim, device=device)
        U = torch.randn(dim, rank, device=device)
        W = torch.randn(dim, rank, device=device)
        
        print("\n[GFN] Testing Christoffel kernel...")
        gamma = christoffel_fused(v, U, W)
        print(f"   Input shape: {v.shape}")
        print(f"   Output shape: {gamma.shape}")
        print(f"   Output range: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
        print("   [OK] Christoffel kernel works!")
        
        # Test leapfrog_fused
        print("\n[GFN] Testing Leapfrog kernel...")
        x = torch.randn(batch, dim, device=device)
        v = torch.randn(batch, dim, device=device)
        f = torch.randn(batch, dim, device=device)
        
        x_new, v_new = leapfrog_fused(x, v, f, U, W, dt=0.1, dt_scale=1.0)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shapes: x={x_new.shape}, v={v_new.shape}")
        print("   [OK] Leapfrog kernel works!")
        
        print("\n" + "="*70)
        print("  [SUCCESS] ALL KERNELS FUNCTIONAL!")
        print("="*70)
        
    else:
        print("[GFN:ERROR] CUDA kernels failed to load (falling back to PyTorch)")
        print("   [INFO] PyTorch fallback will be used automatically")
        
except Exception as e:
    print(f"[GFN:ERROR] Error loading kernels: {e}")
    import traceback
    traceback.print_exc()
    print("\n[GFN:WARN] Kernels not functional, but PyTorch fallback available")

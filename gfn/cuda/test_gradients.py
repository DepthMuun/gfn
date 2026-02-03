"""
Verify Analytical CUDA Gradients using torch.autograd.gradcheck
"""

import torch
import os
import sys

# Add path for gfn package
cuda_dir = os.path.dirname(os.path.abspath(__file__))
gfn_dir = os.path.dirname(cuda_dir)
manifold_dir = os.path.dirname(gfn_dir)
sys.path.insert(0, manifold_dir)

try:
    import gfn_cuda
    from gfn.cuda import ops
    print("✓ Modules loaded")
except ImportError as e:
    print(f"✗ Load failed: {e}")
    sys.exit(1)

def test_christoffel_gradients():
    print("\n" + "="*80)
    print("Verifying Christoffel Analytical Gradients")
    print("="*80)

    device = 'cuda'
    dtype = torch.float64
    B, D, R = 4, 16, 4

    # Inputs requiring gradients
    # Inputs requiring gradients
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    
    # Static inputs (no grad)
    V_w = torch.randn(D, device=device, dtype=dtype)
    plasticity = 0.5
    sing_thresh = 0.5
    sing_strength = 2.0
    topology = 1 # Toroidal (test cos(x) gradients)
    Radi = 2.0
    radi = 1.0

    # Wrapper for gradcheck
    def christoffel_func(v_in, U_in, W_in, x_in):
        return ops.christoffel_fused(
            v_in, U_in, W_in, x_in, V_w,
            plasticity, sing_thresh, sing_strength,
            topology, Radi, radi
        )

    print(f"Running gradcheck (Double Precision, Singularities Active)...")
    try:
        # eps is the step for numerical diff
        # atol/rtol are for comparison
        torch.autograd.gradcheck(
            christoffel_func, (v, U, W, x),
            eps=1e-6, atol=1e-5, rtol=1e-4
        )
        print("✓ PASS: Analytical gradients match numerical gradients including singularities!")
    except Exception as e:
        print(f"✗ FAIL: Gradient mismatch!")
        print(e)

if __name__ == "__main__":
    test_christoffel_gradients()

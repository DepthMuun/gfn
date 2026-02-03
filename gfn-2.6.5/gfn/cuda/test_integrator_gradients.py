import torch
import torch.nn as nn
import os
import sys

# Add local directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import autograd

def test_leapfrog_gradients():
    print("\n" + "="*80)
    print("Verifying Leapfrog Adjoint Gradients")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.float64 # Use double precision for gradcheck
    
    B, D, R = 4, 8, 4
    steps = 5
    dt = 0.1
    
    # Inputs requiring gradients
    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    force = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    
    # Optional parameters for friction
    Wf = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True)
    bf = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    
    plasticity = 0.5
    topology = 0 # Euclidean
    Radi = 2.0
    radi = 1.0

    def integrator_func(x_in, v_in, force_in, U_in, W_in, Wf_in, bf_in):
        x_out, v_out = autograd.leapfrog_fused_autograd(
            x_in, v_in, force_in, U_in, W_in,
            dt, 1.0, steps, topology,
            Wf_in, bf_in, plasticity, Radi, radi
        )
        # Use a scalar loss from both outputs
        return (x_out**2).sum() + (v_out**2).sum()

    print(f"Running gradcheck (Double Precision, Steps={steps})...")
    try:
        torch.autograd.gradcheck(
            integrator_func, (x, v, force, U, W, Wf, bf),
            eps=1e-6, atol=1e-4, rtol=1e-3
        )
        print("✓ PASS: Leapfrog Adjoint gradients match numerical gradients!")
    except Exception as e:
        print(f"✗ FAIL: Leapfrog Gradient mismatch!")
        print(e)

def test_heun_gradients():
    print("\n" + "="*80)
    print("Verifying Heun (RK2) Adjoint Gradients")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.float64
    
    B, D, R = 4, 8, 4
    steps = 3
    dt = 0.1
    
    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    force = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    
    topology = 0
    Radi = 2.0
    radi = 1.0

    def integrator_func(x_in, v_in, force_in, U_in, W_in):
        x_out, v_out = autograd.heun_fused_autograd(
            x_in, v_in, force_in, U_in, W_in,
            dt, 1.0, steps, topology, Radi, radi
        )
        return (x_out**2).sum() + (v_out**2).sum()

    print(f"Running gradcheck (Double Precision, Steps={steps})...")
    try:
        torch.autograd.gradcheck(
            integrator_func, (x, v, force, U, W),
            eps=1e-6, atol=1e-4, rtol=1e-3
        )
        print("✓ PASS: Heun Adjoint gradients match numerical gradients!")
    except Exception as e:
        print(f"✗ FAIL: Heun Gradient mismatch!")
        print(e)

def test_recurrent_manifold_gradients():
    print("\n" + "="*80)
    print("Verifying Recurrent Manifold Gradients (Python Fallback)")
    print("="*80)

    device = torch.device("cuda")
    dtype = torch.float64

    B, D, heads, layers, rank, T = 2, 8, 2, 1, 2, 3
    head_dim = D // heads
    dt = 0.05

    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    f = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    U_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)
    W_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)

    dt_scales = torch.ones(layers, heads, device=device, dtype=dtype)
    forget_rates = torch.ones(heads, device=device, dtype=dtype)

    mix_x = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    mix_v = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    mix_x_bias = torch.zeros(layers, D, device=device, dtype=dtype, requires_grad=True)
    mix_v_bias = torch.zeros(layers, D, device=device, dtype=dtype, requires_grad=True)

    empty = torch.empty(0, device=device, dtype=dtype)

    def fused_loss(x_in, v_in, f_in, U_in, W_in, mx_in, mv_in, bx_in, bv_in):
        x_out, v_out, x_seq, _ = autograd.recurrent_manifold_fused_autograd(
            x_in, v_in, f_in,
            U_in, W_in,
            dt, dt_scales, forget_rates, heads,
            0.0, 1.0, 1.0,
            mx_in, mv_in,
            empty, empty, empty, empty, empty,
            0, 2.0, 1.0,
            mix_x_bias=bx_in, mix_v_bias=bv_in,
            norm_x_weight=empty, norm_x_bias=empty,
            norm_v_weight=empty, norm_v_bias=empty,
            gate_W1=empty, gate_b1=empty, gate_W2=empty, gate_b2=empty,
            integrator_type=1
        )
        return (x_out**2).sum() + (v_out**2).sum() + (x_seq**2).sum()

    print("Running gradcheck (Double Precision)...")
    try:
        torch.autograd.gradcheck(
            fused_loss, (x, v, f, U_stack, W_stack, mix_x, mix_v, mix_x_bias, mix_v_bias),
            eps=1e-6, atol=5e-4, rtol=5e-3
        )
        print("✓ PASS: Recurrent manifold gradients match numerical gradients!")
    except Exception as e:
        print("✗ FAIL: Recurrent manifold gradient mismatch!")
        print(e)

if __name__ == "__main__":
    test_leapfrog_gradients()
    test_heun_gradients()
    test_recurrent_manifold_gradients()

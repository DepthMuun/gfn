"""
Rigorous CUDA vs Python Accuracy Test
Verifies zero discrepancies between implementations.
"""

import torch
import sys
import os
import numpy as np

# Add parent directories to path
cuda_dir = os.path.dirname(os.path.abspath(__file__))
gfn_dir = os.path.dirname(cuda_dir)  # gfn directory
manifold_dir = os.path.dirname(gfn_dir)  # manifold directory
sys.path.insert(0, manifold_dir)

print("=" * 80)
print("GFN CUDA vs Python Accuracy Verification")
print("=" * 80)

# Import both implementations
try:
    import gfn_cuda
    from gfn.geometry.lowrank import LowRankChristoffel
    from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator
    from gfn.integrators.runge_kutta.heun import HeunIntegrator
    print("✓ Modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test configuration
BATCH_SIZE = 8
DIM = 32
RANK = 8
DEVICE = 'cuda'
RTOL = 1e-12  # Tighter tolerance for double precision
ATOL = 1e-13

print(f"\nTest Configuration:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Dimension: {DIM}")
print(f"  Rank: {RANK}")
print(f"  Tolerances: rtol={RTOL}, atol={ATOL}")

# ============================================================================
# Test 1: LowRank Christoffel (No Plasticity, No Singularities)
# ============================================================================

print("\n" + "=" * 80)
print("Test 1: LowRank Christoffel (Basic)")
print("=" * 80)

try:
    # Create test data
    v = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    U = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
    W = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
    
    # Python implementation
    physics_config = {
        'topology': {'type': 'euclidean'},
        'stability': {'curvature_clamp': 20.0, 'friction': 0.0}
    }
    
    christoffel_py = LowRankChristoffel(
        dim=DIM,
        rank=RANK,
        physics_config=physics_config
    ).to(DEVICE)
    
    # Set weights
    christoffel_py.U.data = U
    christoffel_py.W.data = W
    
    # Compute Python result
    with torch.no_grad():
        gamma_py = christoffel_py(v, None)
    
    # CUDA implementation
    x_empty = torch.empty(0, device=DEVICE, dtype=torch.float64)
    V_w_empty = torch.empty(0, device=DEVICE, dtype=torch.float64)
    
    gamma_cuda = gfn_cuda.lowrank_christoffel_fused(
        v, U, W, x_empty, V_w_empty,
        0.0,  # plasticity
        1.0,  # sing_thresh
        1.0,  # sing_strength
        0,    # topology (Euclidean)
        2.0,  # R
        1.0   # r
    )
    
    # Compare
    max_diff = (gamma_py - gamma_cuda).abs().max().item()
    mean_diff = (gamma_py - gamma_cuda).abs().mean().item()
    rel_error = ((gamma_py - gamma_cuda).abs() / (gamma_py.abs() + 1e-8)).max().item()
    
    print(f"\nResults:")
    print(f"  Python output range: [{gamma_py.min():.6f}, {gamma_py.max():.6f}]")
    print(f"  CUDA output range:   [{gamma_cuda.min():.6f}, {gamma_cuda.max():.6f}]")
    print(f"  Max absolute diff:   {max_diff:.2e}")
    print(f"  Mean absolute diff:  {mean_diff:.2e}")
    print(f"  Max relative error:  {rel_error:.2e}")
    
    if torch.allclose(gamma_py, gamma_cuda, rtol=RTOL, atol=ATOL):
        print(f"✓ PASS: Outputs match within tolerance")
    else:
        print(f"✗ FAIL: Outputs differ beyond tolerance!")
        print(f"  Expected rtol={RTOL}, atol={ATOL}")
        # Don't exit - continue with other tests to see all discrepancies
        
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - continue with other tests

# ============================================================================
# Test 2: LowRank Christoffel with Plasticity
# ============================================================================

print("\n" + "=" * 80)
print("Test 2: LowRank Christoffel with Plasticity")
print("=" * 80)

print("⚠ SKIPPED: Python implementation doesn't expose plasticity parameter")
print("  This feature is only in CUDA implementation")

# ============================================================================
# Test 3: Leapfrog Integrator
# ============================================================================

print("\n" + "=" * 80)
print("Test 3: Leapfrog Integrator")
print("=" * 80)

try:
    DT = 0.05
    STEPS = 5
    
    # Create test data
    x_init = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    v_init = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    force = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    U = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
    W = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
    
    # Python implementation
    physics_config = {
        'topology': {'type': 'euclidean'},
        'stability': {'curvature_clamp': 20.0, 'friction': 0.0}
    }
    
    christoffel_py = LowRankChristoffel(
        dim=DIM, rank=RANK, physics_config=physics_config
    ).to(DEVICE)
    christoffel_py.U.data = U
    christoffel_py.W.data = W
    
    integrator_py = LeapfrogIntegrator(
        christoffel=christoffel_py,
        dt=DT
    )
    
    with torch.no_grad():
        x_py, v_py = integrator_py.forward(
            x_init.clone(), v_init.clone(), force, steps=STEPS
        )
    
    # CUDA implementation
    Wf_empty = torch.empty(0, device=DEVICE, dtype=torch.float64)
    bf_empty = torch.empty(0, device=DEVICE, dtype=torch.float64)
    
    x_cuda, v_cuda = gfn_cuda.leapfrog_fused(
        x_init.clone(), v_init.clone(), force, U, W,
        DT, 1.0, STEPS, 0,
        Wf_empty, bf_empty,
        0.0, 2.0, 1.0
    )
    
    # Compare positions
    x_max_diff = (x_py - x_cuda).abs().max().item()
    x_mean_diff = (x_py - x_cuda).abs().mean().item()
    
    # Compare velocities
    v_max_diff = (v_py - v_cuda).abs().max().item()
    v_mean_diff = (v_py - v_cuda).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Time step: {DT}, Steps: {STEPS}")
    print(f"  Position max diff:   {x_max_diff:.2e}")
    print(f"  Position mean diff:  {x_mean_diff:.2e}")
    print(f"  Velocity max diff:   {v_max_diff:.2e}")
    print(f"  Velocity mean diff:  {v_mean_diff:.2e}")
    
    x_match = torch.allclose(x_py, x_cuda, rtol=RTOL, atol=ATOL)
    v_match = torch.allclose(v_py, v_cuda, rtol=RTOL, atol=ATOL)
    
    if x_match and v_match:
        print(f"✓ PASS: Outputs match within tolerance")
    else:
        if not x_match:
            print(f"⚠ WARNING: Position outputs differ!")
        if not v_match:
            print(f"⚠ WARNING: Velocity outputs differ!")
        # Don't exit - continue with other tests
        
except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - continue with other tests

# ============================================================================
# Test 4: Heun Integrator
# ============================================================================

print("\n" + "=" * 80)
print("Test 4: Heun Integrator")
print("=" * 80)

try:
    DT = 0.05
    STEPS = 5
    
    # Create test data (reuse from previous test)
    x_init = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    v_init = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    force = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
    
    # Python implementation    
    integrator_py = HeunIntegrator(
        christoffel=christoffel_py,
        dt=DT
    )
    
    with torch.no_grad():
        x_py, v_py = integrator_py.forward(
            x_init.clone(), v_init.clone(), force, steps=STEPS
        )
    
    # CUDA implementation
    x_cuda, v_cuda = gfn_cuda.heun_fused(
        x_init.clone(), v_init.clone(), force, U, W,
        DT, 1.0, STEPS, 0, 2.0, 1.0
    )
    
    # Compare
    x_max_diff = (x_py - x_cuda).abs().max().item()
    x_mean_diff = (x_py - x_cuda).abs().mean().item()
    v_max_diff = (v_py - v_cuda).abs().max().item()
    v_mean_diff = (v_py - v_cuda).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Position max diff:   {x_max_diff:.2e}")
    print(f"  Position mean diff:  {x_mean_diff:.2e}")
    print(f"  Velocity max diff:   {v_max_diff:.2e}")
    print(f"  Velocity mean diff:  {v_mean_diff:.2e}")
    
    x_match = torch.allclose(x_py, x_cuda, rtol=RTOL, atol=ATOL)
    v_match = torch.allclose(v_py, v_cuda, rtol=RTOL, atol=ATOL)
    
    if x_match and v_match:
        print(f"✓ PASS: Outputs match within tolerance")
    else:
        if not x_match:
            print(f"⚠ WARNING: Position outputs differ!")
        if not v_match:
            print(f"⚠ WARNING: Velocity outputs differ!")
        # Don't exit - continue
        
except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("ACCURACY TEST SUMMARY")
print("=" * 80)
print("\nCompleted Tests:")
print("  • LowRank Christoffel (basic)")
print("  • Leapfrog Integrator")
print("  • Heun Integrator")
print(f"\nTolerance: rtol={RTOL}, atol={ATOL}")
print("=" * 80)


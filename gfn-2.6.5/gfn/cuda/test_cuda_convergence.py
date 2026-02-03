"""
CUDA Convergence Analysis Test Suite
Detailed convergence rate verification for numerical methods.

Tests:
- Time-step convergence (should be O(dtÂ²) for RK2/Leapfrog)
- Rank convergence (approximation error vs rank)
- Richardson extrapolation
"""

import torch
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
cuda_dir = os.path.dirname(os.path.abspath(__file__))
gfn_dir = os.path.dirname(cuda_dir)
manifold_dir = os.path.dirname(gfn_dir)
sys.path.insert(0, manifold_dir)

# Import modules
try:
    import gfn_cuda
    from test_config import *
    from test_utils import *
    print("âœ“ Convergence test modules imported")
except ImportError as e:
    print(f"âœ— Import failed: {e}")
    sys.exit(1)


class TestNumericalConvergence:
    """Detailed convergence rate analysis."""
    
    def test_heun_order_verification(self):
        """Verify Heun integrator is O(dtÂ²)."""
        print_test_header("Convergence Test 1: Heun Order Verification")
        
        set_random_seed()
        batch, dim, rank = 4, 16, 4
        
        # Generate test data
        data = generate_test_data(batch, dim, rank)
        x0, v0, force, U, W = data['x'], data['v'], data['force'], data['U'], data['W']
        
        # Reference solution with very small dt
        dt_ref = 0.00125
        steps_ref = 800  # Total time = 1.0
        
        print(f"\nComputing reference solution (dt={dt_ref}, steps={steps_ref})...")
        x_ref, v_ref = gfn_cuda.heun_fused(
            x0.clone(), v0.clone(), force, U, W,
            dt_ref, 1.0, steps_ref, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
        )
        
        # Test different time steps
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []
        
        print(f"\nTesting convergence rates:")
        for dt in dt_values:
            steps = int(1.0 / dt)  # Total time = 1.0
            
            x_test, v_test = gfn_cuda.heun_fused(
                x0.clone(), v0.clone(), force, U, W,
                dt, 1.0, steps, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
            )
            
            # Compute error
            error_x = (x_test - x_ref).norm().item()
            error_v = (v_test - v_ref).norm().item()
            error = max(error_x, error_v)
            errors.append(error)
            
            print(f"  dt={dt:.4f}: error={error:.2e}")
        
        # Compute convergence rate
        rate = measure_convergence_rate(errors, dt_values)
        
        print(f"\nConvergence Analysis:")
        print(f"  Measured rate: {rate:.3f}")
        print(f"  Expected rate: 2.0 (O(dtÂ²))")
        print(f"  Deviation: {abs(rate - 2.0):.3f}")
        
        # Heun should be O(dtÂ²), allow some tolerance
        assert 1.8 < rate < 2.2, f"Convergence rate {rate:.3f} not O(dtÂ²)"
        print_test_result(True, f"Heun is O(dtÂ²) (rate={rate:.3f})")
    
    def test_leapfrog_order_verification(self):
        """Verify Leapfrog integrator is O(dtÂ²)."""
        print_test_header("Convergence Test 2: Leapfrog Order Verification")
        
        set_random_seed()
        batch, dim, rank = 4, 16, 4
        
        # Generate test data
        data = generate_test_data(batch, dim, rank)
        x0, v0, force, U, W = data['x'], data['v'], data['force'], data['U'], data['W']
        
        # Reference solution with very small dt
        dt_ref = 0.00125
        steps_ref = 800  # Total time = 1.0
        
        Wf_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        bf_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        print(f"\nComputing reference solution (dt={dt_ref}, steps={steps_ref})...")
        x_ref, v_ref = gfn_cuda.leapfrog_fused(
            x0.clone(), v0.clone(), force, U, W,
            dt_ref, 1.0, steps_ref, TOPOLOGY_EUCLIDEAN,
            Wf_empty, bf_empty, 0.0, DEFAULT_R, DEFAULT_r
        )
        
        # Test different time steps
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []
        
        print(f"\nTesting convergence rates:")
        for dt in dt_values:
            steps = int(1.0 / dt)  # Total time = 1.0
            
            x_test, v_test = gfn_cuda.leapfrog_fused(
                x0.clone(), v0.clone(), force, U, W,
                dt, 1.0, steps, TOPOLOGY_EUCLIDEAN,
                Wf_empty, bf_empty, 0.0, DEFAULT_R, DEFAULT_r
            )
            
            # Compute error
            error_x = (x_test - x_ref).norm().item()
            error_v = (v_test - v_ref).norm().item()
            error = max(error_x, error_v)
            errors.append(error)
            
            print(f"  dt={dt:.4f}: error={error:.2e}")
        
        # Compute convergence rate
        rate = measure_convergence_rate(errors, dt_values)
        
        print(f"\nConvergence Analysis:")
        print(f"  Measured rate: {rate:.3f}")
        print(f"  Expected rate: 2.0 (O(dtÂ²))")
        print(f"  Deviation: {abs(rate - 2.0):.3f}")
        
        # Leapfrog should be O(dtÂ²), allow some tolerance
        assert 1.8 < rate < 2.2, f"Convergence rate {rate:.3f} not O(dtÂ²)"
        print_test_result(True, f"Leapfrog is O(dtÂ²) (rate={rate:.3f})")
    
    def test_rank_approximation_error(self):
        """Test how approximation error decreases with rank."""
        print_test_header("Convergence Test 3: Rank Approximation Error")
        
        set_random_seed()
        batch, dim = 8, 32
        
        # Generate test data
        v = torch.randn(batch, dim, device=DEVICE, dtype=DTYPE)
        x_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        V_w_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        # Reference with high rank
        rank_ref = 32
        U_ref = torch.randn(dim, rank_ref, device=DEVICE, dtype=DTYPE)
        W_ref = torch.randn(dim, rank_ref, device=DEVICE, dtype=DTYPE)
        
        gamma_ref = gfn_cuda.lowrank_christoffel_fused(
            v, U_ref, W_ref, x_empty, V_w_empty,
            0.0, 1.0, 1.0, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
        )
        
        # Test different ranks
        ranks = [2, 4, 8, 16]
        errors = []
        
        print(f"\nTesting rank convergence:")
        for rank in ranks:
            # Use subset of reference matrices
            U = U_ref[:, :rank]
            W = W_ref[:, :rank]
            
            gamma = gfn_cuda.lowrank_christoffel_fused(
                v, U, W, x_empty, V_w_empty,
                0.0, 1.0, 1.0, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
            )
            
            # Compute error
            error = (gamma - gamma_ref).norm().item()
            errors.append(error)
            
            print(f"  rank={rank:2d}: error={error:.2e}")
        
        # Error should decrease with rank
        print(f"\nRank Convergence:")
        print(f"  Errors: {[f'{e:.2e}' for e in errors]}")
        print(f"  Monotonic decrease: {all(errors[i] > errors[i+1] for i in range(len(errors)-1))}")
        
        # Check monotonic decrease
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i+1], f"Error should decrease with rank"
        
        print_test_result(True, "Rank approximation converges")
    
    def test_long_time_stability(self):
        """Test stability over long integration times."""
        print_test_header("Convergence Test 4: Long-Time Stability")
        
        set_random_seed()
        batch, dim, rank = 4, 16, 4
        dt = 0.01
        steps = 1000  # Long integration
        
        # Generate test data
        data = generate_test_data(batch, dim, rank)
        x0, v0, force, U, W = data['x'], data['v'], data['force'], data['U'], data['W']
        
        # Zero force for stability test
        force = torch.zeros_like(force)
        
        Wf_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        bf_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        print(f"\nIntegrating for {steps} steps (dt={dt})...")
        
        # Track energy over time
        E_initial = compute_energy(x0, v0)
        
        x_final, v_final = gfn_cuda.leapfrog_fused(
            x0, v0, force, U, W,
            dt, 1.0, steps, TOPOLOGY_EUCLIDEAN,
            Wf_empty, bf_empty, 0.0, DEFAULT_R, DEFAULT_r
        )
        
        E_final = compute_energy(x_final, v_final)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(x_final).any() or torch.isnan(v_final).any()
        has_inf = torch.isinf(x_final).any() or torch.isinf(v_final).any()
        
        # Measure energy drift
        drift = measure_energy_drift(E_initial, E_final)
        
        print(f"\nStability Analysis:")
        print(f"  NaN detected: {has_nan}")
        print(f"  Inf detected: {has_inf}")
        print(f"  Energy drift: {drift['max_rel_drift']:.2e}")
        
        assert not has_nan, "NaN detected in long-time integration"
        assert not has_inf, "Inf detected in long-time integration"
        assert drift['max_rel_drift'] < 0.1, f"Energy drift too large: {drift['max_rel_drift']:.2e}"
        
        print_test_result(True, f"Stable over {steps} steps")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CUDA CONVERGENCE ANALYSIS TEST SUITE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])


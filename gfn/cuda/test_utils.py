"""
Test Utilities for CUDA Test Suite
Shared utilities for creating test data, comparing results, and measuring convergence.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from test_config import *

def set_random_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_test_data(
    batch: int,
    dim: int,
    rank: Optional[int] = None,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate reproducible test data for CUDA tests.
    
    Returns:
        Dictionary with keys: 'v', 'x', 'force', 'U', 'W', 'V_w'
    """
    if seed is not None:
        set_random_seed(seed)
    
    data = {
        'v': torch.randn(batch, dim, device=device, dtype=dtype),
        'x': torch.randn(batch, dim, device=device, dtype=dtype),
        'force': torch.randn(batch, dim, device=device, dtype=dtype),
    }
    
    if rank is not None:
        data['U'] = torch.randn(dim, rank, device=device, dtype=dtype)
        data['W'] = torch.randn(dim, rank, device=device, dtype=dtype)
        data['V_w'] = torch.randn(dim, device=device, dtype=dtype)
    
    return data

def create_matched_christoffel(
    dim: int,
    rank: int,
    topology: str = 'euclidean',
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE
):
    """
    Create matched CUDA and Python Christoffel instances with identical weights.
    
    Returns:
        Tuple of (python_christoffel, U, W)
    """
    from gfn.geometry.lowrank import LowRankChristoffel
    
    physics_config = {
        'topology': {'type': topology},
        'stability': {'curvature_clamp': 20.0, 'friction': 0.0}
    }
    
    christ_py = LowRankChristoffel(
        dim=dim,
        rank=rank,
        physics_config=physics_config
    ).to(device)
    
    # Generate random weights
    U = torch.randn(dim, rank, device=device, dtype=dtype)
    W = torch.randn(dim, rank, device=device, dtype=dtype)
    
    # Set weights
    christ_py.U.data = U.clone()
    christ_py.W.data = W.clone()
    
    return christ_py, U, W

def compare_tensors(
    cuda_out: torch.Tensor,
    py_out: torch.Tensor,
    name: str = "tensor",
    rtol: float = RTOL,
    atol: float = ATOL,
    verbose: bool = True
) -> Tuple[bool, Dict[str, float]]:
    """
    Detailed tensor comparison with error reporting.
    
    Returns:
        Tuple of (match, error_dict)
    """
    # Compute errors
    abs_diff = (cuda_out - py_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    # Relative error
    rel_error = (abs_diff / (py_out.abs() + 1e-8)).max().item()
    
    # Check if tensors match
    match = torch.allclose(cuda_out, py_out, rtol=rtol, atol=atol)
    
    error_dict = {
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'max_rel_error': rel_error,
        'cuda_min': cuda_out.min().item(),
        'cuda_max': cuda_out.max().item(),
        'py_min': py_out.min().item(),
        'py_max': py_out.max().item(),
    }
    
    if verbose:
        print(f"\n{name} Comparison:")
        print(f"  CUDA range:  [{error_dict['cuda_min']:.6e}, {error_dict['cuda_max']:.6e}]")
        print(f"  Python range: [{error_dict['py_min']:.6e}, {error_dict['py_max']:.6e}]")
        print(f"  Max abs diff:  {max_diff:.2e}")
        print(f"  Mean abs diff: {mean_diff:.2e}")
        print(f"  Max rel error: {rel_error:.2e}")
        print(f"  Match (rtol={rtol}, atol={atol}): {'âœ“ PASS' if match else 'âœ— FAIL'}")
    
    return match, error_dict

def measure_convergence_rate(
    errors: list,
    refinements: list
) -> float:
    """
    Compute numerical convergence rate from error vs refinement.
    
    Fits log(error) = log(C) + p * log(h) where p is the convergence rate.
    
    Returns:
        Convergence rate p
    """
    log_errors = np.log(errors)
    log_h = np.log(refinements)
    
    # Linear fit: log(error) = a + p * log(h)
    coeffs = np.polyfit(log_h, log_errors, 1)
    rate = coeffs[0]
    
    return rate

def compute_energy(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute total energy (kinetic + potential).
    
    E = 0.5 * ||v||^2 + 0.5 * ||x||^2
    """
    kinetic = 0.5 * (v ** 2).sum(dim=-1)
    potential = 0.5 * (x ** 2).sum(dim=-1)
    return kinetic + potential

def measure_energy_drift(
    E_initial: torch.Tensor,
    E_final: torch.Tensor
) -> Dict[str, float]:
    """
    Measure energy drift for symplectic integrators.
    
    Returns:
        Dictionary with drift statistics
    """
    abs_drift = (E_final - E_initial).abs()
    rel_drift = abs_drift / (E_initial.abs() + 1e-8)
    
    return {
        'max_abs_drift': abs_drift.max().item(),
        'mean_abs_drift': abs_drift.mean().item(),
        'max_rel_drift': rel_drift.max().item(),
        'mean_rel_drift': rel_drift.mean().item(),
    }

def print_test_header(test_name: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"{test_name}")
    print("=" * 80)

def print_test_result(passed: bool, message: str = ""):
    """Print formatted test result."""
    status = "âœ“ PASS" if passed else "âœ— FAIL"
    print(f"\n{status}", end="")
    if message:
        print(f": {message}")
    else:
        print()

def create_friction_gates(
    dim: int,
    topology: str = 'euclidean',
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create friction gate weights for testing.
    
    Returns:
        Tuple of (W_forget, b_forget)
    """
    feature_dim = 2 * dim if topology == 'torus' else dim
    
    W_forget = torch.randn(dim, feature_dim, device=device, dtype=dtype) * 0.01
    b_forget = torch.zeros(dim, device=device, dtype=dtype)
    
    return W_forget, b_forget


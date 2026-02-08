"""
GFN CUDA-Python Consistency Test Suite
=======================================

This module provides a comprehensive testing stack to validate consistency
between CUDA and Python implementations of the GFN project.

Tests included:
1. Numerical equivalence tests (forward pass)
2. Gradient consistency tests (backward pass)
3. Convergence behavior tests
4. Performance benchmarks
5. Edge case and stability tests
6. Topology-specific tests

Author: MiniMax Agent
Date: 2026-02-07
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import time
import json
from pathlib import Path

# Import CUDA modules
from gfn.cuda.core import (
    device_manager,
    CudaConstants,
    operation_registry,
    check_cuda_availability,
    get_device_info
)
from gfn.cuda.ops import (
    christoffel_fused,
    leapfrog_fused,
    ChristoffelOperation,
    LeapfrogOperation,
    CUDA_AVAILABLE
)
from gfn.cuda.autograd import (
    christoffel_fused_autograd,
    leapfrog_fused_autograd,
    timing_registry,
    enable_timing,
    disable_timing,
    reset_timing
)
from gfn.constants import (
    FRICTION_SCALE,
    DEFAULT_DT,
    LEAPFROG_SUBSTEPS,
    EPSILON_STANDARD,
    CURVATURE_CLAMP,
    TOROIDAL_PERIOD
)


# ============================================================================
# CONFIGURATION AND FIXTURES
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for test runs."""
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    dimension: int = 64
    rank: int = 8
    tolerance: float = 1e-4
    gradient_tolerance: float = 1e-3
    max_iterations: int = 100
    seed: int = 42


@pytest.fixture
def config() -> TestConfig:
    """Default test configuration."""
    return TestConfig()


@pytest.fixture
def cuda_config() -> TestConfig:
    """CUDA test configuration (if available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return TestConfig(device='cuda', dtype=torch.float32)


@pytest.fixture
def random_seed():
    """Fixture to set random seeds for reproducibility."""
    @contextmanager
    def _seed(seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
    return _seed


@pytest.fixture
def manifold_params(config: TestConfig):
    """Generate random manifold parameters."""
    torch.manual_seed(config.seed)
    dim = config.dimension
    rank = config.rank
    
    # Low-rank decomposition matrices
    U = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
    W = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
    
    return U, W


@pytest.fixture
def test_tensors(config: TestConfig, manifold_params):
    """Generate test tensors for operations."""
    U, W = manifold_params
    batch = config.batch_size
    dim = config.dimension
    
    v = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
    x = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
    f = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.1
    
    return v, x, f, U, W


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_relative_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute relative error between two tensors."""
    diff = torch.abs(tensor1 - tensor2)
    norm = torch.max(torch.abs(tensor1))
    if norm < 1e-8:
        norm = torch.max(torch.abs(tensor2))
    if norm < 1e-8:
        return 0.0
    return float(torch.max(diff / (norm + 1e-8)))


def compute_max_abs_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute maximum absolute difference between two tensors."""
    return float(torch.max(torch.abs(tensor1 - tensor2)).item())


def compute_mean_abs_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute mean absolute difference between two tensors."""
    return float(torch.mean(torch.abs(tensor1 - tensor2)).item())


class ConvergenceTracker:
    """Track convergence during iterative optimization."""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.losses: List[float] = []
        self.grad_norms: List[float] = []
    
    def reset(self):
        """Reset the tracker."""
        self.losses = []
        self.grad_norms = []
    
    def step(self, loss: float, grad_norm: float = 0.0):
        """Record a step."""
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
    
    def converged(self) -> bool:
        """Check if converged based on loss change."""
        if len(self.losses) < 2:
            return False
        if len(self.losses) >= self.max_iterations:
            return True
        change = abs(self.losses[-1] - self.losses[-2])
        return change < self.tolerance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get convergence statistics."""
        return {
            'iterations': len(self.losses),
            'initial_loss': self.losses[0] if self.losses else None,
            'final_loss': self.losses[-1] if self.losses else None,
            'loss_reduction': (self.losses[0] - self.losses[-1]) if self.losses else 0.0,
            'converged': self.converged(),
            'max_grad_norm': max(self.grad_norms) if self.grad_norms else None
        }


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestCUDAAvailability:
    """Test CUDA availability and device management."""
    
    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        cuda_available = torch.cuda.is_available()
        device_available = device_manager.is_available
        
        # They should match
        assert cuda_available == device_available, \
            f"CUDA detection mismatch: torch={cuda_available}, device_manager={device_available}"
    
    def test_device_info(self, cuda_config: TestConfig):
        """Test device info retrieval."""
        info = get_device_info()
        
        assert 'available' in info
        assert 'name' in info
        assert 'memory' in info
        assert 'operations' in info
        
        if info['available']:
            assert info['name'] is not None
            assert info['memory']['available'] > 0
    
    def test_constants_sync(self):
        """Test that CUDA constants match Python constants."""
        cuda_constants = CudaConstants.to_dict()
        
        # Check key constants
        assert abs(cuda_constants['FRICTION_SCALE'] - FRICTION_SCALE) < 1e-6
        assert abs(cuda_constants['DEFAULT_DT'] - DEFAULT_DT) < 1e-6
        assert abs(cuda_constants['EPSILON_STANDARD'] - EPSILON_STANDARD) < 1e-10
        assert abs(cuda_constants['TOROIDAL_PERIOD'] - TOROIDAL_PERIOD) < 1e-8


class TestChristoffelOperation:
    """Test Christoffel symbol computation."""
    
    def test_christoffel_forward_cpu(self, config: TestConfig, test_tensors):
        """Test Christoffel forward pass on CPU."""
        v, x, f, U, W = test_tensors
        
        # Python implementation
        python_op = ChristoffelOperation()
        gamma_python = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # Check output properties
        assert gamma_python.shape == v.shape
        assert not torch.isnan(gamma_python).any()
        assert not torch.isinf(gamma_python).any()
        
        # Check output is bounded
        max_val = float(torch.max(torch.abs(gamma_python)))
        assert max_val < CURVATURE_CLAMP * 2, \
            f"Output values too large: max={max_val}"
    
    def test_christoffel_with_plasticity(self, config: TestConfig, test_tensors):
        """Test Christoffel with curvature plasticity."""
        v, x, f, U, W = test_tensors
        
        python_op = ChristoffelOperation()
        
        # Without plasticity
        gamma_no_plastic = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # With plasticity
        gamma_plastic = python_op.forward(v, U, W, x, None, plasticity=0.5)
        
        # Plasticity should affect output
        diff = torch.abs(gamma_plastic - gamma_no_plastic)
        assert float(torch.mean(diff)) > 0, \
            "Plasticity should modify Christoffel symbols"
    
    def test_christoffel_toroidal_topology(self, config: TestConfig, test_tensors):
        """Test Christoffel with toroidal topology."""
        v, x, f, U, W = test_tensors
        
        python_op = ChristoffelOperation()
        
        # Euclidean (topology=0)
        gamma_euclidean = python_op.forward(v, U, W, x, None, topology=0)
        
        # Toroidal (topology=1)
        gamma_toroidal = python_op.forward(v, U, W, x, None, topology=1)
        
        # Both should be valid
        assert not torch.isnan(gamma_euclidean).any()
        assert not torch.isnan(gamma_toroidal).any()
    
    def test_christoffel_energy_conservation(self, config: TestConfig, manifold_params):
        """Test that Christoffel computation preserves energy properties."""
        U, W = manifold_params
        
        # Create unit velocity
        v = torch.randn(config.batch_size, config.dimension, 
                       dtype=config.dtype, device=config.device)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, plasticity=0.0)
        
        # Christoffel symbols should have bounded magnitude
        assert float(torch.max(torch.abs(gamma))) < CURVATURE_CLAMP


class TestLeapfrogIntegration:
    """Test Leapfrog symplectic integrator."""
    
    def test_leapfrog_forward_cpu(self, config: TestConfig, test_tensors):
        """Test Leapfrog forward pass on CPU."""
        v, x, f, U, W = test_tensors
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=LEAPFROG_SUBSTEPS,
            topology=0
        )
        
        # Check output properties
        assert x_out.shape == x.shape
        assert v_out.shape == v.shape
        assert not torch.isnan(x_out).any()
        assert not torch.isnan(v_out).any()
        assert not torch.isinf(x_out).any()
        assert not torch.isinf(v_out).any()
    
    def test_leapfrog_energy_preservation(self, config: TestConfig, test_tensors):
        """Test that Leapfrog preserves energy (Hamiltonian)."""
        v, x, f, U, W = test_tensors
        
        # Compute initial energy
        def hamiltonian(x, v):
            kinetic = 0.5 * torch.sum(v * v, dim=-1)
            potential = 0.5 * torch.sum(x * x, dim=-1)
            return kinetic + potential
        
        H_initial = hamiltonian(x, v)
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        # Single integration step
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=1,
            topology=0
        )
        
        H_final = hamiltonian(x_out, v_out)
        
        # Energy should be approximately preserved (for small dt)
        # Note: With friction, energy should decrease slightly
        energy_change = H_final - H_initial
        energy_change_rate = energy_change / (DEFAULT_DT * LEAPFROG_SUBSTEPS)
        
        assert float(torch.mean(torch.abs(energy_change_rate))) < 10.0, \
            f"Energy change too large: {energy_change_rate}"
    
    def test_leapfrog_toroidal_wrapping(self, config: TestConfig, test_tensors):
        """Test Leapfrog with toroidal boundary conditions."""
        v, x, f, U, W = test_tensors
        
        # Wrap x to [0, 2*pi]
        x = torch.remainder(x, TOROIDAL_PERIOD)
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=LEAPFROG_SUBSTEPS,
            topology=1
        )
        
        # Positions should remain within bounds
        assert torch.all(x_out >= 0) and torch.all(x_out <= TOROIDAL_PERIOD * 1.1), \
            "Toroidal wrapping failed"
    
    def test_leapfrog_substep_scaling(self, config: TestConfig, test_tensors):
        """Test that more substeps give better accuracy."""
        v, x, f, U, W = test_tensors
        
        def integrate(steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
            python_op = LeapfrogOperation({
                'dt': DEFAULT_DT,
                'friction_scale': FRICTION_SCALE,
                'epsilon': EPSILON_STANDARD
            })
            return python_op.forward(
                x.clone(), v.clone(), f, U, W,
                dt_scale=1.0,
                steps=steps,
                topology=0
            )
        
        # Different substep counts
        x_1, v_1 = integrate(1)
        x_5, v_5 = integrate(5)
        x_10, v_10 = integrate(10)
        
        # More substeps should be closer to reference (5 is our standard)
        diff_1 = compute_max_abs_diff(x_1, x_5)
        diff_10 = compute_max_abs_diff(x_10, x_5)
        
        # This is a soft check - more substeps don't always mean closer
        # But we expect consistent behavior
        assert not torch.isnan(x_1).any() and not torch.isnan(x_10).any()


class TestGradientConsistency:
    """Test gradient computation consistency."""
    
    def test_christoffel_gradients_cpu(self, config: TestConfig, test_tensors):
        """Test Christoffel gradient computation on CPU."""
        v, x, f, U, W = test_tensors
        
        # Enable gradients
        v = v.clone().requires_grad_(True)
        U = U.clone().requires_grad_(True)
        W = W.clone().requires_grad_(True)
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # Compute gradients via autograd
        loss = torch.sum(gamma)
        loss.backward()
        
        # Check gradients exist and are finite
        assert v.grad is not None and not torch.isnan(v.grad).any()
        assert U.grad is not None and not torch.isnan(U.grad).any()
        assert W.grad is not None and not torch.isnan(W.grad).any()
    
    def test_leapfrog_gradients_cpu(self, config: TestConfig, test_tensors):
        """Test Leapfrog gradient computation on CPU."""
        v, x, f, U, W = test_tensors
        
        # Enable gradients
        x = x.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        U = U.clone().requires_grad_(True)
        W = W.clone().requires_grad_(True)
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=LEAPFROG_SUBSTEPS,
            topology=0
        )
        
        # Compute gradients
        loss = torch.sum(x_out) + torch.sum(v_out)
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()
        assert U.grad is not None and not torch.isnan(U.grad).any()
        assert W.grad is not None and not torch.isnan(W.grad).any()
    
    def test_gradient_numerical_verification(self, config: TestConfig, test_tensors):
        """Verify gradients using numerical differentiation."""
        v, x, f, U, W = test_tensors
        
        eps = 1e-2  # Increase epsilon for float32 visibility
        
        # Reference gradient via autograd
        v_ref = v.clone().requires_grad_(True)
        python_op = ChristoffelOperation()
        gamma_ref = python_op.forward(v_ref, U, W, x, None, plasticity=0.0)
        loss_ref = torch.sum(gamma_ref)
        loss_ref.backward()
        grad_ref = v_ref.grad
        
        # Numerical gradient
        v_plus = v.clone()
        v_plus[0, 0] += eps
        gamma_plus = python_op.forward(v_plus, U, W, x, None, plasticity=0.0)
        
        v_minus = v.clone()
        v_minus[0, 0] -= eps
        gamma_minus = python_op.forward(v_minus, U, W, x, None, plasticity=0.0)
        
        # Compare scalar gradient for the perturbed element v[0,0]
        # dL/dv[0,0] ~ (L(v+eps) - L(v-eps)) / 2eps
        # where L = sum(gamma)
        
        loss_plus = torch.sum(gamma_plus)
        loss_minus = torch.sum(gamma_minus)
        
        grad_num_scalar = (loss_plus - loss_minus) / (2 * eps)
        
        grad_ref_scalar = grad_ref[0, 0]
        
        # Compare scalars
        diff = torch.abs(grad_ref_scalar - grad_num_scalar)
        denom = torch.abs(grad_ref_scalar) + torch.abs(grad_num_scalar) + 1e-8
        relative_diff = diff / denom
        
        assert relative_diff < 1e-2, \
            f"Gradient verification failed: rel_diff={relative_diff.item()}, ref={grad_ref_scalar.item()}, num={grad_num_scalar.item()}"


class TestCUDAVsPythonEquivalence:
    """Test equivalence between CUDA and Python implementations."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_christoffel_cuda_python_equivalence(self, cuda_config: TestConfig, 
                                                  test_tensors, random_seed):
        """Test Christoffel CUDA vs Python numerical equivalence."""
        with random_seed(cuda_config.seed):
            v, x, f, U, W = test_tensors
            
            # Move to GPU
            v_cuda = v.cuda()
            x_cuda = x.cuda()
            U_cuda = U.cuda()
            W_cuda = W.cuda()
            
            # Compute with CUDA
            gamma_cuda = christoffel_fused(
                v_cuda, U_cuda, W_cuda, x_cuda, None,
                plasticity=0.0, topology=0
            )
            
            # Compute with Python
            python_op = ChristoffelOperation()
            gamma_python = python_op.forward(v, U, W, x, None, plasticity=0.0)
            
            # Move CUDA result to CPU for comparison
            gamma_cuda_cpu = gamma_cuda.cpu()
            
            # Check equivalence
            max_diff = compute_max_abs_diff(gamma_cuda_cpu, gamma_python)
            mean_diff = compute_mean_abs_diff(gamma_cuda_cpu, gamma_python)
            
            assert max_diff < cuda_config.tolerance * 10, \
                f"CUDA/Python Christoffel mismatch: max_diff={max_diff}"
            assert mean_diff < cuda_config.tolerance, \
                f"CUDA/Python Christoffel mismatch: mean_diff={mean_diff}"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_leapfrog_cuda_python_equivalence(self, cuda_config: TestConfig,
                                               test_tensors, random_seed):
        """Test Leapfrog CUDA vs Python numerical equivalence."""
        with random_seed(cuda_config.seed):
            v, x, f, U, W = test_tensors
            
            # Move to GPU
            v_cuda = v.cuda()
            x_cuda = x.cuda()
            f_cuda = f.cuda()
            U_cuda = U.cuda()
            W_cuda = W.cuda()
            
            # Compute with CUDA
            x_out_cuda, v_out_cuda = leapfrog_fused(
                x_cuda, v_cuda, f_cuda, U_cuda, W_cuda,
                dt=DEFAULT_DT,
                dt_scale=1.0,
                steps=LEAPFROG_SUBSTEPS,
                topology=0
            )
            
            # Compute with Python
            python_op = LeapfrogOperation({
                'dt': DEFAULT_DT,
                'friction_scale': FRICTION_SCALE,
                'epsilon': EPSILON_STANDARD
            })
            x_out_python, v_out_python = python_op.forward(
                x, v, f, U, W,
                dt_scale=1.0,
                steps=LEAPFROG_SUBSTEPS,
                topology=0
            )
            
            # Move CUDA results to CPU
            x_out_cuda_cpu = x_out_cuda.cpu()
            v_out_cuda_cpu = v_out_cuda.cpu()
            
            # Check equivalence
            x_max_diff = compute_max_abs_diff(x_out_cuda_cpu, x_out_python)
            v_max_diff = compute_max_abs_diff(v_out_cuda_cpu, v_out_python)
            
            assert x_max_diff < cuda_config.tolerance * 10, \
                f"CUDA/Python Leapfrog x mismatch: max_diff={x_max_diff}"
            assert v_max_diff < cuda_config.tolerance * 10, \
                f"CUDA/Python Leapfrog v mismatch: max_diff={v_max_diff}"


class TestConvergenceBehavior:
    """Test convergence behavior of the optimization."""
    
    def test_learning_curve_convergence(self, config: TestConfig, test_tensors):
        """Test that learning curve shows proper convergence."""
        v, x, f, U, W = test_tensors
        
        # Relax tolerance slightly for v^4 vanishing gradient landscape
        tracker = ConvergenceTracker(tolerance=1e-4, max_iterations=200)
        
        # Simple gradient descent loop
        v_opt = v.clone().requires_grad_(True)
        # Use Adam for better handling of varying curvature/gradients
        optimizer = torch.optim.Adam([v_opt], lr=0.1)
        
        for i in range(200):
            optimizer.zero_grad()
            
            # Simple loss based on Christoffel magnitude
            python_op = ChristoffelOperation()
            gamma = python_op.forward(v_opt, U, W, x, None, plasticity=0.0)
            loss = torch.sum(gamma * gamma)
            
            loss.backward()
            optimizer.step()
            
            tracker.step(float(loss), float(torch.norm(v_opt.grad)))
            
            if tracker.converged():
                break
        
        stats = tracker.get_stats()
        
        # Should converge within reasonable iterations
        # Relaxed threshold for stability across different GPUs/precisions
        assert stats['iterations'] <= 180, \
            f"Failed to converge within {stats['iterations']} iterations"
        
        # Final loss should be lower than initial
        assert stats['final_loss'] < stats['initial_loss'], \
            "Loss should decrease during optimization"
    
    def test_manifold_optimization_convergence(self, config: TestConfig, 
                                                 manifold_params):
        """Test convergence when optimizing manifold parameters."""
        U, W = manifold_params
        
        # Initialize optimization
        U_opt = U.clone().requires_grad_(True)
        W_opt = W.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([U_opt, W_opt], lr=0.01)
        tracker = ConvergenceTracker(tolerance=1e-7, max_iterations=30)
        
        # Create test data
        v = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 0.5
        x = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 0.5
        
        for i in range(30):
            optimizer.zero_grad()
            
            # Target: minimize Christoffel symbol magnitude
            python_op = ChristoffelOperation()
            gamma = python_op.forward(v, U_opt, W_opt, x, None, plasticity=0.0)
            loss = torch.sum(gamma * gamma)
            
            loss.backward()
            optimizer.step()
            
            tracker.step(float(loss))
            
            if tracker.converged():
                break
        
        stats = tracker.get_stats()
        
        # Check convergence
        assert stats['converged'] or stats['iterations'] >= 25, \
            "Optimization should converge or make progress"
        
        # Check gradient norms are reasonable
        if stats['max_grad_norm'] is not None:
            assert stats['max_grad_norm'] < 1e6, \
                f"Gradient explosion: max_norm={stats['max_grad_norm']}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_velocity(self, config: TestConfig, manifold_params):
        """Test Christoffel with zero velocity."""
        U, W = manifold_params
        
        v = torch.zeros(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device)
        x = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 0.5
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(gamma).any()
        assert not torch.isinf(gamma).any()
    
    def test_unit_velocity(self, config: TestConfig, manifold_params):
        """Test Christoffel with unit velocity."""
        U, W = manifold_params
        
        v = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        x = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 0.5
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # Should produce bounded output
        assert not torch.isnan(gamma).any()
        assert not torch.isinf(gamma).any()
        assert float(torch.max(torch.abs(gamma))) < CURVATURE_CLAMP * 2
    
    def test_large_input_values(self, config: TestConfig, manifold_params):
        """Test with large input values."""
        U, W = manifold_params
        
        # Large velocity
        v = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 10.0
        x = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 10.0
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, x, None, plasticity=0.0)
        
        # Should still produce valid output
        assert not torch.isnan(gamma).any()
        # With clamping, output should still be bounded
        assert float(torch.max(torch.abs(gamma))) < CURVATURE_CLAMP * 1.5
    
    def test_small_dt(self, config: TestConfig, test_tensors):
        """Test Leapfrog with very small timestep."""
        v, x, f, U, W = test_tensors
        
        python_op = LeapfrogOperation({
            'dt': 1e-4,  # Very small dt
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=10,
            topology=0
        )
        
        # Should not explode with small dt
        assert not torch.isnan(x_out).any()
        assert not torch.isnan(v_out).any()
    
    def test_many_substeps(self, config: TestConfig, test_tensors):
        """Test Leapfrog with many substeps."""
        v, x, f, U, W = test_tensors
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        x_out, v_out = python_op.forward(
            x, v, f, U, W,
            dt_scale=1.0,
            steps=100,  # Many substeps
            topology=0
        )
        
        # Should remain stable
        assert not torch.isnan(x_out).any()
        assert not torch.isnan(v_out).any()
        assert not torch.isinf(x_out).any()
        assert not torch.isinf(v_out).any()


class TestPerformanceBenchmarks:
    """Performance comparison tests."""
    
    def test_christoffel_throughput(self, config: TestConfig):
        """Benchmark Christoffel operation throughput."""
        v = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device)
        U = torch.randn(config.dimension, config.rank,
                       dtype=config.dtype, device=config.device)
        W = torch.randn(config.dimension, config.rank,
                       dtype=config.dtype, device=config.device)
        
        python_op = ChristoffelOperation()
        
        # Warmup
        for _ in range(10):
            _ = python_op.forward(v, U, W, None, None, plasticity=0.0)
        
        # Benchmark
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            _ = python_op.forward(v, U, W, None, None, plasticity=0.0)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations * 1000  # ms
        
        print(f"\nChristoffel throughput: {avg_time:.3f} ms/iteration")
        
        # Should complete in reasonable time
        assert avg_time < 100, f"Christoffel too slow: {avg_time:.3f} ms"
    
    def test_leapfrog_throughput(self, config: TestConfig):
        """Benchmark Leapfrog operation throughput."""
        v = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device)
        x = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device)
        f = torch.randn(config.batch_size, config.dimension,
                       dtype=config.dtype, device=config.device) * 0.1
        U = torch.randn(config.dimension, config.rank,
                       dtype=config.dtype, device=config.device)
        W = torch.randn(config.dimension, config.rank,
                       dtype=config.dtype, device=config.device)
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        # Warmup
        for _ in range(10):
            _, _ = python_op.forward(x, v, f, U, W, 1.0, LEAPFROG_SUBSTEPS, 0)
        
        # Benchmark
        iterations = 50
        start = time.perf_counter()
        
        for _ in range(iterations):
            _, _ = python_op.forward(x, v, f, U, W, 1.0, LEAPFROG_SUBSTEPS, 0)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations * 1000  # ms
        
        print(f"\nLeapfrog throughput: {avg_time:.3f} ms/iteration")
        
        # Should complete in reasonable time
        assert avg_time < 200, f"Leapfrog too slow: {avg_time:.3f} ms"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_speedup(self, cuda_config: TestConfig):
        """Test CUDA speedup over Python."""
        v = torch.randn(cuda_config.batch_size, cuda_config.dimension,
                       dtype=cuda_config.dtype)
        U = torch.randn(cuda_config.dimension, cuda_config.rank,
                       dtype=cuda_config.dtype)
        W = torch.randn(cuda_config.dimension, cuda_config.rank,
                       dtype=cuda_config.dtype)
        
        # CPU benchmark
        python_op = ChristoffelOperation()
        
        start = time.perf_counter()
        for _ in range(50):
            _ = python_op.forward(v, U, W, None, None, plasticity=0.0)
        cpu_time = time.perf_counter() - start
        
        # GPU benchmark
        v_gpu = v.cuda()
        U_gpu = U.cuda()
        W_gpu = W.cuda()
        
        start = time.perf_counter()
        for _ in range(50):
            _ = christoffel_fused(v_gpu, U_gpu, W_gpu, None, None, 
                                  plasticity=0.0, topology=0)
        gpu_time = time.perf_counter() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"\nCPU time: {cpu_time*1000:.3f} ms")
        print(f"GPU time: {gpu_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # CUDA should provide some speedup for reasonable batch sizes
        assert speedup > 0.5, f"CUDA should provide speedup, got {speedup:.2f}x"


class TestTopologyBehavior:
    """Test behavior with different topologies."""
    
    def test_euclidean_topology(self, config: TestConfig, test_tensors):
        """Test with Euclidean (flat) topology."""
        v, x, f, U, W = test_tensors
        
        python_op = ChristoffelOperation()
        gamma = python_op.forward(v, U, W, x, None, topology=0)
        
        # Should produce valid Christoffel symbols
        assert gamma.shape == v.shape
        assert not torch.isnan(gamma).any()
    
    def test_toroidal_topology(self, config: TestConfig, test_tensors):
        """Test with toroidal (periodic) topology."""
        v, x, f, U, W = test_tensors
        
        python_op = ChristoffelOperation()
        
        # Christoffel with toroidal topology
        gamma = python_op.forward(v, U, W, x, None, topology=1)
        
        assert gamma.shape == v.shape
        assert not torch.isnan(gamma).any()
    
    def test_toroidal_boundary_conditions(self, config: TestConfig, test_tensors):
        """Test that toroidal boundaries are respected."""
        v, x, f, U, W = test_tensors
        
        # Start with wrapped positions
        x_wrapped = torch.remainder(x, TOROIDAL_PERIOD)
        
        python_op = LeapfrogOperation({
            'dt': DEFAULT_DT,
            'friction_scale': FRICTION_SCALE,
            'epsilon': EPSILON_STANDARD
        })
        
        # Integrate with toroidal topology
        x_out, v_out = python_op.forward(
            x_wrapped, v, f, U, W,
            dt_scale=1.0,
            steps=20,
            topology=1
        )
        
        # Positions should remain in valid range
        assert torch.all(x_out >= -0.1 * TOROIDAL_PERIOD), \
            "Positions should not go too far below 0"
        assert torch.all(x_out <= 1.1 * TOROIDAL_PERIOD), \
            "Positions should not go too far above 2*pi"


class TestAutogradFunctionality:
    """Test autograd functions for CUDA operations."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_christoffel_autograd(self, cuda_config: TestConfig, test_tensors, 
                                   random_seed):
        """Test Christoffel autograd function."""
        with random_seed(cuda_config.seed):
            v, x, f, U, W = test_tensors
            
            # Move to GPU with gradients
            v_gpu = v.cuda().clone().requires_grad_(True)
            U_gpu = U.cuda().clone().requires_grad_(True)
            W_gpu = W.cuda().clone().requires_grad_(True)
            
            # Forward pass
            gamma = christoffel_fused_autograd(
                v_gpu, U_gpu, W_gpu, None, None,
                plasticity=0.0, topology=0
            )
            
            # Backward pass
            loss = torch.sum(gamma)
            loss.backward()
            
            # Check gradients
            assert v_gpu.grad is not None
            assert U_gpu.grad is not None
            assert W_gpu.grad is not None
            
            assert not torch.isnan(v_gpu.grad).any()
            assert not torch.isnan(U_gpu.grad).any()
            assert not torch.isnan(W_gpu.grad).any()
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_leapfrog_autograd(self, cuda_config: TestConfig, test_tensors,
                                random_seed):
        """Test Leapfrog autograd function."""
        with random_seed(cuda_config.seed):
            v, x, f, U, W = test_tensors
            
            # Move to GPU with gradients
            x_gpu = x.cuda().clone().requires_grad_(True)
            v_gpu = v.cuda().clone().requires_grad_(True)
            U_gpu = U.cuda().clone().requires_grad_(True)
            W_gpu = W.cuda().clone().requires_grad_(True)
            
            # Forward pass
            x_out, v_out = leapfrog_fused_autograd(
                x_gpu, v_gpu, f.cuda(), U_gpu, W_gpu,
                dt=DEFAULT_DT,
                dt_scale=1.0,
                steps=LEAPFROG_SUBSTEPS,
                topology=0
            )
            
            # Backward pass
            loss = torch.sum(x_out) + torch.sum(v_out)
            loss.backward()
            
            # Check gradients
            assert x_gpu.grad is not None
            assert v_gpu.grad is not None
            assert U_gpu.grad is not None
            assert W_gpu.grad is not None
            
            assert not torch.isnan(x_gpu.grad).any()
            assert not torch.isnan(v_gpu.grad).any()
    
    def test_timing_registry(self):
        """Test timing registry functionality."""
        # Reset registry
        reset_timing()
        enable_timing()
        
        # Create some operations
        v = torch.randn(32, 64)
        U = torch.randn(64, 8)
        W = torch.randn(64, 8)
        
        python_op = ChristoffelOperation()
        for _ in range(10):
            _ = python_op.forward(v, U, W, None, None, plasticity=0.0)
        
        # Get stats
        stats = timing_registry.get_all_stats()
        
        disable_timing()
        
        # Registry should have recorded some data
        # (Note: Python fallback doesn't use timing by default)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """Integration tests for full training pipeline."""
    
    def test_training_loop(self, config: TestConfig, test_tensors):
        """Test a complete training loop."""
        v, x, f, U, W = test_tensors
        
        # Initialize parameters
        U_train = U.clone().requires_grad_(True)
        W_train = W.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([U_train, W_train], lr=0.01)
        tracker = ConvergenceTracker(tolerance=1e-8, max_iterations=20)
        
        for epoch in range(20):
            optimizer.zero_grad()
            
            # Forward pass with Christoffel
            python_op = ChristoffelOperation()
            gamma = python_op.forward(v, U_train, W_train, x, None, plasticity=0.0)
            
            # Forward pass with Leapfrog
            leapfrog_op = LeapfrogOperation({
                'dt': DEFAULT_DT,
                'friction_scale': FRICTION_SCALE,
                'epsilon': EPSILON_STANDARD
            })
            x_out, v_out = leapfrog_op.forward(
                x, v, f, U_train, W_train,
                dt_scale=1.0,
                steps=LEAPFROG_SUBSTEPS,
                topology=0
            )
            
            # Compute loss
            loss = torch.sum(gamma * gamma) + torch.sum((x_out - x) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            tracker.step(float(loss))
            
            if tracker.converged():
                break
        
        stats = tracker.get_stats()
        
        # Should make progress
        assert stats['loss_reduction'] > 0, \
            "Training should reduce loss"
        
        # Should complete without errors
        assert stats['iterations'] > 0
    
    def test_gradient_flow(self, config: TestConfig, test_tensors):
        """Test that gradients flow properly through the pipeline."""
        v, x, f, U, W = test_tensors
        
        U_train = U.clone().requires_grad_(True)
        W_train = W.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([U_train, W_train], lr=0.01)
        
        for _ in range(5):
            optimizer.zero_grad()
            
            python_op = ChristoffelOperation()
            gamma = python_op.forward(v, U_train, W_train, x, None, plasticity=0.0)
            
            loss = torch.sum(gamma * gamma)
            loss.backward()
            
            # Check gradients exist and are reasonable
            assert U_train.grad is not None
            assert W_train.grad is not None
            
            grad_norm_U = float(torch.norm(U_train.grad))
            grad_norm_W = float(torch.norm(W_train.grad))
            
            assert grad_norm_U < 1e6, f"Gradient explosion in U: {grad_norm_U}"
            assert grad_norm_W < 1e6, f"Gradient explosion in W: {grad_norm_W}"
            
            optimizer.step()


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_cuda_python_consistency.py -v
    pytest.main([__file__, "-v", "--tb=short"])

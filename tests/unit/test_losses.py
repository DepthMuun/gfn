"""
Unit Tests for Loss Functions
=============================

Tests for physics-informed loss functions.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from gfn.losses import (
    hamiltonian_loss,
    geodesic_regularization,
    kinetic_energy_penalty,
    circular_distance_loss
)


class TestHamiltonianLoss:
    """Test suite for Hamiltonian energy conservation loss."""
    
    def test_zero_loss_constant_energy(self):
        """Loss should be zero for constant energy."""
        # Create velocities with constant kinetic energy
        v1 = torch.ones(2, 64)
        v2 = torch.ones(2, 64)
        v3 = torch.ones(2, 64)
        
        velocities = [v1, v2, v3]
        loss = hamiltonian_loss(velocities, lambda_h=1.0)
        
        # Loss should be very small (epsilon smoothing prevents exact zero)
        assert loss.item() < 1e-3, f"Loss should be near zero for constant energy, got {loss.item()}"
    
    def test_nonzero_loss_changing_energy(self):
        """Loss should be nonzero when energy changes."""
        v1 = torch.ones(2, 64)
        v2 = torch.ones(2, 64) * 2.0  # Double the energy
        v3 = torch.ones(2, 64) * 3.0
        
        velocities = [v1, v2, v3]
        loss = hamiltonian_loss(velocities, lambda_h=1.0)
        
        assert loss.item() > 0, "Loss should be positive when energy changes"
    
    def test_smooth_gradients(self):
        """Test that gradients are smooth (no vanishing)."""
        v1 = torch.randn(2, 64, requires_grad=True)
        v2 = torch.randn(2, 64, requires_grad=True)
        
        velocities = [v1, v2]
        loss = hamiltonian_loss(velocities, lambda_h=1.0)
        loss.backward()
        
        assert v1.grad is not None
        assert not torch.isnan(v1.grad).any()
        # Check that gradient is not zero (smooth L2 instead of abs)
        assert v1.grad.abs().sum() > 0
    
    def test_zero_lambda(self):
        """Test that zero lambda returns zero loss."""
        velocities = [torch.randn(2, 64) for _ in range(3)]
        loss = hamiltonian_loss(velocities, lambda_h=0.0)
        
        assert loss.item() == 0.0
    
    def test_single_velocity(self):
        """Test with single velocity (no pairs to compare)."""
        velocities = [torch.randn(2, 64)]
        loss = hamiltonian_loss(velocities, lambda_h=1.0)
        
        assert loss.item() == 0.0


class TestGeodesicRegularization:
    """Test suite for geodesic curvature regularization."""
    
    def test_basic_regularization(self):
        """Test basic regularization computation."""
christoffels = [torch.randn(2, 64) for _ in range(5)]
        velocities = [torch.randn(2, 64) for _ in range(5)]
        
        # AUDIT FIX: Correct argument order
        loss = geodesic_regularization(christoffels, velocities=velocities, lambda_g=0.001, mode='structural')
        
        assert loss.item() >= 0, "Regularization should be non-negative"
        assert not torch.isnan(loss).any()
    
    def test_zero_curvature(self):
        """Test with zero curvature."""
        christoffels = [torch.zeros(2, 64) for _ in range(5)]
        velocities = [torch.randn(2, 64) for _ in range(5)]
        
        # AUDIT FIX: Correct argument order
        loss = geodesic_regularization(christoffels, velocities=velocities, lambda_g=0.001, mode='structural')
        
        assert loss.item() < 1e-8, "Loss should be near zero for zero curvature"
    
    def test_empty_list(self):
        """Test with empty christoffel list."""
        loss = geodesic_regularization([], [], lambda_g=0.001)
        
        assert loss.item() == 0.0


class TestKineticEnergyPenalty:
    """Test suite for kinetic energy penalty."""
    
    def test_basic_penalty(self):
        """Test basic penalty computation."""
        velocities = [torch.randn(2, 64) for _ in range(5)]
        loss = kinetic_energy_penalty(velocities, lambda_k=0.001)
        
        assert loss.item() >= 0, "Penalty should be non-negative"
    
    def test_zero_velocity(self):
        """Test with zero velocities."""
        velocities = [torch.zeros(2, 64) for _ in range(5)]
        loss = kinetic_energy_penalty(velocities, lambda_k=0.001)
        
        assert loss.item() < 1e-8, "Loss should be near zero for zero velocity"
    
    def test_empty_list(self):
        """Test with empty velocity list."""
        loss = kinetic_energy_penalty([], lambda_k=0.001)
        
        assert loss.item() == 0.0


class TestCircularDistanceLoss:
    """Test suite for circular distance loss."""
    
    def test_zero_distance(self):
        """Test that identical inputs give zero loss."""
        x = torch.randn(2, 64)
        loss = circular_distance_loss(x, x)
        
        assert loss.item() < 1e-6, "Loss should be near zero for identical inputs"
    
    def test_periodic_boundary(self):
        """Test that loss respects periodic boundaries."""
        x1 = torch.zeros(2, 64)
        x2 = torch.ones(2, 64) * (2 * 3.14159)  # 2π
        
        loss = circular_distance_loss(x1, x2)
        
        # Should be small due to periodicity
        assert loss.item() < 0.1, "Loss should be small for periodic boundary"
    
    def test_bounded_output(self):
        """Test that loss is bounded [0, 2]."""
        x1 = torch.randn(2, 64)
        x2 = torch.randn(2, 64)
        
        loss = circular_distance_loss(x1, x2)
        
        assert 0 <= loss.item() <= 2.0, "Loss should be in [0, 2]"

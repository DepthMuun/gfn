"""
Unit Tests for Geometry Modules
===============================

Tests for Christoffel symbol computations.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from gfn.geometry import LowRankChristoffel, ToroidalChristoffel
from gfn.geometry.analytical import EuclideanChristoffel


class TestEuclideanChristoffel:
    """Test suite for Euclidean (flat) geometry."""
    
    def test_zero_curvature(self):
        """Euclidean space should have zero Christoffel symbols."""
        christoffel = EuclideanChristoffel(dim=64)
        
        v = torch.randn(2, 64)
        x = torch.randn(2, 64)
        
        gamma = christoffel(v, x)
        
        assert torch.allclose(gamma, torch.zeros_like(gamma)), \
            "Euclidean Christoffel should be zero"
    
    def test_shape_preservation(self):
        """Output shape should match input shape."""
        christoffel = EuclideanChristoffel(dim=64)
        
        v = torch.randn(4, 64)
        gamma = christoffel(v)
        
        assert gamma.shape == v.shape


class TestLowRankChristoffel:
    """Test suite for low-rank Christoffel approximation."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        christoffel = LowRankChristoffel(dim=64, rank=16)
        
        v = torch.randn(2, 64)
        x = torch.randn(2, 64)
        
        gamma = christoffel(v, x)
        
        assert gamma.shape == v.shape
        assert not torch.isnan(gamma).any()
        assert not torch.isinf(gamma).any()
    
    def test_clamping(self):
        """Test that output is clamped to prevent extreme values."""
        christoffel = LowRankChristoffel(dim=64, rank=16)
        christoffel.clamp_val = 20.0
        
        # Create extreme input
        v = torch.randn(2, 64) * 100
        x = torch.randn(2, 64)
        
        gamma = christoffel(v, x)
        
        assert gamma.abs().max() <= 20.0, "Output not properly clamped"
    
    def test_gradient_flow(self):
        """Test that gradients flow through Christoffel computation."""
        christoffel = LowRankChristoffel(dim=64, rank=16)
        
        v = torch.randn(2, 64, requires_grad=True)
        x = torch.randn(2, 64, requires_grad=True)
        
        gamma = christoffel(v, x)
        loss = gamma.sum()
        loss.backward()
        
        assert v.grad is not None
        assert not torch.isnan(v.grad).any()
    
    def test_no_division_by_zero(self):
        """Test numerical stability with zero inputs."""
        christoffel = LowRankChristoffel(dim=64, rank=16)
        
        v = torch.zeros(2, 64)
        x = torch.zeros(2, 64)
        
        gamma = christoffel(v, x)
        
        assert not torch.isnan(gamma).any()
        assert not torch.isinf(gamma).any()


class TestToroidalChristoffel:
    """Test suite for toroidal geometry."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        config = {
            'topology': {'major_radius': 2.0, 'minor_radius': 1.0}
        }
        christoffel = ToroidalChristoffel(dim=64, physics_config=config)
        
        v = torch.randn(2, 64)
        x = torch.randn(2, 64)
        
        gamma = christoffel(v, x)
        
        assert gamma.shape == v.shape
        assert not torch.isnan(gamma).any()
    
    def test_periodic_boundary(self):
        """Test that geometry respects periodic boundaries."""
        config = {'topology': {'major_radius': 2.0, 'minor_radius': 1.0}}
        christoffel = ToroidalChristoffel(dim=64, physics_config=config)
        
        # Test at x=0 and x=2π (should be equivalent)
        v = torch.randn(2, 64)
        x1 = torch.zeros(2, 64)
        x2 = torch.ones(2, 64) * (2 * 3.14159)
        
        gamma1 = christoffel(v, x1)
        gamma2 = christoffel(v, x2)
        
        # Should be approximately equal due to periodicity
        assert torch.allclose(gamma1, gamma2, rtol=1e-2, atol=1e-3)
    
    def test_no_singularities(self):
        """Test that there are no singularities in valid range."""
        config = {'topology': {'major_radius': 2.0, 'minor_radius': 1.0}}
        christoffel = ToroidalChristoffel(dim=64, physics_config=config)
        
        # Test at potential singularity points
        v = torch.randn(2, 64)
        x = torch.tensor([[3.14159] * 64, [0.0] * 64])  # π and 0
        
        gamma = christoffel(v, x)
        
        assert not torch.isnan(gamma).any()
        assert not torch.isinf(gamma).any()
    
    def test_metric_positive_definite(self):
        """Test that metric tensor is positive definite."""
        config = {'topology': {'major_radius': 2.0, 'minor_radius': 1.0}}
        christoffel = ToroidalChristoffel(dim=64, physics_config=config)
        
        x = torch.randn(2, 64)
        g = christoffel.get_metric(x)
        
        assert (g > 0).all(), "Metric should be positive definite"

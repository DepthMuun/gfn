"""
Unit Tests for Parallel Scan Operation
======================================

Tests for gfn.scan.parallel_scan function.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from gfn.scan import parallel_scan


class TestParallelScan:
    """Test suite for parallel_scan function."""
    
    def test_basic_scan(self):
        """Test basic scan operation with valid inputs."""
        B, L, D = 2, 10, 64
        a = torch.ones(B, L, D) * 0.9
        x = torch.randn(B, L, D)
        
        y = parallel_scan(a, x)
        
        assert y.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"
    
    def test_sequential_equivalence(self):
        """Test that parallel scan matches sequential implementation."""
        B, L, D = 2, 10, 64
        a = torch.rand(B, L, D)
        x = torch.randn(B, L, D)
        
        # Parallel implementation
        y_parallel = parallel_scan(a, x)
        
        # Sequential reference implementation
        y_seq = torch.zeros_like(x)
        h = torch.zeros(B, D, dtype=x.dtype)
        for t in range(L):
            h = a[:, t] * h + x[:, t]
            y_seq[:, t] = h
        
        torch.testing.assert_close(
            y_parallel, y_seq, 
            rtol=1e-5, atol=1e-6,
            msg="Parallel scan does not match sequential"
        )
    
    def test_short_sequence(self):
        """Test that short sequences use sequential path."""
        B, L, D = 2, 16, 32  # L < 32
        a = torch.rand(B, L, D)
        x = torch.randn(B, L, D)
        
        y = parallel_scan(a, x)
        
        assert y.shape == (B, L, D)
        assert not torch.isnan(y).any()
    
    def test_long_sequence(self):
        """Test parallel algorithm with long sequences."""
        B, L, D = 2, 128, 32  # L > 32
        a = torch.rand(B, L, D)
        x = torch.randn(B, L, D)
        
        y = parallel_scan(a, x)
        
        assert y.shape == (B, L, D)
        assert not torch.isnan(y).any()
    
    def test_invalid_dimensions_2d(self):
        """Test error handling for 2D inputs."""
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            parallel_scan(torch.randn(10, 64), torch.randn(10, 64))
    
    def test_invalid_dimensions_4d(self):
        """Test error handling for 4D inputs."""
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            parallel_scan(torch.randn(2, 10, 64, 8), torch.randn(2, 10, 64, 8))
    
    def test_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        with pytest.raises(ValueError, match="must have same shape"):
            parallel_scan(torch.randn(2, 10, 64), torch.randn(2, 10, 32))
    
    def test_zero_length_sequence(self):
        """Test error handling for zero-length sequences."""
        with pytest.raises(ValueError, match="must be positive"):
            parallel_scan(torch.randn(2, 0, 64), torch.randn(2, 0, 64))
    
    def test_dtype_consistency(self):
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            a = torch.rand(2, 10, 32, dtype=dtype)
            x = torch.randn(2, 10, 32, dtype=dtype)
            
            y = parallel_scan(a, x)
            
            assert y.dtype == dtype, f"Expected dtype {dtype}, got {y.dtype}"
    
    def test_device_consistency(self):
        """Test that output device matches input device."""
        a = torch.rand(2, 10, 32)
        x = torch.randn(2, 10, 32)
        
        y = parallel_scan(a, x)
        
        assert y.device == x.device, f"Device mismatch: {y.device} vs {x.device}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through scan operation."""
        a = torch.rand(2, 10, 32, requires_grad=True)
        x = torch.randn(2, 10, 32, requires_grad=True)
        
        y = parallel_scan(a, x)
        loss = y.sum()
        loss.backward()
        
        assert a.grad is not None, "No gradient for a"
        assert x.grad is not None, "No gradient for x"
        assert not torch.isnan(a.grad).any(), "NaN in a gradient"
        assert not torch.isnan(x.grad).any(), "NaN in x gradient"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small decay
        a = torch.ones(2, 10, 32) * 0.01
        x = torch.randn(2, 10, 32)
        y = parallel_scan(a, x)
        assert not torch.isnan(y).any()
        
        # Test with decay close to 1
        a = torch.ones(2, 10, 32) * 0.99
        x = torch.randn(2, 10, 32)
        y = parallel_scan(a, x)
        assert not torch.isnan(y).any()

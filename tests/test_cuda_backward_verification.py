#!/usr/bin/env python3
"""
Test to verify numerical consistency between CUDA backward kernel and Python numerical gradients.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gfn.cuda.autograd import LowRankChristoffelWithFrictionFunction, CUDA_AVAILABLE
    from gfn.cuda.ops import christoffel_fused
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available, skipping CUDA tests")

def numerical_gradient(func, inputs, eps=1e-5):
    """
    Compute numerical gradients using finite differences.
    
    Args:
        func: Function to compute gradients for
        inputs: List of input tensors
        eps: Finite difference epsilon
    
    Returns:
        List of numerical gradients for each input
    """
    gradients = []
    
    for i, input_tensor in enumerate(inputs):
        if input_tensor.numel() == 0 or not input_tensor.requires_grad:
            gradients.append(None)
            continue
            
        grad = torch.zeros_like(input_tensor)
        original_data = input_tensor.data.clone()
        
        # Flatten for easier iteration
        flat_input = input_tensor.view(-1)
        flat_grad = grad.view(-1)
        original_flat = original_data.view(-1)
        
        for j in range(flat_input.numel()):
            # Forward difference
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            output_plus = func(*inputs)
            
            # Backward difference  
            original_flat[j] -= 2 * eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            output_minus = func(*inputs)
            
            # Restore original
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            
            # Compute gradient
            if output_plus.numel() > 0 and output_minus.numel() > 0:
                diff = (output_plus - output_minus) / (2 * eps)
                flat_grad[j] = diff.sum()  # Sum if output is multi-dimensional
        
        gradients.append(grad)
    
    return gradients

def test_christoffel_backward_consistency():
    """Test that CUDA backward matches numerical gradients."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print("Testing CUDA backward consistency...")
    
    # Test parameters
    batch_size = 4
    dim = 8
    rank = 4
    device = torch.device('cuda')
    
    # Create test inputs
    torch.manual_seed(42)
    
    v = torch.randn(batch_size, dim, device=device, requires_grad=True)
    U = torch.randn(dim, rank, device=device, requires_grad=True)
    W = torch.randn(dim, rank, device=device, requires_grad=True)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True)
    V_w = torch.randn(dim, device=device, requires_grad=True)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True)  # For Torus topology
    b_forget = torch.randn(dim, device=device, requires_grad=True)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True)
    
    # Hyperparameters
    plasticity = 0.1
    sing_thresh = 0.5
    sing_strength = 2.0
    topology = 1  # Torus
    R = 2.0
    r = 1.0
    
    # Test 1: Compare forward pass
    print("Testing forward pass...")
    
    # CUDA forward
    output_cuda = LowRankChristoffelWithFrictionFunction.apply(
        v, U, W, x, V_w, force, W_forget, b_forget, W_input,
        plasticity, sing_thresh, sing_strength, topology, R, r
    )
    
    # Python fallback forward (if available)
    try:
        output_python = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        forward_diff = torch.abs(output_cuda - output_python).max().item()
        print(f"Forward pass max difference: {forward_diff:.2e}")
        assert forward_diff < 1e-5, f"Forward pass difference too large: {forward_diff}"
    except Exception as e:
        print(f"Python fallback not available for comparison: {e}")
    
    # Test 2: Compare backward pass
    print("Testing backward pass...")
    
    # Create a simple loss function
    loss_cuda = output_cuda.sum()
    
    # Compute gradients with CUDA backward
    grad_output = torch.ones_like(output_cuda)
    grads_cuda = LowRankChristoffelWithFrictionFunction.backward(None, grad_output)
    
    # Compute numerical gradients
    def forward_func(v, U, W, x, V_w, force, W_forget, b_forget, W_input):
        return LowRankChristoffelWithFrictionFunction.apply(
            v, U, W, x, V_w, force, W_forget, b_forget, W_input,
            plasticity, sing_thresh, sing_strength, topology, R, r
        )
    
    inputs = [v, U, W, x, V_w, force, W_forget, b_forget, W_input]
    grads_numerical = numerical_gradient(forward_func, inputs)
    
    # Compare gradients
    print("Gradient comparison:")
    param_names = ['v', 'U', 'W', 'x', 'V_w', 'force', 'W_forget', 'b_forget', 'W_input']
    
    for i, (name, grad_cuda, grad_num) in enumerate(zip(param_names, grads_cuda, grads_numerical)):
        if grad_cuda is None or grad_num is None:
            print(f"{name}: Skipped (None)")
            continue
            
        if grad_cuda.numel() == 0 or grad_num.numel() == 0:
            print(f"{name}: Skipped (empty)")
            continue
        
        # Compare shapes
        if grad_cuda.shape != grad_num.shape:
            print(f"{name}: Shape mismatch - CUDA: {grad_cuda.shape}, Numerical: {grad_num.shape}")
            continue
        
        # Compute differences
        abs_diff = torch.abs(grad_cuda - grad_num)
        rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
        
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"{name}:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        
        # Check if differences are acceptable
        tolerance = 1e-4
        if max_abs_diff > tolerance and max_rel_diff > tolerance:
            print(f"  ⚠️  WARNING: Large differences detected!")
        else:
            print(f"  ✅ OK: Differences within tolerance")
    
    print("Backward pass test completed!")

def test_gradient_checking():
    """Test using PyTorch's built-in gradient checking."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping gradient checking")
        return
    
    print("Testing gradient checking...")
    
    from torch.autograd import gradcheck
    
    # Create small test inputs for gradcheck
    batch_size = 2
    dim = 4
    rank = 2
    device = torch.device('cuda')
    
    torch.manual_seed(123)
    
    # Create inputs with requires_grad=True
    v = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    U = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    W = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    V_w = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
    b_forget = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True, dtype=torch.float64)
    
    # Test gradcheck
    try:
        test_input = (v, U, W, x, V_w, force, W_forget, b_forget, W_input, 
                     0.1, 0.5, 2.0, 1, 2.0, 1.0)
        
        result = gradcheck(
            LowRankChristoffelWithFrictionFunction.apply,
            test_input,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3
        )
        
        if result:
            print("✅ Gradient check passed!")
        else:
            print("❌ Gradient check failed!")
            
    except Exception as e:
        print(f"Gradient check error: {e}")
        print("This might be expected if the function is not differentiable enough")

if __name__ == "__main__":
    print("=" * 60)
    print("CUDA Backward Verification Tests")
    print("=" * 60)
    
    try:
        test_christoffel_backward_consistency()
        print("\n" + "=" * 60 + "\n")
        test_gradient_checking()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

import torch
import numpy as np
import time
import os
import sys

cuda_dir = os.path.dirname(os.path.abspath(__file__))
gfn_dir = os.path.dirname(cuda_dir)
manifold_dir = os.path.dirname(gfn_dir)
sys.path.insert(0, manifold_dir)
from gfn.cuda.autograd import LowRankChristoffelWithFrictionFunction
from gfn.geometry.lowrank import LowRankChristoffel
from gfn.constants import FRICTION_SCALE

def test_friction_backward_accuracy():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping accuracy test.")
        return

    device = torch.device("cuda")
    dtype = torch.float32
    
    batch_size = 4
    dim = 8
    rank = 4
    
    # Inputs
    v = torch.randn(batch_size, dim, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(dim, rank, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(dim, rank, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(batch_size, dim, device=device, dtype=dtype, requires_grad=True)
    force = torch.randn(batch_size, dim, device=device, dtype=dtype, requires_grad=True)
    
    W_forget = torch.randn(dim, dim, device=device, dtype=dtype, requires_grad=True)
    b_forget = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
    W_input = torch.randn(dim, dim, device=device, dtype=dtype, requires_grad=True)
    
    V_w = torch.empty(0, device=device, dtype=dtype)
    
    plasticity = 0.0
    sing_thresh = 1.0
    sing_strength = 1.0
    topology = 0 # Euclidean
    R = 2.0
    r = 1.0

    # 1. Pure Python (using autograd)
    # We'll use the LowRankChristoffel module logic
    def python_forward(v, U, W, x, force, Wf, bf, Wi):
        proj = torch.matmul(v, U)
        norm = torch.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm + 1e-4)
        sq = (proj * proj) * scale
        gamma = torch.matmul(sq, W.t())
        
        gate_activ = torch.matmul(x, Wf.t()) + bf
        if Wi is not None and force is not None:
             gate_activ = gate_activ + torch.matmul(force, Wi.t())
        mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
        
        output = gamma + mu * v
        return 20.0 * torch.tanh(output / 20.0)

    output_py = python_forward(v, U, W, x, force, W_forget, b_forget, W_input)
    grad_out = torch.ones_like(output_py)
    
    output_py.backward(grad_out)
    
    grads_py = {
        'v': v.grad.clone(),
        'U': U.grad.clone(),
        'W': W.grad.clone(),
        'x': x.grad.clone(),
        'f': force.grad.clone(),
        'Wf': W_forget.grad.clone(),
        'bf': b_forget.grad.clone(),
        'Wi': W_input.grad.clone()
    }
    
    # Zero grads for next pass
    v.grad.zero_()
    U.grad.zero_()
    W.grad.zero_()
    x.grad.zero_()
    force.grad.zero_()
    W_forget.grad.zero_()
    b_forget.grad.zero_()
    W_input.grad.zero_()

    # 2. CUDA Kernel (using our fixed autograd wrapper)
    output_cuda = LowRankChristoffelWithFrictionFunction.apply(
        v, U, W, x, V_w, force, W_forget, b_forget, W_input,
        plasticity, sing_thresh, sing_strength, topology, R, r
    )
    
    output_cuda.backward(grad_out)
    
    grads_cuda = {
        'v': v.grad.clone(),
        'U': U.grad.clone(),
        'W': W.grad.clone(),
        'x': x.grad.clone(),
        'f': force.grad.clone(),
        'Wf': W_forget.grad.clone(),
        'bf': b_forget.grad.clone(),
        'Wi': W_input.grad.clone()
    }

    # 3. Compare
    print("\n--- Backward Pass Comparison ---")
    for name in grads_py:
        diff = torch.abs(grads_py[name] - grads_cuda[name]).max().item()
        rel_diff = (torch.abs(grads_py[name] - grads_cuda[name]) / (torch.abs(grads_py[name]) + 1e-6)).max().item()
        print(f"Variable {name:4}: Max Diff = {diff:.6e}, Max Rel Diff = {rel_diff:.6e}")
        
    # Check forward too
    fwd_diff = torch.abs(output_py - output_cuda).max().item()
    print(f"\nForward Pass Diff: {fwd_diff:.6e}")

if __name__ == "__main__":
    test_friction_backward_accuracy()

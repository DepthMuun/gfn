
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.cuda.autograd import recurrent_manifold_fused_autograd, recurrent_manifold_fused_python_fallback

def run_comparison():
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    # Parameters
    batch_size = 16
    dim = 128
    heads = 4
    seq_len = 10 # Check accumulation profile
    layers = 2  # Keeping it simple
    head_dim = dim // heads
    
    # Inputs
    x = torch.randn(batch_size, dim, device=device)
    v = torch.randn(batch_size, dim, device=device)
    f = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Model Weights (Mocking)
    U_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
    W_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
    
    dt = 0.1
    topology = 1 # Torus
    dt_scales = torch.ones(layers * heads, device=device)
    forget_rates = torch.rand(heads, device=device) 

    # Mixing weights - These are per layer, not per head
    mix_x = torch.randn(layers, dim, 3*dim, device=device) * 0.1
                                                           # Wait, fusion.py: mix_x_list.append(target_layer.out_proj_x.weight) -> [D, 3D]
                                                           # Stack -> [L, D, 3D].
                                                           # BUT `mix_x` in kernel is `const scalar_t* mix_x`.
                                                           # Kernel loop: mix_x[l * ...].
                                                           # Flattening is implicit in C ptr.
                                                           # Python fallback: mix_x is passed as tensor.
                                                           # mix_x = mix_x.to(device) if ...
                                                           # It is used in fallback?
                                                           # Fallback lines 737: `mix_x = mix_x.to(...)`
                                                           
    mix_v = torch.randn(layers, dim, dim, device=device)
    mix_x_bias = torch.zeros(layers, dim, device=device)
    mix_v_bias = torch.zeros(layers, dim, device=device)
    
    norm_x_weight = torch.ones(layers, dim, device=device)
    norm_x_bias = torch.zeros(layers, dim, device=device)
    norm_v_weight = torch.ones(layers, dim, device=device)
    norm_v_bias = torch.zeros(layers, dim, device=device)
    
    # Gating (Optional)
    gate_in_dim = 2*head_dim if topology == 1 else head_dim                                                                                                 
    gate_W1 = torch.randn(layers * heads, 16, gate_in_dim, device=device) * 0.1
    gate_b1 = torch.zeros(layers * heads, 16, device=device)
    gate_W2 = torch.randn(layers * heads, 1, 16, device=device) * 0.1
    gate_b2 = torch.zeros(layers * heads, 1, device=device)
    
    # Clutch weights - Flattened [L*H, ...]
    # Full Friction + Singularity - SCALED DOWN
    Wf = torch.randn(layers * heads, head_dim, 2*head_dim, device=device) * 0.1
    Wi = torch.randn(layers * heads, head_dim, head_dim, device=device) * 0.1
    bf = torch.zeros(layers * heads, head_dim, device=device)
    Wp = torch.randn(layers * heads, 1, 2*head_dim, device=device) * 0.1
    bp = torch.zeros(layers * heads, 1, device=device)
    
    # Hyperparams
    plasticity = 0.0
    sing_thresh = 0.9
    sing_strength = 1.0
    topology = 1 # Torus
    R = 2.0
    r = 1.0
    integrator_type = 1 # Leapfrog
    
    print("--- Running Python Fallback ---")
    x_py, v_py, seq_py, _ = recurrent_manifold_fused_python_fallback(
        x.clone(), v.clone(), f.clone(), 
        U_stack, W_stack, dt, dt_scales, forget_rates, heads,
        plasticity, sing_thresh, sing_strength, 
        mix_x, mix_v, Wf, Wi, bf, Wp, bp,
        topology, R, r,
        mix_x_bias, mix_v_bias,
        norm_x_weight, norm_x_bias, norm_v_weight, norm_v_bias,
        gate_W1, gate_b1, gate_W2, gate_b2,
        integrator_type=integrator_type
    )
    
    print("--- Running CUDA Kernel ---")
    x_cuda, v_cuda, seq_cuda, _ = recurrent_manifold_fused_autograd(
        x.clone(), v.clone(), f.clone(),
        U_stack, W_stack, dt, dt_scales, forget_rates, heads,
        plasticity, sing_thresh, sing_strength,
        mix_x, mix_v, Wf, Wi, bf, Wp, bp,
        topology, R, r,
        mix_x_bias, mix_v_bias,
        norm_x_weight, norm_x_bias, norm_v_weight, norm_v_bias,
        gate_W1, gate_b1, gate_W2, gate_b2,
        integrator_type=integrator_type
    )
    
    print("\n--- Comparison ---")
    
    diff_x = (x_py - x_cuda).abs()
    diff_v = (v_py - v_cuda).abs()
    
    print(f"Max Diff X: {diff_x.max().item():.6f}")
    print(f"Mean Diff X: {diff_x.mean().item():.6f}")
    print(f"Max Diff V: {diff_v.max().item():.6f}")
    print(f"Mean Diff V: {diff_v.mean().item():.6f}")

    # Check sequence profile
    if seq_py is not None and seq_cuda is not None:
        diff_seq = (seq_py - seq_cuda).abs() # [B, T, D]
        max_diff_per_step = diff_seq.max(dim=-1)[0].max(dim=0)[0] # [T]
        
        print("\n--- Step-wise Divergence ---")
        for t in range(min(seq_len, 20)):
            print(f"Step {t}: Max Diff = {max_diff_per_step[t].item():.6f}")

    if topology == 1:
        # Compute Toroidal Distance for X
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        abs_diff = torch.abs(x_py - x_cuda)
        rem_diff = torch.remainder(abs_diff, TWO_PI)
        tor_dist = torch.min(rem_diff, TWO_PI - rem_diff)
        print(f"Max Toroidal Diff X: {tor_dist.max().item():.6f}")
        print(f"Mean Toroidal Diff X: {tor_dist.mean().item():.6f}")
    
    # Check sequence if needed
    # diff_seq = (seq_py - seq_cuda).abs()
    # print(f"Max Diff Seq: {diff_seq.max().item():.6f}")

    if diff_x.max() > 1e-4:
        print("X mismatches signifcantly!")
        # Print first few elements to see the nature of divergence
        print("Python X[0, :5]:", x_py[0, :5])
        print("CUDA X[0, :5]:  ", x_cuda[0, :5])
        
    if diff_v.max() > 1e-4:
        print("V mismatches signifcantly!")

if __name__ == "__main__":
    run_comparison()

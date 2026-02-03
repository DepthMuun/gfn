
import torch
import torch.nn as nn
import math
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.cuda.ops import recurrent_manifold_fused, CUDA_AVAILABLE

def test_torus_wrapping():
    print("=== TESTING TOROIDAL WRAPPING (CUDA) ===")
    
    if not CUDA_AVAILABLE:
        print("[FAIL] CUDA Extension not available. Cannot test kernel wrapping.")
        return

    B, T, D = 1, 100, 4
    # Create inputs that drift FAR beyond 2PI
    # x0 = 0, v normalized to 1.0 inside kernel
    # dt = 0.1 => dx/step = 0.1
    # x_final = 100 * 0.1 = 10.0 > 2PI (6.28)
    
    x_state = torch.zeros(B, D).cuda()
    v_state = torch.ones(B, D).cuda() * 50.0 # Direction matters, mag normalized
    forces = torch.zeros(B, T, D).cuda()
    
    # Dummy Params
    U = torch.randn(1*D, D, 16).cuda() * 0.01 
    W = torch.randn(1*D, D, 16).cuda() * 0.01
    dt = 0.1
    dt_scales = torch.ones(1).cuda()
    forget_rates = torch.zeros(1).cuda()
    
    # KERNEL CALL (EUCLIDEAN, topology=0)
    print("\n--- Running EUCLIDEAN Check ---")
    res_euc = recurrent_manifold_fused(
        x_state.clone(), v_state.clone(), forces, U, W, 
        dt, dt_scales, forget_rates, num_heads=1,
        topology=0 # Euclidean
    )
    if res_euc is None:
        print("[FAIL] Kernel returned None")
        return
        
    x_seq_euc = res_euc[2]
    max_val_euc = x_seq_euc.max().item()
    print(f"Euclidean Max Value: {max_val_euc:.2f}")
    if max_val_euc < 6.28:
        print("[WARN] Euclidean did not drift far enough to test wrapping.")
    
    # KERNEL CALL (TORUS, topology=1)
    print("\n--- Running TORUS Check ---")
    res_tor = recurrent_manifold_fused(
        x_state.clone(), v_state.clone(), forces, U, W, 
        dt, dt_scales, forget_rates, num_heads=1,
        topology=1 # Torus
    )
    
    x_seq_tor = res_tor[2]
    max_val_tor = x_seq_tor.max().item()
    min_val_tor = x_seq_tor.min().item()
    
    print(f"Torus Max Value: {max_val_tor:.4f}")
    print(f"Torus Min Value: {min_val_tor:.4f}")
    
    # Verification
    # Should be within [0, 2*PI)
    TWO_PI = 2 * math.pi
    if max_val_tor > TWO_PI + 0.1: # Tolerance
        print("[FAIL] Torus wrapping FAILED. Value exceeded 2PI.")
    elif min_val_tor < -0.1:
        print("[FAIL] Torus wrapping FAILED. Value below 0.")
    elif max_val_tor < 1.0:
         print("[WARN] Torus didn't move much (unexpected for v=50).")
    else:
        print("[PASS] Torus wrapping SUCCESSFUL (Bounded [0, 2PI)).")
        
        # Check that it differs from Euclidean
        diff = torch.abs(x_seq_euc - x_seq_tor).mean()
        if diff > 1.0:
            print(f"[PASS] Trajectories diverged significantly (Mean Diff: {diff:.2f})")
        else:
            print(f"[FAIL] Trajectories identical! Wrapping did not happen.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_torus_wrapping()
    else:
        print("Skipping CUDA test (No GPU)")

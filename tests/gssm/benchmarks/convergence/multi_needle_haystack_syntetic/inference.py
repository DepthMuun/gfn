#!/usr/bin/env python3
"""
inference.py — MNIAH Generalization Test
Tests if a model trained on K=2 needles can generalize to K=3 or more.
"""

import sys
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt

# ── Bootstrap ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]  # ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

PI = math.pi

def make_custom_batch(batch_size, seq_len, num_needles, device):
    """
    Standalone batch generator for inference.
    """
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y_class = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    all_positions = []
    
    for i in range(batch_size):
        # Sample K unique positions
        lo = max(1, int(0.05 * seq_len))
        hi = max(lo + num_needles + 1, int(0.95 * seq_len))
        pool = torch.randperm(hi - lo, device=device)[:num_needles]
        positions = sorted((pool + lo).tolist())
        all_positions.append(positions)
        
        for p in positions:
            x[i, p] = 1
            
        # Target flips to 1 only after the LAST needle
        y_class[i, positions[-1]:] = 1
        
    return x, y_class, all_positions

def run_inference(model_path, num_needles=3, seq_len=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg_dict = checkpoint['config']
    
    # Construcción exacta replicando run.py
    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=2,
        dim=cfg_dict['dim'],
        depth=cfg_dict['depth'],
        heads=cfg_dict['heads'],
        integrator=cfg_dict['integrator'],
        impulse_scale=cfg_dict['impulse_scale'],
        holographic=True,
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"[Inference] Model loaded. Testing generalization -> K={num_needles} needles")
    
    x, y_class, all_pos = make_custom_batch(1, seq_len, num_needles, device)
    
    with torch.no_grad():
        # Forward pass returning metrics for visualization
        _, _, metrics = model(x)
        x_seq = metrics['x_seq']  # [B, L, H, HD]
        # x_seq is the state of the last layer if not specified, 
        # but ManifoldModel returns the sequence of hidden states.
        # Actually, ManifoldModel forward returns (output, final_v, metrics)
        # and metrics['x_seq'] is [B, L, H, HD]
        
        # Average across heads to get a single 1D torus position (angle)
        angles = x_seq[0].mean(dim=1)  # [L, HD] -> but we want a scalar angle per token
        # In this task, the model is likely using a 1D torus (toroidal geometry).
        # We'll take the mean across all embedding dimensions to visualize the "phase".
        phase = angles.mean(dim=-1).cpu().numpy()
        
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(phase, label='Manifold State (Phase)', color='royalblue', lw=2)
    
    # Draw needles
    for p in all_pos[0]:
        plt.axvline(x=p, color='red', linestyle='--', alpha=0.6, label='Needle' if p == all_pos[0][0] else "")
        plt.text(p, plt.ylim()[1]*0.9, '1', color='red', fontweight='bold', ha='center')
        
    # Draw ground truth flip
    last_needle = all_pos[0][-1]
    plt.axvspan(last_needle, seq_len-1, color='green', alpha=0.1, label='Target Region (1)')
    
    plt.title(f"MNIAH Generalization: Trained K={cfg_dict['num_needles']} | Test K={num_needles}")
    plt.xlabel("Sequence Position")
    plt.ylabel("Phase (internal state)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = HERE / f"generalization_k{num_needles}.png"
    plt.savefig(save_path)
    print(f"[Inference] Plot saved to {save_path}")
    
    # Textual confirmation
    print("\nState evolution at needle points:")
    for i, p in enumerate(all_pos[0]):
        val = phase[p]
        print(f"  Needle {i+1} at pos {p:>3}: state = {val:.4f}")
    
    print(f"  Final state (pos {seq_len-1}): {phase[-1]:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/mniah_model_final.pt')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--len', type=int, default=128)
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = HERE / model_path
        
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    run_inference(model_path, num_needles=args.k, seq_len=args.len)

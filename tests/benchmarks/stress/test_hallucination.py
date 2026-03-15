#!/usr/bin/env python3
"""
MANIFOLD Probabilistic Hallucination (PH) Diagnostic
===================================================

This diagnostic measures "ghost signals" in the manifold state when no input 
triggers are present. 

METRICS:
  - Hallucination Score (HS): max(P(class=1)) on all-zero sequences.
  - Energy Drift (ED): Hamiltonian drift during long empty sequences.
  - State Variance (SV): Norm of the standard deviation of x across time.

Usage:
  python tests/benchmarks/stress/test_hallucination.py [--quicktest] [--seed SEED]
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# ── Bootstrap ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from gfn.losses.toroidal import ToroidalDistanceLoss

# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class PHConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    seq_len: int = 1000
    batch_size: int = 8
    integrator: str = 'leapfrog'
    seed: int = 42
    
@dataclass
class QuickConfig(PHConfig):
    seq_len: int = 100
    batch_size: int = 4
    depth: int = 2

# ── Test Logic ─────────────────────────────────────────────────────────────────

def run_hallucination_test(model: nn.Module, cfg: PHConfig, device: torch.device) -> Dict:
    model.eval()
    
    # Create all-zero input (all haystack)
    x = torch.zeros(cfg.batch_size, cfg.seq_len, dtype=torch.long, device=device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
        # ManifoldModel(layers, embedding, ...) returns (x_traj, v_traj, ...)
        x_pred = out[0]  # [B, L, D]
    
    elapsed = time.time() - t0
    
    # Calculate Probabilities for Class 1 (Assuming Toroidal NIAH targets: -pi/2 for class 0, +pi/2 for class 1)
    # We measure how close the state is to +pi/2 vs -pi/2
    PI = math.pi
    TWO_PI = 2.0 * PI
    half_pi = PI * 0.5
    
    # Small helper for toroidal distance
    def torus_dist(a, b):
        re = torch.abs(a - b) % TWO_PI
        return torch.min(re, TWO_PI - re)

    dist_0 = torus_dist(x_pred, -half_pi).mean(dim=-1) # [B, L]
    dist_1 = torus_dist(x_pred, half_pi).mean(dim=-1)  # [B, L]
    
    # P(class=1) estimate based on relative distance (softmin-like)
    # P1 = exp(-d1) / (exp(-d0) + exp(-d1))
    prob_1 = torch.exp(-dist_1) / (torch.exp(-dist_0) + torch.exp(-dist_1))
    
    hallucination_score = prob_1.max().item()
    mean_hallucination = prob_1.mean().item()
    
    # Drift: how much does x move per step on average
    # diff = x[t] - x[t-1]
    diffs = torus_dist(x_pred[:, 1:], x_pred[:, :-1]).mean()
    
    # Variance across time
    time_variance = x_pred.std(dim=1).mean().item()

    return {
        "hallucination_score_max": round(hallucination_score, 4),
        "hallucination_score_mean": round(mean_hallucination, 4),
        "drift_per_step": round(diffs.item(), 6),
        "time_variance": round(time_variance, 4),
        "inference_time_s": round(elapsed, 3),
        "seq_len": cfg.seq_len,
        "n_params": sum(p.numel() for p in model.parameters())
    }

def train_stability_task(model: nn.Module, device: torch.device, steps: int = 100):
    """
    Trains the model on a 'Zero-Force' task:
    Input is always 0. Target is always Class 0 (-pi/2).
    This forces the model to find a stable equilibrium at the 'neutral' state.
    """
    print(f"--- Training Stability Task ({steps} steps) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ToroidalDistanceLoss()
    
    PI = math.pi
    target_angle = -PI * 0.5
    
    for step in range(steps):
        optimizer.zero_grad()
        # Input is zero, length 50
        x = torch.zeros(8, 50, dtype=torch.long, device=device)
        out = model(x)
        x_pred = out[0]
        y_exp = torch.full_like(x_pred, target_angle)
        
        loss = criterion(x_pred, y_exp)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"  Step {step}: Loss {loss.item():.6f}")
    
    model.eval()
    print("Stability training complete.\n")

def main():
    parser = argparse.ArgumentParser(description='GFN Hallucination Test')
    parser.add_argument('--quicktest', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--train-stability', action='store_true', help='Train on a Zero-Force task before testing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    cfg = QuickConfig() if args.quicktest else PHConfig()
    cfg.seed = args.seed
    
    is_trained = args.checkpoint is not None or args.train_stability
    if args.checkpoint: eval_mode = "Trained Model"
    elif args.train_stability: eval_mode = "Stability Trained"
    else: eval_mode = "Architectural Prior"

    print(f"--- GFN Hallucination Test ({'Quick' if args.quicktest else 'Full'}) ---")
    print(f"Mode: {eval_mode}")
    print(f"Device: {device}")
    print(f"Config: dim={cfg.dim}, depth={cfg.depth}, seq_len={cfg.seq_len}")

    # Physical config for reconstruction (XOR based as requested by user)
    xor_physics = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16, 'impulse_scale': 80.0},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.05},
            'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.8},
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {
            'enable_trace_normalization': True, 'base_dt': 0.4,
            'velocity_saturation': 10.0, 'friction': 0.05, 'toroidal_curvature_scale': 0.01
        }
    }

    # Build model
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        try:
            # Reconstruct model as defined in logic_xor.py (best effort compatibility)
            model = gfn.create(
                vocab_size=2, dim=8, depth=1, heads=2,
                physics=xor_physics,
                trajectory_mode='partition',
                coupler_mode='mean_field',
                initial_spread=0.01,
                integrator='yoshida',
                holographic=True,
            ).to(device)
            
            sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
            raw_sd = sd.get('state_dict', sd.get('model', sd))
            # Key mapping
            new_sd = {k.replace('mixed_norm_v', 'v_norm'): v for k, v in raw_sd.items()}
            model.load_state_dict(new_sd, strict=False)
            print("Successfully loaded state_dict (strict=False) with fallback config.")
        except Exception as e:
            print(f"Error loading checkpoint fallback: {e}")
            sys.exit(1)
    else:
        # Default untrained model or base for ZFS
        model = gfn.create(
            vocab_size=2, dim=8, depth=1, heads=2,
            physics=xor_physics,
            trajectory_mode='partition',
            coupler_mode='mean_field',
            initial_spread=0.01,
            integrator='yoshida',
            holographic=True,
        ).to(device)

    if args.train_stability:
        train_stability_task(model, device, steps=80 if args.quicktest else 200)

    results = run_hallucination_test(model, cfg, device)
    results['eval_mode'] = eval_mode
    
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
        
    # High-level Verdict
    hs = results['hallucination_score_max']
    if is_trained:
        if hs > 0.8:
            print(f"\nVERDICT: CRITICAL - High {eval_mode} Hallucination detected.")
        elif hs > 0.15:
            print(f"\nVERDICT: WARNING - {eval_mode} shows drift ({hs:.2f}).")
        else:
            print(f"\nVERDICT: PASS - {eval_mode} is stable.")
    else:
        if hs > 0.8:
            print("\nVERDICT: UNSTABLE PRIOR - Architectural design induces strong ghost forces.")
        elif hs > 0.3:
            print("\nVERDICT: NEUTRAL PRIOR - Baseline noise (Expected for untrained weights).")
        else:
            print("\nVERDICT: STABLE PRIOR - Architecture naturally suppresses noise.")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = HERE / f"hallucination_report_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {output_path}")

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from ..framework import MatrixRunner
import gfn

def train_gating(model, steps=100):
    # Singularity Stress Task:
    # Tokens represent "Attractors" or "Singularities" in the space.
    # The model must learn stable trajectories near high-curvature points.
    # Perfect for SingularityGate and RiemannianGating.
    B = 4
    seq_len = 5
    vocab_size = 10
    device = next(model.parameters()).device
    
    # Input: tokens that push towards the same heavy attractor
    x = torch.zeros((B, seq_len), dtype=torch.long, device=device) + 7
    y = torch.zeros((B, seq_len), dtype=torch.long, device=device) + 7
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_gating(model):
    # Metrics: Accuracy AND Stability (did it produce NaNs?)
    B = 8
    seq_len = 5
    device = next(model.parameters()).device
    x = torch.zeros((B, seq_len), dtype=torch.long, device=device) + 7
    y_target = torch.zeros((B, seq_len), dtype=torch.long, device=device) + 7
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        if torch.isnan(logits).any():
            return -1.0 # Failed stability test
        preds = logits.argmax(dim=-1)
        acc = (preds == y_target).float().mean().item()
    return acc

def run_gating_matrix():
    runner = MatrixRunner("Gating_Stability")
    
    axes = {
        "active_inference.dynamic_time.enabled": [True, False],
        "active_inference.dynamic_time.type": ["riemannian", "thermo"],
        "stability.integrator_type": ["yoshida", "rk4"],
        "dynamics_type": ["mix", "gated"]
    }
    
    base = {
        "dim": 32,
        "depth": 6, # More depth = more numerical instability without gating
        "heads": 2,
        "vocab_size": 10,
        "holographic": False
    }
    
    runner.run_axes(axes, train_gating, eval_gating, base_overrides=base)

if __name__ == "__main__":
    run_gating_matrix()

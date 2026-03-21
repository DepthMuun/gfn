import torch
import torch.nn as nn
import torch.optim as optim
from ..framework import MatrixRunner
import gfn

def train_xor(model, steps=100):
    # XOR Data: [0, 0] -> 0, [0, 1] -> 1, [1, 0] -> 1, [1, 1] -> 0
    # Map to tokens: 0, 1
    device = next(model.parameters()).device
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)
    y = torch.tensor([0, 1, 1, 0], device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        # Force encoding in GFN happens via embedding
        logits, _, _ = model(x)
        
        # XOR is a classification task, evaluate on the last state
        # logits shape: [B, L, V]
        final_logits = logits[:, -1, :] 
        loss = criterion(final_logits, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_xor(model):
    device = next(model.parameters()).device
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)
    y = torch.tensor([0, 1, 1, 0], device=device)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        final_logits = logits[:, -1, :]
        preds = final_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc

def run_xor_matrix():
    runner = MatrixRunner("XOR_Logic")
    
    # Define rotation axes for the matrix
    axes = {
        "dynamics_type": ["direct", "residual", "mix"],
        "integrator": ["yoshida", "rk4"],
        "topology_type": ["torus", "euclidean"]
    }
    
    # Base configuration for XOR
    base = {
        "dim": 32,
        "depth": 2,
        "heads": 2,
        "vocab_size": 2,
        "initial_spread": 0.05,
        "holographic": False
    }
    
    runner.run_axes(axes, train_xor, eval_xor, base_overrides=base)

if __name__ == "__main__":
    run_xor_matrix()

import torch
import torch.nn as nn
import torch.optim as optim
from ..framework import MatrixRunner
import gfn

def train_arithmetic(model, steps=150):
    # Sum task: [1, 1, 1] -> 3, [2, 0, 1] -> 3
    # Small numbers, sequence length 3
    B = 8
    seq_len = 3
    vocab_size = 10
    device = next(model.parameters()).device
    
    # Random sums
    x = torch.randint(0, 5, (B, seq_len), device=device)
    y = x.sum(dim=1).clamp(0, vocab_size - 1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        final_logits = logits[:, -1, :] 
        loss = criterion(final_logits, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_arithmetic(model):
    B = 16
    seq_len = 3
    vocab_size = 10
    device = next(model.parameters()).device
    x = torch.randint(0, 5, (B, seq_len), device=device)
    y = x.sum(dim=1).clamp(0, vocab_size - 1)
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        final_logits = logits[:, -1, :]
        preds = final_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc

def run_arithmetic_matrix():
    runner = MatrixRunner("Arithmetic_Sum")
    
    axes = {
        "dynamics_type": ["direct", "residual"],
        "integrator": ["yoshida", "rk4", "leapfrog"],
        "topology_type": ["torus", "euclidean"]
    }
    
    base = {
        "dim": 64,
        "depth": 3,
        "heads": 4,
        "vocab_size": 10,
        "holographic": False
    }
    
    runner.run_axes(axes, train_arithmetic, eval_arithmetic, base_overrides=base)

if __name__ == "__main__":
    run_arithmetic_matrix()

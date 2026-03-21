import torch
import torch.nn as nn
import torch.optim as optim
from ..framework import MatrixRunner
import gfn

def train_hierarchical(model, steps=200):
    # Hierarchical dependency task:
    # Tokens: [GroupA, Item1_A, Item2_A] or [GroupB, Item1_B, Item2_B]
    # The model must learn that the context of GroupA restricts the valid Item space.
    # Perfect for HierarchicalGeometry and Hyperbolic spaces.
    B = 8
    seq_len = 3
    vocab_size = 10
    
    # Simple grammar: 0 -> {2, 3}, 1 -> {4, 5}
    # [0, 2, 2], [0, 3, 3], [1, 4, 4], [1, 5, 5]
    x = torch.zeros((B, seq_len), dtype=torch.long, device=next(model.parameters()).device)
    y = torch.zeros((B, seq_len), dtype=torch.long, device=next(model.parameters()).device)
    
    for i in range(B):
        group = i % 2
        x[i, 0] = group
        item = 2 + (group * 2) + (torch.randint(0, 2, (1,)).item())
        x[i, 1] = item
        x[i, 2] = item
        # Targets are shifted for autoregressive learning
        y[i, 0] = x[i, 1]
        y[i, 1] = x[i, 2]
        y[i, 2] = 9 # End token
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        # Sequence-wide loss
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_hierarchical(model):
    # Similar to train but only check accuracy on the second token (the one that depends on the first)
    B = 16
    seq_len = 3
    device = next(model.parameters()).device
    x = torch.zeros((B, seq_len), dtype=torch.long, device=device)
    y_target = torch.zeros(B, dtype=torch.long, device=device)
    
    for i in range(B):
        group = i % 2
        x[i, 0] = group
        item = 2 + (group * 2) + (torch.randint(0, 2, (1,)).item())
        x[i, 1] = item # Provide the item as context
        y_target[i] = item # We expect the model to predict the second token correctly
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        # Accuracy on the transition from token 0 to 1
        preds = logits[:, 0, :].argmax(dim=-1)
        acc = (preds == y_target).float().mean().item()
    return acc

def run_hierarchical_matrix():
    runner = MatrixRunner("Hierarchical_Tree")
    
    axes = {
        "topology_type": ["euclidean", "hyperbolic", "hierarchical"],
        "dynamics_type": ["direct", "residual"],
        "integrator": ["yoshida", "leapfrog"]
    }
    
    base = {
        "dim": 64,
        "depth": 4,
        "heads": 4,
        "vocab_size": 10,
        "holographic": False
    }
    
    runner.run_axes(axes, train_hierarchical, eval_hierarchical, base_overrides=base)

if __name__ == "__main__":
    run_hierarchical_matrix()

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from tests.benchmarks.matrix.framework import MatrixRunner
import gfn

def train_structural(model, steps=100):
    # Structural ROI Task:
    # Character-level prediction on a tiny sequence.
    # Tests how different embeddings and readouts affect convergence speed.
    
    device = next(model.parameters()).device
    
    # Tiny data: "abc" -> "bcd"
    chars = "abcdefghijklmnopqrstuvwxyz"
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    x = torch.tensor([[char_to_id['a'], char_to_id['b'], char_to_id['c']]], device=device)
    y = torch.tensor([[char_to_id['b'], char_to_id['c'], char_to_id['d']]], device=device)
    
    # Pre-initialize projection if needed
    with torch.no_grad():
        logits, _, _ = model(x)
        if logits.dim() == 4:
            V_actual = logits.shape[-1] * model.config.heads
        else:
            V_actual = logits.shape[-1]
            
        if V_actual != 26:
            model.struct_proj = nn.Linear(V_actual, 26).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 100.0
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        
        # logits: [B, S, ...]
        B, S = logits.shape[:2]
        if logits.dim() == 4: # [B, S, H, HD]
            logits = logits.reshape(B, S, -1) 
        
        # Flatten for CrossEntropy [B*S, V]
        V_actual = logits.shape[-1]
        logits_flat = logits.reshape(B * S, V_actual)
        y_flat = y.reshape(B * S)
        
        # If we are in holographic/Identity mode, logits_flat might be [B*S, D]
        # and y_flat is [B*S]. We MUST ensure they match for the NLL loss
        # chosen in this micro-task (NLL). 
        # In a real scenario we'd use MSE or similar, but let's keep it simple.
        if V_actual != 26: 
             logits_flat = model.struct_proj(logits_flat)

        loss = criterion(logits_flat, y_flat)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_structural(model):
    device = next(model.parameters()).device
    chars = "abcdefghijklmnopqrstuvwxyz"
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    x = torch.tensor([[char_to_id['a'], char_to_id['b'], char_to_id['c']]], device=device)
    y = torch.tensor([[char_to_id['b'], char_to_id['c'], char_to_id['d']]], device=device)
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        # Robust flattening for comparison
        if logits.dim() == 4:
             logits = logits.reshape(logits.shape[0], logits.shape[1], -1)
        
        # If Holographic mode, logits might be [1, 3, D] and y is [1, 3]
        # We take the argmax over the latent dim or use a projection
        if logits.shape[-1] != 26:
            if hasattr(model, 'struct_proj'):
                logits = model.struct_proj(logits.reshape(-1, logits.shape[-1])).reshape(logits.shape[0], logits.shape[1], -1)
            else:
                # fallback for eval if proj not created (should not happen if trained)
                return 0.0

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc

def run_structural_matrix():
    runner = MatrixRunner("Structural_ROI")
    
    # We test the 3 Readouts and the 3 Main Embedding types
    axes = {
        "embedding_mode": ["lookup", "binary", "linear", "siren"],
        "readout_type": ["categorical", "implicit", "identity"],
        "topology_type": ["euclidean", "torus"]
    }
    
    # Task to test Holographic (Identity) separately as it requires different supervision
    runner.run_axes(axes, train_structural, eval_structural, base_overrides={
        "dim": 64, "depth": 2, "heads": 2, "vocab_size": 26, "holographic": False
    })

    # Holographic test (Identity Readout)
    runner.run_axes({
        "embedding_mode": ["linear"],
        "topology_type": ["torus"]
    }, train_structural, eval_structural, base_overrides={
        "holographic": True, "dim": 24, "vocab_size": 26, "heads": 4
    })

if __name__ == "__main__":
    run_structural_matrix()

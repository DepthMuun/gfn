import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from tests.benchmarks.matrix.framework import MatrixRunner
import gfn

class TopologyAwareProjection(nn.Module):
    def __init__(self, dim, vocab_size, topology_type):
        super().__init__()
        self.topology_type = str(topology_type).lower().strip()
        in_dim = dim * 2 if self.topology_type == 'torus' else dim
        self.linear = nn.Linear(in_dim, vocab_size)
        
    def forward(self, x):
        if self.topology_type == 'torus':
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.linear(x)

def train_pooling(model, steps=300):
    device = next(model.parameters()).device
    batch_size = 8
    seq_len = 5
    vocab_size = 10
    topology = model.config.physics.topology.type
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.sum(dim=1) % vocab_size
    
    with torch.no_grad():
        _, _, info = model(x[:1])
        pool_res = info.get("plugin_results", [])
        d_agg = 0
        for res in pool_res:
            if isinstance(res, tuple) and len(res) >= 2:
                d_agg = res[0].shape[-1]
                break
        
        if d_agg > 0:
            model.pool_proj = TopologyAwareProjection(d_agg, vocab_size, topology).to(device)
            print(f"DEBUG INIT: created pool_proj TopologyAware({d_agg}, {vocab_size}, {topology})")
        else:
            logits, _, _ = model(x[:1])
            d_in = logits.shape[-1]
            if logits.dim() == 4: d_in = logits.shape[-1] * model.config.heads
            
            if d_in != vocab_size:
                model.fallback_proj = nn.Linear(d_in, vocab_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        _, _, info = model(x)
        pool_res = info.get("plugin_results", [])
        pooled_out = None
        for res in pool_res:
            if isinstance(res, tuple) and len(res) >= 2:
                # momentum might return a diff batch size if not careful, but plugin handles it
                pooled_out = res[0]
                break

        if pooled_out is not None:
            logits = model.pool_proj(pooled_out)
        else:
            logits, _, _ = model(x)
            if logits.dim() == 4:
                logits = logits[:, -1].reshape(logits.shape[0], -1)
            elif logits.dim() == 3:
                logits = logits[:, -1]
            
            if hasattr(model, 'fallback_proj'):
                logits = model.fallback_proj(logits)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"  Step {step}: Loss {loss.item():.4f}")
            
    return {"steps": steps, "loss": loss.item()}

def eval_pooling(model):
    device = next(model.parameters()).device
    batch_size = 16
    seq_len = 5
    vocab_size = 10
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.sum(dim=1) % vocab_size
    
    model.eval()
    with torch.no_grad():
        _, _, info = model(x)
        pool_res = info.get("plugin_results", [])
        pooled_out = None
        for res in pool_res:
            if isinstance(res, tuple) and len(res) >= 2:
                pooled_out = res[0]
                break

        if pooled_out is not None:
            logits = model.pool_proj(pooled_out)
        else:
            logits, _, _ = model(x)
            if logits.dim() == 4:
                logits = logits[:, -1].reshape(logits.shape[0], -1)
            elif logits.dim() == 3:
                logits = logits[:, -1]
            
            if hasattr(model, 'fallback_proj'):
                logits = model.fallback_proj(logits)
            
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc

def run_pooling_matrix():
    runner = MatrixRunner("Pooling_ROI")
    
    axes = {
        "pooling_type": ["hamiltonian", "hierarchical", "momentum", None],
        "topology_type": ["euclidean", "torus"],
        "readout_type": ["categorical", "implicit", "identity"]
    }
    
    base = {
        "dim": 64,
        "depth": 2,
        "heads": 2,
        "vocab_size": 10,
        "holographic": False
    }
    
    runner.run_axes(axes, train_pooling, eval_pooling, base_overrides=base)

if __name__ == "__main__":
    run_pooling_matrix()

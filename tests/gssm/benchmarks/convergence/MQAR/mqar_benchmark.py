"""
MANIFOLD MQAR (Multi-Query Associative Recall) Benchmark
=========================================================
Evaluates long-range associative recall performance using:
  - Torus topology
  - Yoshida integrator
  - Rich logging and Matplotlib visualization
"""
import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

# Standalone execution support
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from gfn.losses import ToroidalDistanceLoss

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================
PRODUCTION_PHYSICS_CONFIG = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',
        'coord_dim': 16,
        'impulse_scale': 80.0,
    },
    'readout': {'type': 'implicit', 'coord_dim': 16},
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
        'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8},
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
    'topology': {'type': 'torus'},
    'stability': {'base_dt': 0.4},   # Faster exploration
}

# ============================================================================
# MQAR DATASET
# ============================================================================
class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall Dataset.
    Sequence: [K1, V1, K2, V2, ..., Kn, Vn, SEP, Q1, Q2, ..., Qm]
    Target:   [ _,  _,  _,  _, ...,  _,  _,   _, Vq1, Vq2, ..., Vqm]
    """
    def __init__(self, num_pairs=8, num_queries=4, vocab_size=64, num_samples=10000):
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.sep_token = vocab_size - 1
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample unique keys (excluding SEP)
        keys = torch.randperm(self.vocab_size - 1)[:self.num_pairs]
        vals = torch.randint(0, self.vocab_size - 1, (self.num_pairs,))
        
        # Build sequence: [K1, V1, K2, V2, ..., Kn, Vn, SEP, Q1, Q2, ..., Qm]
        input_seq = []
        for i in range(self.num_pairs):
            input_seq.append(keys[i])
            input_seq.append(vals[i])
        
        input_seq.append(self.sep_token)
        
        # Queries
        query_indices = torch.randint(0, self.num_pairs, (self.num_queries,))
        queries = keys[query_indices]
        query_vals = vals[query_indices]
        
        input_seq.extend(queries)
        
        full_input = torch.tensor(input_seq, dtype=torch.long)
        
        # Map value tokens to angles using unified helper
        target_angles = token_to_angle(query_vals, self.vocab_size)
        
        return full_input, target_angles, query_vals, query_indices

def token_to_angle(token_idx: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Unified token-to-angle mapping to avoid off-by-one errors."""
    return (token_idx.float() / (vocab_size - 1)) * (2.0 * math.pi) - math.pi

def compute_retrieval_accuracy(x_pred: torch.Tensor, target_tokens: torch.Tensor, vocab_size: int) -> float:
    """
    Computes accuracy by finding the nearest token angle on the torus.
    x_pred: [B, num_queries, D] (only query output steps)
    target_tokens: [B, num_queries]
    """
    B, M, D = x_pred.shape
    # Precompute token angles using the unified helper
    token_angles = token_to_angle(torch.arange(vocab_size - 1), vocab_size).to(x_pred.device) # [V]
    
    # tor-dist between each pred [B, M, D] and all tokens [V]
    # We compare the first dimension of the torus (D=0)
    preds_theta = x_pred[:, :, 0] # [B, M]
    
    # Token angles [V] -> [1, 1, V]
    # Preds [B, M] -> [B, M, 1]
    dist = torch.abs(preds_theta.unsqueeze(-1) - token_angles.unsqueeze(0).unsqueeze(0))
    dist = torch.min(dist, 2.0 * math.pi - dist)
    
    pred_tokens = torch.argmin(dist, dim=-1) # [B, M]
    
    acc = (pred_tokens == target_tokens).float().mean().item()
    return acc

# ============================================================================
# TRAINING
# ============================================================================
def train_mqar(max_steps=100000, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path("tests/benchmarks/convergence/MQAR/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_pairs = 8
    num_queries = 4
    vocab_size = 64
    dim = 32
    
    dataset = MQARDataset(num_pairs=num_pairs, num_queries=num_queries, vocab_size=vocab_size)
    
    # 1. Create Model (Synced with User XOR Edit)
    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=vocab_size,
        dim=dim,
        depth=4,
        heads=4,
        integrator='yoshida',
        dynamics_type='direct',
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps)
    criterion = ToroidalDistanceLoss()
    
    history = {"loss": [], "acc": []}
    
    console.print(f"\n[bold cyan]MANIFOLD MQAR Benchmark[/]")
    console.print(f"Device: {device} | Pairs: {num_pairs} | Queries: {num_queries} | Integrator: Yoshida")
    
    pbar = tqdm(range(max_steps), desc="Training MQAR")
    best_acc = 0.0
    
    # Optimized Data Loading
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(loader)
    
    for step in pbar:
        # Get Batch from persistent iterator
        try:
            x, target_angles, target_tokens, q_idx = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, target_angles, target_tokens, q_idx = next(data_iter)
            
        x, target_angles, target_tokens = x.to(device), target_angles.to(device), target_tokens.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(x) # [B, T, D]
        
        # We only care about the last M tokens (the queries)
        query_outputs = output[0][:, -num_queries:, :] # [B, M, D]
        
        # BUG FIX: Only optimize the manifold dimension used for classification (dim=0)
        # Expansion should match [B, M, 1] if we only use one dim, or we keep [B, M, D]
        # and ensure the target is only meaningful on dim 0.
        y_expanded = target_angles.unsqueeze(-1).expand_as(query_outputs[:, :, :1])
        
        loss = criterion(query_outputs[:, :, :1], y_expanded)
        
        if torch.isnan(loss):
            console.print("[red]NaN Loss detected, skipping...[/]")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        with torch.no_grad():
            acc = compute_retrieval_accuracy(query_outputs, target_tokens, vocab_size)
            
        history["loss"].append(loss.item())
        history["acc"].append(acc)
        
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.1f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_dir / "mqar_best_model.pt")
            
    # 2. Saving Results
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)
        
    # 3. Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history['loss'], color='tab:red', alpha=0.3, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history['acc'], color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'MQAR Convergence (Yoshida + Torus)\nPairs={num_pairs}, Queries={num_queries}')
    fig.tight_layout()
    plt.savefig(output_dir / "convergence_plot.png")
    console.print(f"\n[bold green]Training Complete![/] Plots saved to {output_dir}")
    console.print(f"Best Retrieval Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_mqar()

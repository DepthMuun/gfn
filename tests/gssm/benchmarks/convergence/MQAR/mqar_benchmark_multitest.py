"""
MANIFOLD MQAR Multi-Configuration Benchmark (Final Refined)
==========================================================
Geometric configurations comparison with 1D coordinate focus.
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
from mqar_benchmark import MQARDataset

console = Console()


def run_model_test(config_name, topo_type, riem_type, results_dir, max_steps=1000, batch_size=32, device='cuda'):
    console.print(f"\n[bold yellow]>>> Starting Test: {config_name}[/]")
    
    num_pairs, num_queries, vocab_size, dim = 6, 3, 64, 64
    is_torus = (topo_type == 'torus')
    
    dataset = MQARDataset(num_pairs=num_pairs, num_queries=num_queries, vocab_size=vocab_size)
    
    # 1D focus: coord_dim=1 maps tokens directly to a single coordinate
    p_cfg = {
        'topology': {'type': topo_type, 'riemannian_type': riem_type, 'riemannian_rank': 16},
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16, 'impulse_scale': 120.0},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.1},
            'singularities': {'enabled': True, 'strength': 30.0, 'threshold': 0.8},
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'stability': {'base_dt': 0.4}
    }
    
    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=vocab_size, dim=dim, depth=1, heads=1,
        integrator='yoshida',
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # DECLARATIVE LOSS: gfn.loss handles component selection automatically.
    # Note: 'toroidal' is our custom distance loss for MQAR angles.
    criterion = gfn.loss(['toroidal' if is_torus else 'kinetic'], p_cfg)
    
    # UNIFIED BENCHMARKING
    evaluator = gfn.benchmark('retrieval', vocab_size=vocab_size, topology=topo_type)
    
    history = {"loss": [], "acc": []}
    best_acc = 0.0
    safe_name = config_name.replace("+", "_").replace(" ", "_")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_iter = iter(loader)
    
    pbar = tqdm(range(max_steps), desc=f" {config_name}", leave=False)
    for step in pbar:
        try:
            data = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader); data = next(loader_iter)
            
        x, target_angles, target_tokens = data[0].to(device), data[1].to(device), data[2].to(device)
        
        optimizer.zero_grad()
        output = model(x)
        logits = output[0][:, -num_queries:, 0:1] # [B, M, 1]
        
        if is_torus:
            # Angles are in [-pi, pi]
            loss = criterion(logits, target_angles.unsqueeze(-1))
        else:
            # Basic coord match
            loss = criterion(logits, target_angles.unsqueeze(-1))
            
        if torch.isnan(loss): continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            # Unified accuracy calculation via evaluator helper
            acc = evaluator(logits, target_tokens)
        
        history["loss"].append(loss.item())
        history["acc"].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), results_dir / f"mqar_best_{safe_name}.pt")
            
        if step % 20 == 0:
            pbar.set_postfix(acc=f"{acc*100:.1f}%", loss=f"{loss.item():.4f}")
            
    return history

def run_multitest(max_steps=1200, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path("tests/benchmarks/convergence/MQAR/results/multitest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tests = [
        ("Torus+Reactive", "torus", "reactive"),
        ("Torus+Hierarchical", "torus", "hierarchical"),
        ("Torus+Ricci", "torus", "ricci"),
        ("Euclidean+Reactive", "euclidean", "reactive"),
        ("Hyperbolic+Analytical", "hyperbolic", "analytical"),
    ]
    
    all_histories = {}
    for name, topo, riem in tests:
        try:
            history = run_model_test(name, topo, riem, output_dir, max_steps=max_steps, batch_size=batch_size, device=device)
            all_histories[name] = history
        except Exception as e:
            console.print(f"[bold red]Failed {name}: {e}[/]"); import traceback; traceback.print_exc()
            
    with open(output_dir / "multitest_results.json", "w") as f: json.dump(all_histories, f)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    for name, hist in all_histories.items():
        window = 30
        acc, loss = hist['acc'], hist['loss']
        smoothed_acc = [sum(acc[max(0, i-window):i+1]) / len(acc[max(0, i-window):i+1]) for i in range(len(acc))]
        smoothed_loss = [sum(loss[max(0, i-window):i+1]) / len(loss[max(0, i-window):i+1]) for i in range(len(loss))]
        ax1.plot(smoothed_acc, label=name); ax2.plot(smoothed_loss, label=name)
        
    ax1.set_title("MQAR Retrieval Accuracy (Smoothed)"); ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("MQAR Training Loss (Log Scale)"); ax2.set_xlabel("Steps"); ax2.set_ylabel("Loss"); ax2.set_yscale('log'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_dir / "mqar_comparison.png")
    
    table = Table(title="MQAR Geometric Performance Summary")
    table.add_column("Configuration", style="cyan"); table.add_column("Final Acc", justify="right"); table.add_column("Peak Acc", justify="right"); table.add_column("Min Loss", justify="right")
    for name, hist in all_histories.items():
        if not hist['acc']: continue
        table.add_row(name, f"{hist['acc'][-1]*100:.1f}%", f"{max(hist['acc'])*100:.1f}%", f"{min(hist['loss']):.6f}")
    console.print("\n", table)
    console.print(f"\n[bold green]Multi-test Complete![/] Saved to {output_dir}")

if __name__ == "__main__":
    run_multitest()

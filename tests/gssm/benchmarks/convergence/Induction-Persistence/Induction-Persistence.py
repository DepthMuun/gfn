#!/usr/bin/env python3
import torch
import torch.nn as nn
import time
import sys
import math
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 0. Setup Paths & Environment
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

console = Console()
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PI = math.pi

@dataclass
class LangContextConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    integrator: str = 'yoshida'  # High-stability symplectic
    dynamics_type: str = 'direct' # Optimized for O(1) scaling
    train_seq_len: int = 128
    train_batch_size: int = 16
    train_steps: int = 250
    eval_batch_size: int = 8
    eval_lengths: List[int] = None
    
    def __post_init__(self):
        if self.eval_lengths is None:
            # Test deep context scaling
            self.eval_lengths = [128, 512, 1024, 2048, 4096, 8192]

def generate_noise_needle_batch(batch_size: int, seq_len: int, device: torch.device):
    """
    Task: Needle-in-a-Haystack (Induction).
    The model must latch onto the 'needle' (token 1) and maintain that state 
    on the manifold despite distractor noise (token 2-9).
    """
    # 0 is padding/null, 1 is needle, 2-9 are distractors
    x = torch.randint(2, 10, (batch_size, seq_len), dtype=torch.long, device=device)
    y_class = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # Place needle in first 10% of sequence
        pos = torch.randint(1, max(2, seq_len // 10), (1,)).item()
        x[i, pos] = 1
        y_class[i, pos:] = 1 # State latch after needle
        
    y_angle = (y_class.float() * 2.0 - 1.0) * (PI * 0.5)
    return x, y_angle, y_class

def compute_accuracy(x_pred: torch.Tensor, y_class: torch.Tensor) -> float:
    # Toroidal Nearest-Neighbor classification
    half_pi = PI * 0.5
    dist_pos = 1.0 - torch.cos(x_pred - half_pi)
    dist_neg = 1.0 - torch.cos(x_pred + half_pi)
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
    return (preds == y_class).float().mean().item()

def run_language_context_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quicktest", action="store_true")
    args = parser.parse_args()

    cfg = LangContextConfig()
    if args.quicktest:
        cfg.train_steps = 50
        cfg.eval_lengths = [128, 512]
        cfg.dim = 64
        cfg.depth = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Model Setup (Production Optimized)
    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=10,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        integrator=cfg.integrator,
        dynamics_type=cfg.dynamics_type,
        mixer_type='low_rank',
        holographic=True # Internal readout for maximum O(1) efficiency
    ).to(device)

    console.print(f"\n[bold cyan]MANIFOLD Long-Context Consolidation[/]")
    console.print(f"Arch: [yellow]{cfg.dim}D, {cfg.depth}L[/] | Integrator: [yellow]{cfg.integrator}[/]")
    console.print(f"Device: [green]{device}[/] | Task: [magenta]Induction-Persistence[/]")

    # 2. Induction Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Using specialized Toroidal loss via simplified proxy
    history = {"loss": [], "acc": []}
    
    model.train()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Training...", total=cfg.train_steps)
        
        for step in range(cfg.train_steps):
            x, y_angle, y_class = generate_noise_needle_batch(cfg.train_batch_size, cfg.train_seq_len, device)
            
            optimizer.zero_grad()
            out = model(x)
            x_pred = out[0] # Holistic state
            
            # Toroidal distance loss
            loss = (1.0 - torch.cos(x_pred - y_angle.unsqueeze(-1))).mean()
            loss.backward()
            optimizer.step()
            
            acc = compute_accuracy(x_pred, y_class)
            history["loss"].append(loss.item())
            history["acc"].append(acc)
            
            progress.update(task, advance=1, description=f"Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    # 3. Scaling Evaluation
    model.eval()
    eval_results = []
    
    table = Table(title="Context Scaling Verification")
    table.add_column("Length", justify="right", style="cyan")
    table.add_column("VRAM (MB)", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="magenta")
    table.add_column("Lat/Token (ms)", justify="right")
    
    with torch.no_grad():
        for L in cfg.eval_lengths:
            x, y_angle, y_class = generate_noise_needle_batch(cfg.eval_batch_size, L, device)
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            t0 = time.time()
            out = model(x)
            t1 = time.time()
            
            mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == 'cuda' else 0
            acc = compute_accuracy(out[0], y_class)
            latency = (t1 - t0) / (L * cfg.eval_batch_size) * 1000
            
            eval_results.append({"L": L, "mem": mem, "acc": acc, "lat": latency})
            table.add_row(f"{L}", f"{mem:.1f}", f"{acc*100:.1f}%", f"{latency:.3f}")
    
    console.print(table)
    
    # 4. Save Artifacts
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump({"train_history": history, "eval_scaling": eval_results}, f)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Convergence
    ax1.plot(history['acc'], label='Accuracy', color='blue')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Induction Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scaling
    lengths = [r['L'] for r in eval_results]
    vram = [r['mem'] for r in eval_results]
    latency = [r['lat'] for r in eval_results]
    
    line1 = ax2.plot(lengths, vram, marker='o', color='green', label='VRAM (MB)')
    ax2_rhs = ax2.twinx()
    line2 = ax2_rhs.plot(lengths, latency, marker='x', color='red', label='Latency/Token (ms)')
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('VRAM (MB)', color='green')
    ax2_rhs.set_ylabel('Latency/Token (ms)', color='red')
    ax2.set_title('O(1) Scaling Verification')
    ax2.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scaling_analysis.png")
    
    model_path = OUTPUT_DIR / "lang_context_best.bin"
    torch.save(model.state_dict(), model_path)
    
    console.print(f"\n[bold green]Consolidation Complete![/]")
    console.print(f"Artifacts saved to: [blue]{OUTPUT_DIR}[/]")
    
    # O(1) Verification Check
    if len(eval_results) > 1:
        vram_increase = eval_results[-1]['mem'] - eval_results[0]['mem']
        if vram_increase < 5.0: # Very strict bound for O(1)
            console.print(f"[bold green]Verified: O(1) Memory Complexity (VRAM delta: {vram_increase:.2f}MB)[/]")
        else:
            console.print(f"[bold yellow]Warning: VRAM increased by {vram_increase:.2f}MB. Check for hidden caches.[/]")

if __name__ == "__main__":
    run_language_context_benchmark()

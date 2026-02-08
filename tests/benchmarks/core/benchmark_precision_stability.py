"""
Professional Precision & Numerical Stability Benchmark (v2.6.5)
=============================================================

Evaluates geometric stability under different numerical formats:
- FP32 (Full Precision)
- BF16 (Brain Float - Dynamic Range)
- FP16 (Half Precision - Standard)

Monitors for NaNs, Infs, and gradient explosions in the physics engine.
"""

import torch
import torch.nn as nn
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN Components
from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

class ParityTask:
    """Standard stability probe: Toroidal Parity Check."""
    def __init__(self, length=32):
        self.length = length
        
    def generate(self, batch_size, device='cpu'):
        x = torch.randint(0, 2, (batch_size, self.length), device=device)
        y = torch.cumsum(x, dim=1) % 2
        return x, y

def evaluate_stability(dtype_name, dtype, device, steps=300):
    model = Manifold(
        vocab_size=2, dim=128, depth=4, heads=4,
        physics_config={'topology': 'torus', 'plasticity': 0.1}
    ).to(device)
    
    if dtype != torch.float16:
        model.to(dtype=dtype)
        
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    task = ParityTask()
    
    history = []
    scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None
    
    model.train()
    for _ in range(steps):
        x, y = task.generate(32, device=device)
        target = y.float().unsqueeze(-1).expand(-1, -1, 128) # Matching dim for stability test
        
        optimizer.zero_grad()
        
        try:
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, target.to(dtype=logits.dtype))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            if torch.isnan(loss) or torch.isinf(loss):
                return history, False
                
            history.append(loss.item())
            
        except RuntimeError:
            return history, False
            
    return history, True

def run_precision_benchmark():
    logger = ResultsLogger("precision_stability", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN PRECISION STABILITY AUDIT[/] (Manifold v2.6.5)\n")

    formats = {
        "FP32": torch.float32,
        "BF16": torch.bfloat16,
        "FP16": torch.float16
    }
    
    report_data = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        for name, dtype in formats.items():
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
                
            task_id = progress.add_task(f"Prying into {name}...", total=100)
            history, success = evaluate_stability(name, dtype, device)
            
            report_data.append({
                "Format": name,
                "Stability": "[green]STABLE[/]" if success else "[red]FAILED[/]",
                "Final Loss": history[-1] if history else float('nan'),
                "history": history
            })
            progress.update(task_id, completed=100)

    table = Table(title="Precision Stability Summary", box=None)
    table.add_column("Format")
    table.add_column("Status")
    table.add_column("Final Loss", justify="right")
    
    for r in report_data:
        table.add_row(r["Format"], r["Stability"], f"{r['Final Loss']:.4f}")
    
    console.print("\n", table)
    
    # Plotting
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#121212')
    
    for r in report_data:
        if r["history"]:
            ax.plot(r["history"], label=r["Format"], lw=2)
            
    ax.set_title("Training Stability by Precision Format", color='white', fontweight='bold')
    ax.set_xlabel("Steps", color='white')
    ax.set_ylabel("Loss", color='white')
    ax.legend()
    
    plt.tight_layout()
    logger.save_plot(fig, "stability_analysis.png")
    logger.save_json(report_data)
    
    console.print(f"\n[bold green][SUCCESS][/] Stability analysis complete. Logs in [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_precision_benchmark()

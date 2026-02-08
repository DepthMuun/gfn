"""
Professional Performance Benchmarks (v2.6.5)
===========================================

Comparative analysis of GFN vs Transformer vs Mamba:
- Memory scaling ($O(1)$ vs $O(N^2)$ vs $O(N)$)
- Training throughput (total seq/s)
- Forward/Backward pass breakdown
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import gc
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss
from tests.benchmarks.baselines import MicroGPT, MicroMamba
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

console = Console()

def measure_efficiency(model, batch_size, seq_len, device):
    """Measures Peak VRAM and Training Throughput."""
    model.train()
    x = torch.randint(0, 16, (batch_size, seq_len)).to(device)
    
    # 1. Warmup
    with torch.cuda.amp.autocast():
        for _ in range(5):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits.mean().backward()
    
    # 2. Timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        with torch.cuda.amp.autocast():
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits.mean().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tput = (20 * batch_size) / elapsed
    
    # 3. Peak VRAM
    def run_step():
        with torch.cuda.amp.autocast():
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits.mean().backward()
            
    vram = PerformanceStats.measure_peak_memory(model, run_step)
    
    return tput, vram

def run_performance_suite():
    logger = ResultsLogger("performance", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN PERFORMANCE AUDIT[/] (Manifold v2.6.5)\n")

    if device.type == 'cpu':
        console.print("[red]ERROR: Performance benchmarking requires CUDA.[/]")
        return

    # Model parameters
    dim, depth, heads, vocab = 256, 6, 4, 32
    
    # Physics configuration matching the reference
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.4}
    }
    
    models = {
        "Manifold-GFN ($O(1)$)": Manifold(
            vocab, dim, depth, heads,
            integrator_type='leapfrog',
            physics_config=physics_config,
            impulse_scale=80.0,
            holographic=True
        ).to(device),
        "Transformer ($O(N^2)$)": MicroGPT(vocab, dim, depth, heads).to(device)
    }
    
    # Optional Mamba if available
    try:
        models["Mamba ($O(N)$)"] = MicroMamba(vocab, dim, depth).to(device)
    except:
        console.print("[yellow]Notice: MicroMamba baseline skipped (not found).[/]")
        
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192]
    batch_size = 1 # Focus on sequence scaling
    report_data = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        perf_task = progress.add_task("Profiling Models...", total=len(models) * len(seq_lengths))
        for name, model in models.items():
            for L in seq_lengths:
                try:
                    gc.collect(); torch.cuda.empty_cache()
                    tput, vram = measure_efficiency(model, batch_size, L, device)
                    
                    report_data.append({
                        "Model": name,
                        "Length": L,
                        "Throughput": tput,
                        "VRAM": vram
                    })
                    progress.update(perf_task, advance=1, description=f"Profiling Models... [cyan]{name} L={L}[/]")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        console.print(f"  [red]{name} OOM at L={L}[/]")
                        progress.update(perf_task, advance=len(seq_lengths) - seq_lengths.index(L) - 1) # Advance for remaining lengths of this model
                        break
                    raise e

    # Summary Table
    table = Table(title="Performance Summary", box=None)
    table.add_column("Model")
    table.add_column("Seq Length", justify="right")
    table.add_column("Throughput (seq/s)", justify="right")
    table.add_column("VRAM (MB)", justify="right")
    
    for r in report_data:
        table.add_row(
            r["Model"], 
            str(r["Length"]), 
            f"{r['Throughput']:.1f}", 
            f"{r['VRAM']:.1f}"
        )
    console.print("\n", table)
    
    # Plotting
    logger.save_json(report_data)
    df = pd.DataFrame(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    
    # VRAM Scaling
    sns.lineplot(data=df, x="Length", y="VRAM", hue="Model", marker='o', lw=3, ax=ax1)
    ax1.set_title("Memory Complexity", color='white', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_ylabel("Peak VRAM (MB)", color='white')
    
    # Throughput Scaling
    sns.lineplot(data=df, x="Length", y="Throughput", hue="Model", marker='s', lw=3, ax=ax2)
    ax2.set_title("Training Throughput", color='white', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_ylabel("Sequences per Second", color='white')
    
    plt.tight_layout()
    logger.save_plot(fig, "performance_scaling.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Performance audit complete. Charts saved to [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_performance_suite()

if __name__ == "__main__":
    run_performance_suite()

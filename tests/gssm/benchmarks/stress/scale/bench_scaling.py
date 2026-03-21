"""
Professional Scaling Laws Benchmark (v2.6.5)
===========================================

Analyzes how Manifold scales with model dimension and depth.
Focuses on:
- Constant memory complexity ($O(1)$ w.r.t sequence length)
- Performance vs. parameter count
- Throughput scalability
"""

import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from gfn import Manifold
from gfn import RiemannianAdam
from gfn import CircularDistanceLoss, geodesic_regularization, hamiltonian_loss
from tests.benchmarks.infra.utils.logger import ResultsLogger, PerformanceStats

console = Console()

def measure_scale_metrics(name, cfg, device, seq_len=1024):
    """Measures VRAM and Speed for a specific geometry."""
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
    
    model = Manifold(
        vocab_size=64, 
        dim=cfg['dim'], 
        depth=cfg['depth'], 
        heads=cfg['heads'],
        integrator_type='leapfrog',
        impulse_scale=80.0,
        holographic=True
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randint(0, 64, (8, seq_len)).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(dummy_input)
        
    # Throughput
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10): model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tput = (10 * 8) / elapsed
    
    # Memory
    vram = PerformanceStats.measure_peak_memory(model, lambda: model(dummy_input))
    
    return {
        "Config": name,
        "Params (M)": params / 1e6,
        "VRAM (MB)": vram,
        "Throughput (seq/s)": tput
    }

def run_benchmark():
    logger = ResultsLogger("scaling", category="performance/scaling")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN ARCHITECTURAL SCALING AUDIT[/] (Manifold v2.6.5)\n")

    # Test Matrix
    configs = [
        {'dim': 128, 'depth': 2, 'heads': 2, 'name': 'XS (Tiny)'},
        {'dim': 128, 'depth': 4, 'heads': 2, 'name': 'S (Base)'},
        {'dim': 256, 'depth': 4, 'heads': 4, 'name': 'M (Medium)'},
        {'dim': 256, 'depth': 8, 'heads': 4, 'name': 'L (Large)'},
        {'dim': 512, 'depth': 6, 'heads': 8, 'name': 'XL (Pro)'},
        {'dim': 512, 'depth': 12, 'heads': 8, 'name': 'XXL (Ultra)'},
    ]
    
    report_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        scale_task = progress.add_task("Scaling up...", total=len(configs))
        
        for c in configs:
            progress.update(scale_task, description=f"Profiling {c['name']}")
            try:
                metrics = measure_scale_metrics(c['name'], c, device)
                report_data.append(metrics)
            except RuntimeError:
                console.print(f"  [red]OOM at {c['name']}[/]")
            progress.update(scale_task, advance=1)

    # Table
    table = Table(title="GFN Scaling Summary", box=None)
    table.add_column("Profile")
    table.add_column("Params (M)", justify="right")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Throughput (seq/s)", justify="right")
    
    for r in report_data:
        table.add_row(r["Config"], f"{r['Params (M)']:.1f}", f"{r['VRAM (MB)']:.1f}", f"{r['Throughput (seq/s)']:.1f}")
    
    console.print("\n", table)
    
    # Save & Plot
    logger.save_json(report_data)
    df = pd.DataFrame(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#121212')
    
    # Params vs Performance
    sns.barplot(data=df, x="Config", y="Params (M)", ax=axes[0], color='steelblue', alpha=0.8)
    ax0_twin = axes[0].twinx()
    sns.lineplot(data=df, x="Config", y="Throughput (seq/s)", ax=ax0_twin, marker='o', color='gold', lw=3)
    axes[0].set_title("Parameter vs Throughput Scaling", color='white', fontweight='bold')
    
    # Params vs VRAM
    sns.regplot(data=df, x="Params (M)", y="VRAM (MB)", ax=axes[1], scatter_kws={'s':100}, line_kws={'color':'red', 'ls':'--'})
    axes[1].set_title("Memory Efficiency Profile", color='white', fontweight='bold')
    
    plt.tight_layout()
    logger.save_plot(fig, "scaling_audit.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Scaling laws analyzed. Results in [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()

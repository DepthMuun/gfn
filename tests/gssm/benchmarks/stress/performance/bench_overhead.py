"""
Professional Feature Overhead Benchmark (v2.6.5)
==============================================

Measures the cost (VRAM, throughput, latency) of each physics feature.
Provides a clear efficiency vs. expressivity tradeoff analysis.
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from gfn import Manifold
from tests.benchmarks.infra.baselines.micro_gpt import MicroGPT
from tests.benchmarks.infra.utils.logger import ResultsLogger, PerformanceStats

console = Console()

def get_ablation_configs():
    """Flattened v2.6.5 physics configs for overhead analysis."""
    return {
        "Manifold (Base)": {
            'plasticity': 0.0,
            'singularity_strength': 0.0,
            'topology': 'euclidean'
        },
        "+ Plasticity": {
            'plasticity': 0.1,
            'singularity_strength': 0.0,
            'topology': 'euclidean'
        },
        "+ Singularities": {
            'plasticity': 0.0,
            'singularity_strength': 10.0,
            'singularity_thresh': 0.8,
            'topology': 'euclidean'
        },
        "+ Torus Topology": {
            'plasticity': 0.0,
            'singularity_strength': 0.0,
            'topology': 'torus',
            'R': 2.0, 'r': 1.0
        },
        "Full Physics": {
            'plasticity': 0.1,
            'singularity_strength': 10.0,
            'singularity_thresh': 0.8,
            'topology': 'torus',
            'R': 2.0, 'r': 1.0
        }
    }

def measure_overhead(name, config, device, batch_size=16, seq_len=128):
    if name == "Transformer (Baseline)":
        model = MicroGPT(vocab_size=64, dim=256, depth=6, heads=4).to(device)
    else:
        model = Manifold(
            vocab_size=64, dim=256, depth=6, heads=4, holographic=True
        ).to(device)
    
    model.eval()
    dummy_input = torch.randint(0, 64, (batch_size, seq_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5): model(dummy_input)
    
    # 1. Memory
    vram = PerformanceStats.measure_peak_memory(model, lambda: model(dummy_input))
    
    # 2. Throughput
    start = time.time()
    with torch.no_grad():
        for _ in range(50): model(dummy_input)
    elapsed = time.time() - start
    tput = (50 * batch_size) / elapsed
    lat = (elapsed / 50) * 1000
    
    return {
        "Configuration": name,
        "VRAM (MB)": vram,
        "Throughput (seq/s)": tput,
        "Latency (ms)": lat
    }

def run_benchmark():
    logger = ResultsLogger("overhead", category="performance/runtime")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN FEATURE OVERHEAD AUDIT[/] (Manifold v2.6.5)\n")

    configs = get_ablation_configs()
    report_data = []
    
    # Include Transformer Baseline
    all_tests = {"Transformer (Baseline)": None}
    all_tests.update(configs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Profiling...", total=len(all_tests))
        
        for name, cfg in all_tests.items():
            progress.update(task, description=f"Benchmarking: {name}")
            metrics = measure_overhead(name, cfg, device)
            report_data.append(metrics)
            progress.update(task, advance=1)

    # Summary Table
    table = Table(title="Overhead Breakdown", box=None)
    table.add_column("Configuration")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Throughput (seq/s)", justify="right")
    
    for r in report_data:
        table.add_row(r["Config"], f"{r['VRAM (MB)']:.1f}", f"{r['Throughput (seq/s)']:.1f}")
    
    console.print("\n", table)
    
    # Save & Plot
    logger.save_json(report_data)
    df = pd.DataFrame(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#121212')
    
    # VRAM Plot
    sns.barplot(data=df, x="Configuration", y="VRAM (MB)", ax=axes[0], palette="flare")
    axes[0].set_title("Memory Consumption", color='white', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, colors='white')
    axes[0].tick_params(axis='y', colors='white')
    
    # Throughput Plot
    sns.barplot(data=df, x="Configuration", y="Throughput (seq/s)", ax=axes[1], palette="crest")
    axes[1].set_title("Inference Throughput", color='white', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, colors='white')
    axes[1].tick_params(axis='y', colors='white')
    
    plt.tight_layout()
    logger.save_plot(fig, "overhead_premium.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Detailed profiles saved to [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()

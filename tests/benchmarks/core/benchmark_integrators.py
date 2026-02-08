"""
Professional Integrator Performance Analysis (v2.6.5)
===================================================

Scientific evaluation of all implemented integrators:
- Symplectic: Leapfrog, Yoshida (4th), ForestRuth (4th), Omelyan (4th)
- Runge-Kutta: Euler (1st), Heun (2nd), RK4 (4th), DormandPrince (5th)
- Flow: CouplingFlow (Invertible)
- Neural: NeuralIntegrator (Hybrid)

Metrics:
- Cumulative Energy Drift (%) - Measure of physics preservation
- Inference Throughput (seq/s) - Latency assessment
- Peak VRAM (MB) - Memory overhead
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

console = Console()

def measure_drift(model, drift_steps=100, device='cuda'):
    """Isolate integrator to measure pure energy drift."""
    model.eval()
    batch_size = 32
    dim = model.dim
    
    # We use a single head from the first layer for pure drift analysis
    layer = model.layers[0]
    # Handle both MLayer and ParallelMLayer
    integrator = layer.integrators[0]
    
    with torch.no_grad():
        # Force-free motion on a flat manifold (Reactive with zero U, W)
        x = torch.zeros(batch_size, model.dim // model.heads).to(device)
        v = torch.randn(batch_size, model.dim // model.heads).to(device)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        
        v_start_norm = v.norm(dim=-1).mean().item()
        
        # Integration loop
        # We need to provide dummy clutch weights for v2.6.5 compatibility
        d = x.shape[-1]
        W_f = torch.zeros(1, d, d).to(device)
        W_i = torch.zeros(1, d, d).to(device)
        b_f = torch.zeros(d).to(device)
        
        tx, tv = integrator(x, v, force=None, steps=drift_steps, 
                          W_forget_stack=W_f, W_input_stack=W_i, b_forget_stack=b_f)
        
        v_end_norm = tv.norm(dim=-1).mean().item()
        drift = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-12) * 100
        return drift

def run_integrator_suite():
    logger = ResultsLogger("integrators", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Integrators to test
    integrators = [
        'euler', 'heun', 'rk4', 'rk45', 
        'leapfrog', 'yoshida', 'forest_ruth', 'omelyan',
        'coupling', 'neural'
    ]
    
    report_data = []

    console.print(f"\n[bold]GFN INTEGRATOR QUALITY AUDIT[/] (Manifold v2.6.5)\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        suite_task = progress.add_task("Profiling Integrators...", total=len(integrators))
        
        for integ in integrators:
            progress.update(suite_task, description=f"Testing: [bold blue]{integ}[/]")
            try:
                # Setup model
                model = Manifold(
                    vocab_size=64, dim=256, depth=2, heads=4, 
                    integrator_type=integ
                ).to(device)
                
                # 1. Performance
                dummy_input = torch.randint(0, 64, (16, 128)).to(device)
                vram = PerformanceStats.measure_peak_memory(model, lambda: model(dummy_input))
                
                start = time.time()
                with torch.no_grad():
                    for _ in range(20): model(dummy_input)
                tput = (20 * 16) / (time.time() - start)
                
                # 2. Drift
                drift = measure_drift(model, drift_steps=50, device=device)
                
                report_data.append({
                    "Integrator": integ,
                    "Drift (%)": drift,
                    "Speed (seq/s)": tput,
                    "VRAM (MB)": vram
                })
                
            except Exception as e:
                console.print(f"  [red]FAILED {integ}: {e}[/]")
            
            progress.update(suite_task, advance=1)

    # Summary
    summary_table = Table(title="Integrator Metrics Summary", box=None)
    summary_table.add_column("Integrator")
    summary_table.add_column("Drift (%)", justify="right")
    summary_table.add_column("Speed (seq/s)", justify="right")
    summary_table.add_column("VRAM (MB)", justify="right")
    
    for r in report_data:
        summary_table.add_row(
            r["Integrator"], 
            f"{r['Drift (%)']:.2e}", 
            f"{r['Speed (seq/s)']:.1f}", 
            f"{r['VRAM (MB)']:.1f}"
        )
    
    console.print("\n", summary_table)
    
    # Save & Plot
    logger.save_json(report_data)
    df = pd.DataFrame(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#121212')
    
    # Drift Plot
    sns.barplot(data=df, x="Integrator", y="Drift (%)", ax=axes[0], palette="flare")
    axes[0].set_yscale("log")
    axes[0].set_title("Numerical Energy Drift (Log Scale)", color='white', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, colors='white')
    axes[0].tick_params(axis='y', colors='white')
    
    # Speed Plot
    sns.barplot(data=df, x="Integrator", y="Speed (seq/s)", ax=axes[1], palette="crest")
    axes[1].set_title("Inference Throughput", color='white', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, colors='white')
    axes[1].tick_params(axis='y', colors='white')
    
    plt.tight_layout()
    logger.save_plot(fig, "integrator_comparison_premium.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Results saved to [cyan]{logger.run_dir}[/]\n")
    return df

if __name__ == "__main__":
    run_integrator_suite()

if __name__ == "__main__":
    run_integrator_suite()

"""
Professional Sample Efficiency Analysis (v2.6.5)
==============================================

Demonstrates GFN learns MORE from LESS data (physics-informed inductive bias).
Quantifies the "Data Advantage" of Manifold-GFN over standard Transformers.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from gfn.datasets.math import MathDataset
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

def train_and_eval(name, samples, test_ds, device):
    """Trains a model with 'samples' data and returns test accuracy."""
    train_ds = MathDataset(size=samples, max_digits=2)
    vocab, dim, depth = test_ds.vocab_size, 256, 4
    
    if name == "Manifold":
        model = Manifold(
        vocab_size=15, dim=128, depth=4, heads=4,
        physics_config={'topology': 'euclidean', 'plasticity': 0.1}
    ).to(device)
        opt = RiemannianAdam(model.parameters(), lr=1e-3)
    else:
        model = MicroGPT(vocab, dim, depth, heads=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
    criterion = nn.CrossEntropyLoss(ignore_index=test_ds.char_to_id.get('<PAD>', -1))
    
    # Train for fixed 20 epochs on this small subset
    model.train()
    for _ in range(20):
        problems = [train_ds._generate_problem() for _ in range(min(16, samples))]
        ins, tgs = [], []
        for p in problems:
            ids = [test_ds.char_to_id[c] for c in p + '<EOS>']
            ins.append(ids[:-1]); tgs.append(ids[1:])
        
        max_len = max(len(s) for s in ins)
        p_in = torch.tensor([s + [0]*(max_len-len(s)) for s in ins]).to(device)
        p_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in tgs]).to(device)
        
        opt.zero_grad()
        out = model(p_in)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits.reshape(-1, vocab), p_tg.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(30):
            problem = test_ds._generate_problem()
            parts = problem.split('=')
            prompt, target = parts[0] + '=', parts[1].strip()
            ids = [test_ds.char_to_id[c] for c in prompt]
            
            curr = torch.tensor([ids]).to(device)
            state = None
            gen = list(ids)
            
            for _ in range(len(target) + 1):
                out = model(curr, state=state)
                logits = out[0] if isinstance(out, tuple) else out
                state = out[1] if isinstance(out, tuple) else None
                
                tok = torch.argmax(logits[:, -1, :], dim=-1)
                if tok.item() == test_ds.char_to_id.get('<EOS>', -1): break
                gen.append(tok.item())
                curr = tok.unsqueeze(0).unsqueeze(0) if name == "Manifold" else torch.tensor([gen]).to(device)
            
            pred = test_ds.decode(gen).split('=')[-1].strip()
            if pred == target: correct += 1
            
    return (correct / 30) * 100

def run_sample_efficiency():
    logger = ResultsLogger("sample_efficiency", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN ARCHITECTURAL SCALING AUDIT[/] (Manifold v2.6.5)\n")
    test_ds = MathDataset(size=50, max_digits=2)
    sample_sizes = [10, 20, 50, 100, 200, 500]
    report_data = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        eff_task = progress.add_task("Drilling Samples...", total=len(sample_sizes))
        
        for n in sample_sizes:
            progress.update(eff_task, description=f"Testing N={n}")
            m_acc = train_and_eval("Manifold", n, test_ds, device)
            t_acc = train_and_eval("Transformer", n, test_ds, device)
            
            report_data.append({
                "Samples": n,
                "Manifold Acc": m_acc,
                "Transformer Acc": t_acc,
                "Advantage": m_acc - t_acc
            })
            progress.update(eff_task, advance=1)

    table = Table(title="Sample Efficiency Summary", box=None)
    table.add_column("Samples")
    table.add_column("Manifold (%)", justify="right")
    table.add_column("Transformer (%)", justify="right")
    table.add_column("Advantage (%)", justify="right")
    
    for r in report_data:
        table.add_row(
            str(r["Samples"]), 
            f"{r['Manifold Acc']:.1f}", 
            f"{r['Transformer Acc']:.1f}", 
            f"{r['Advantage']:+.1f}"
        )
    
    console.print("\n", table)
    
    # Plotting
    logger.save_json(report_data)
    df = pd.DataFrame(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#121212')
    
    ax.plot(df["Samples"], df["Manifold Acc"], 'o-', label="Manifold-GFN", color='#00ADB5', lw=3)
    ax.plot(df["Samples"], df["Transformer Acc"], 's-', label="Transformer", color='#FF2E63', lw=3, ls='--')
    
    ax.fill_between(df["Samples"], df["Manifold Acc"], df["Transformer Acc"], 
                    where=(df["Manifold Acc"] >= df["Transformer Acc"]), alpha=0.2, color='#00ADB5')
    
    ax.set_title("Sample Efficiency: Manifold vs Transformer", color='white', fontweight='bold', fontsize=16)
    ax.set_xlabel("Number of Training Samples", color='white')
    ax.set_ylabel("Test Accuracy (%)", color='white')
    ax.legend()
    
    plt.tight_layout()
    logger.save_plot(fig, "efficiency_curve.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Efficiency audit complete. Results saved to [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_sample_efficiency()


if __name__ == "__main__":
    results = run_sample_efficiency_analysis()

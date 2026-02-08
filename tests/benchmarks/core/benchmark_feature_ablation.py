"""
Professional Feature Ablation Benchmark (v2.6.5)
==============================================
Tests whether each physics feature (plasticity, singularities, topology) 
actually improves learning on an associative recall task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

def create_associative_recall_data(batch_size, num_pairs=5, vocab_size=32, seq_len=12):
    """Associative recall: A->B, C->D, ... query A->?"""
    sep_token = vocab_size - 1
    sequences = []
    targets = []
    
    for _ in range(batch_size):
        keys = torch.randint(0, vocab_size - 1, (num_pairs,))
        values = torch.randint(0, vocab_size - 1, (num_pairs,))
        
        seq = []
        for k, v in zip(keys, values):
            seq.extend([k.item(), v.item()])
        seq.append(sep_token)
        
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        seq.append(keys[query_idx].item())
        
        # Padding if necessary
        if len(seq) < seq_len:
            seq += [sep_token] * (seq_len - len(seq))
            
        sequences.append(seq[:seq_len])
        targets.append(values[query_idx].item())
    
    return torch.tensor(sequences), torch.tensor(targets)

def get_ablation_configs():
    """Configurations to ablate in v2.6.5 style."""
    return {
        "Baseline (Manifold)": {
            'plasticity': 0.0,
            'singularity_thresh': 1.0,
            'singularity_strength': 0.0,
            'topology': 'euclidean'
        },
        "Plasticity Only": {
            'plasticity': 0.1,
            'singularity_thresh': 1.0,
            'singularity_strength': 0.0,
            'topology': 'euclidean'
        },
        "Singularities Only": {
            'plasticity': 0.0,
            'singularity_thresh': 0.8,
            'singularity_strength': 5.0,
            'topology': 'euclidean'
        },
        "Full Physics (Torus)": {
            'plasticity': 0.1,
            'singularity_thresh': 0.8,
            'singularity_strength': 5.0,
            'topology': 'torus',
            'R': 2.0, 'r': 1.0
        }
    }

def train_and_evaluate(config_name, physics_config, device, num_steps=300):
    vocab_size = 32
    dim = 128
    
    model = Manifold(
        vocab_size=vocab_size,
        dim=dim,
        depth=3,
        heads=4,
        integrator_type='leapfrog',
        physics_config=physics_config,
        holographic=True
    ).to(device)
    
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = {"loss": [], "acc": []}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Ablation: {config_name[:20]}", total=num_steps)
        
        for step in range(num_steps):
            inputs, targets = create_associative_recall_data(32, vocab_size=vocab_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            out = model(inputs)
            logits = out[0] if isinstance(out, tuple) else out
            
            # Predict only at the last position (the query result)
            pred_logits = logits[:, -1, :]
            loss = criterion(pred_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()
            
            history["loss"].append(loss.item())
            history["acc"].append(acc)
            progress.update(task, advance=1)

    final_acc = np.mean(history["acc"][-20:]) * 100
    final_loss = np.mean(history["loss"][-20:])
    return {"accuracy": final_acc, "loss": final_loss}

def run_benchmark():
    logger = ResultsLogger("feature_ablation", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(f"\n[bold]GFN FEATURE ABLATION AUDIT[/] (Manifold v2.6.5)\n")
    
    configs = get_ablation_configs()
    report_data = []
    
    for name, physics_config in configs.items():
        metrics = train_and_evaluate(name, physics_config, device)
        report_data.append({
            "Configuration": name,
            "Accuracy": metrics["accuracy"],
            "Loss": metrics["loss"]
        })

    # Summary Table
    summary_table = Table(title="Ablation Results Summary", box=None)
    summary_table.add_column("Configuration")
    summary_table.add_column("Accuracy (%)", justify="right")
    summary_table.add_column("Final Loss", justify="right")
    
    for row in report_data:
        summary_table.add_row(row["Configuration"], f"{row['Accuracy']:.1f}", f"{row['Loss']:.4f}")
    
    console.print("\n", summary_table)
    
    # Save & Plot
    logger.save_json(report_data)
    
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#121212')
    
    df = torch.load # Placeholder, using report_data directly
    names = [r["Configuration"] for r in report_data]
    accs = [r["Accuracy"] for r in report_data]
    
    ax.barh(names, accs, color=['#444444', '#00ADB5', '#FF2E63', '#FFD700'])
    ax.set_title("Feature Impact on Associative Recall", color='white', fontweight='bold', fontsize=16)
    ax.set_xlabel("Accuracy (%)", color='white')
    ax.set_xlim(0, 105)
    
    for i, v in enumerate(accs):
        ax.text(v + 1, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')
        
    plt.tight_layout()
    logger.save_plot(fig, "feature_ablation_premium.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Benchmark Complete. Plot saved to [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()

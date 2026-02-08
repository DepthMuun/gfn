"""
Professional Baseline Comparison Benchmark (v2.6.5)
==================================================

Objective:
- Compare Manifold-GFN with standard RNN architectures (GRU, LSTM).
- Evaluate learning dynamics and final convergence on core sequence tasks.
- Standardized reporting with publication-quality metrics and rich formatting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN Models & Physics
from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

console = Console()

# ============== BASELINES ==============

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, num_layers=depth, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.gru(x)
        return self.head(x)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=depth, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.head(x)

# ============== DATA GENERATION ==============

class ParityTask:
    """Parity Check (Modulo 2) for state tracking."""
    def __init__(self, vocab_size=2, length=50, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_int = torch.cumsum(x, dim=1) % self.mod
        PI = 3.14159265359
        # Map to toroidal target space [-PI/2, PI/2]
        y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
        return x, y_int, y_angle

def train_step_manifold(model, optimizer, inputs, targets, targets_class, device):
    """Proper training step for Manifold models with toroidal loss."""
    optimizer.zero_grad()
    output = model(inputs)
    
    if isinstance(output, tuple):
        x_pred = output[0]
    else:
        x_pred = output
        
    y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred)
    
    criterion = ToroidalDistanceLoss()
    loss_val = criterion(x_pred, y_expanded)
    
    loss_phy = 0.0
    loss_ham = 0.0
    if isinstance(output, tuple) and len(output) >= 6:
        christoffels = output[2]
        v_seq = output[3]
        x_seq = output[4]
        all_forces = output[5]
        
        if christoffels:
            # AUDIT FIX: Correct signature
            loss_phy = geodesic_regularization(christoffels, velocities=None, lambda_g=0.001, mode='structural')
            def first_head_metric(x):
                return model.layers[0].christoffels[0].get_metric(x) if hasattr(model.layers[0].christoffels[0], 'get_metric') else torch.ones_like(x)
            loss_ham = hamiltonian_loss(v_seq, states=x_seq, metric_fn=first_head_metric, lambda_h=0.0, forces=all_forces)
            
    total_loss = loss_val + loss_phy + loss_ham
    if torch.isnan(total_loss):
        return torch.tensor(float('nan')), 0.0
        
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Accuracy calculation for Manifold (Toroidal space)
    with torch.no_grad():
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        half_pi = PI * 0.5
        dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
        dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
        d_pos = dist_pos.mean(dim=-1)
        d_neg = dist_neg.mean(dim=-1)
        preds = (d_pos < d_neg).long()
        acc = (preds == targets_class).float().mean().item()
    
    return total_loss, acc

def run_baseline_comparison():
    logger = ResultsLogger("baseline_comparison", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Config
    vocab_size = 2
    dim = 128
    depth = 2
    batch_size = 32
    num_steps = 300
    seq_len = 50
    
    task = ParityTask(vocab_size=vocab_size, length=seq_len)
    
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

    models_to_test = {
        "Manifold-GFN": Manifold(
            vocab_size=vocab_size, 
            dim=dim, 
            depth=6, 
            heads=4, 
            integrator_type='leapfrog',
            physics_config=physics_config,
            impulse_scale=80.0,
            holographic=True
        ),
        "Vanilla GRU": SimpleGRU(vocab_size, dim, depth),
        "Vanilla LSTM": SimpleLSTM(vocab_size, dim, depth)
    }

    report_data = []
    
    console.print(f"\n[bold]GFN BASELINE COMPARISON[/] (Manifold v2.6.5)\n")

    for name, model in models_to_test.items():
        model = model.to(device)
        
        if "Manifold" in name:
            optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
        history = {"loss": [], "acc": []}
        
        console.print(f"\n[bold]Testing: {name}[/]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            train_task = progress.add_task("Training...", total=num_steps)
            
            start_time = time.time()
            for step in range(num_steps):
                inputs, targets_int, targets_angle = task.generate_batch(batch_size, device=device)
                
                if "Manifold" in name:
                    loss, acc = train_step_manifold(model, optimizer, inputs, targets_angle, targets_int, device)
                else:
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = criterion(output.view(-1, vocab_size), targets_int.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    with torch.no_grad():
                        preds = output.argmax(dim=-1)
                        acc = (preds == targets_int).float().mean().item()
                
                history["loss"].append(loss.item())
                history["acc"].append(acc)
                
                progress.update(train_task, advance=1)
                if step % 20 == 0:
                    progress.update(train_task, description=f"L: {loss.item():.4f} A: {acc*100:.1f}%")

        elapsed = time.time() - start_time
        final_acc = np.mean(history["acc"][-10:]) * 100
        
        report_data.append({
            "Model": name,
            "Accuracy": final_acc,
            "Time (s)": elapsed,
            "Params": sum(p.numel() for p in model.parameters())
        })

    # 2. Saving and Plotting
    summary_table = Table(title="Baseline Comparison Summary", box=None)
    summary_table.add_column("Model")
    summary_table.add_column("Accuracy (%)", justify="right")
    summary_table.add_column("Time (s)", justify="right")
    summary_table.add_column("Params (k)", justify="right")
    
    for r in report_data:
        summary_table.add_row(
            r["Model"], 
            f"{r['Accuracy']:.2f}", 
            f"{r['Time (s)']:.2f}",
            f"{r['Params']/1e3:.1f}k"
        )
    
    console.print("\n", summary_table)
    
    df = pd.DataFrame(report_data)
    logger.save_json(report_data)
    
    # Cyberpunk style plot
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#121212')
    sns.barplot(data=df, x="Model", y="Accuracy", palette=["#00ADB5", "#FF2E63", "#FFD700"], ax=ax)
    
    ax.set_title(f"Baseline Comparison: Parity Task (L={seq_len})", fontweight='bold', fontsize=16, color='white')
    ax.set_ylim(45, 105)
    ax.set_ylabel("Final Accuracy (%)", color='white')
    ax.set_xlabel("Architecture", color='white')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='white', fontweight='bold')
                    
    plt.tight_layout()
    logger.save_plot(fig, "baseline_accuracy.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Benchmark Complete. Results saved to [cyan]{logger.run_dir}[/]\n")
    return df

if __name__ == "__main__":
    run_baseline_comparison()

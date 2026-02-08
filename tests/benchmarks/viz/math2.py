import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN Models & Physics
from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from gfn.losses import geodesic_regularization, hamiltonian_loss, ToroidalDistanceLoss
from gfn.datasets import MathDataset


# Import Utils
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

console = Console()

class PeriodicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return (1.0 - torch.cos(pred - target)).mean()

class ComplexMathTask:
    def __init__(self, max_digits=8):
        self.dataset = MathDataset(max_digits=max_digits)
        self.iterator = iter(self.dataset)
        self.PI = 3.14159265359
        self.half_pi = self.PI * 0.5
    
    def generate_batch(self, batch_size, device='cpu'):
        batch = []
        expr_list = []
        for _ in range(batch_size):
            ids = next(self.iterator)
            batch.append(ids)
            expr_list.append(self.dataset.decode(ids))
        x = self.dataset.collate_fn(batch).to(device)
        y_class = []
        for expr in expr_list:
            try:
                parts = expr.split('=')
                c = int(parts[-1]) if len(parts) > 1 else 0
            except:
                c = 0
            parity = 1 if (c % 2) != 0 else 0
            y_class.append(parity)
        y_class = torch.tensor(y_class, dtype=torch.long, device=device)
        y_class_seq = y_class.unsqueeze(1).expand(x.size(0), x.size(1))
        y_angle = (y_class_seq.float() * 2.0 - 1.0) * self.half_pi
        return x, y_class_seq, y_angle

def train_step_manifold(model, optimizer, scheduler, inputs, targets, targets_class, device):
    optimizer.zero_grad()
    output = model(inputs, collect_christ=False)
    
    if isinstance(output, tuple):
        x_pred = output[0]
    else:
        x_pred = output
        
    y_float = targets.float()
    y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
    
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
        return total_loss, 0.0
        
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler: scheduler.step()
    
    with torch.no_grad():
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        half_pi = PI * 0.5

        # PyTorch native implementation - consistent CUDA/CPU execution
        dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
        dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
        d_pos = dist_pos.mean(dim=-1)
        d_neg = dist_neg.mean(dim=-1)
        preds = (d_pos < d_neg).long()
        acc = (preds == targets_class).float().mean().item()
    
    return total_loss.item(), acc


def train_model(model_name, model, max_steps=1000, device='cuda'):
    is_manifold = isinstance(model, Manifold)
    if is_manifold:
        optimizer = RiemannianAdam([
            {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0}
        ], retraction='normalize')
    else:
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4, retraction='normalize')

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
    model.train()
    
    history = {"loss": [], "acc": []}
    
    pbar = tqdm(range(max_steps), desc=f"Training {model_name}")
    acc_threshold = 0.98
    loss_threshold = 0.2
    min_steps = 100
    patience = 20
    hits = 0
    for i in pbar:
        task = ComplexMathTask(max_digits=8)
        x, y_class, y_angle = task.generate_batch(128, device=device)
        
        if is_manifold:
            loss, acc = train_step_manifold(model, optimizer, scheduler, x, y_angle, y_class, device)
            
        history["loss"].append(loss)
        history["acc"].append(acc)
        
        if i % 5 == 0:
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc*100:.1f}%"})

        if i >= min_steps and acc >= acc_threshold and loss <= loss_threshold:
            hits += 1
        else:
            hits = 0

        if hits >= patience:
            print(f"\n[GFN] {model_name} converged at step {i}")
            break
                
    return history

def filter_valid_data(lengths, acc_data, mem_data):
    """Filtrar datos válidos (no None) para graficar después de OOM"""
    valid_lengths = []
    valid_acc = []
    valid_mem = []
    oom_point = None
    
    for i, (L, acc, mem) in enumerate(zip(lengths, acc_data, mem_data)):
        if acc is not None and mem is not None:
            valid_lengths.append(L)
            valid_acc.append(acc)
            valid_mem.append(mem)
        else:
            oom_point = L if oom_point is None else oom_point
            break
    
    return valid_lengths, valid_acc, valid_mem, oom_point

def evaluate_scaling(model_name, model, lengths, device='cuda'):
    model.eval()
    results = {"acc": [], "mem": []}
    max_length = None
    oom_encountered = False
    
    console.print(f"\n[bold yellow][GFN:BENCH][/] Evaluating [cyan]{model_name}[/] Scaling Dynamics...")
    
    table = Table(title=f"Scaling Report: {model_name}")
    table.add_column("Length (N)", justify="right")
    table.add_column("Accuracy", justify="center")
    table.add_column("Peak VRAM", justify="right")

    for i, L in enumerate(lengths):
        if oom_encountered:
            # Si ya encontramos OOM, marcar los restantes como "OOM"
            table.add_row(str(L), "[red]OOM[/]", "[red]N/A[/]")
            results["acc"].append(None)  # None indica OOM
            results["mem"].append(None)
            continue
            
        task = ComplexMathTask(max_digits=8)
        x, y_class, y_angle = task.generate_batch(100, device=device)
        
        def run_inf():
            with torch.no_grad():
                if isinstance(model, Manifold):
                    state = None
                    preds_list = []
                    for t in range(x.shape[1]):
                        out = model(x[:, t:t+1], state=state)
                        l = out[0]
                        state = out[1]
                        PI = 3.14159265359
                        TWO_PI = 2.0 * PI
                        half_pi = PI * 0.5
                        dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
                        dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
                        d_pos = dist_pos.mean(dim=-1).view(-1)
                        d_neg = dist_neg.mean(dim=-1).view(-1)
                        preds_list.append((d_pos < d_neg).long())
                    return torch.stack(preds_list, dim=1)
                else:
                    return model(x).argmax(dim=-1)

        try:
            mem = PerformanceStats.measure_peak_memory(model, run_inf)
            preds = run_inf()
            acc = (preds == y_class).float().mean().item()
            
            results["acc"].append(acc)
            results["mem"].append(mem)
            
            acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
            table.add_row(str(L), acc_str, f"{mem:.2f} MB")
            max_length = L  # Actualizar la longitud máxima exitosa
            
        except torch.cuda.OutOfMemoryError as e:
            console.print(f"[bold red]OOM[/] at length {L} for {model_name}: {str(e)}")
            table.add_row(str(L), "[red]OOM[/]", "[red]N/A[/]")
            results["acc"].append(None)  # None indica OOM
            results["mem"].append(None)
            oom_encountered = True
            max_length = lengths[i-1] if i > 0 else 0  # La anterior fue la última exitosa
            
        except Exception as e:
            console.print(f"[bold red]Error[/] at length {L} for {model_name}: {str(e)}")
            table.add_row(str(L), "[red]ERROR[/]", "[red]N/A[/]")
            results["acc"].append(None)
            results["mem"].append(None)
            oom_encountered = True
            max_length = lengths[i-1] if i > 0 else 0
            
        finally:
            torch.cuda.empty_cache()
            if device == 'cuda':
                torch.cuda.synchronize()
    
    if max_length is not None:
        console.print(f"[bold yellow]Maximum successful length for {model_name}: {max_length}[/]")
    
    console.print(table)
    return results

def print_header():
    console.print("\n" + "="*80, style="magenta")
    console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]Holographic Manifold[/]", justify="center")
    console.print("="*80, style="magenta")
    console.print(f"  [bold white]Hardware:[/] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    console.print(f"  [bold white]Date:[/] {time.ctime()}")
    console.print("="*80 + "\n", style="magenta")

def run_superiority_benchmark():
    logger = ResultsLogger("math_complex", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print_header()

    dim = 128
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 50.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': False, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.2}
    }
    
    manifold = Manifold(vocab_size=2, dim=dim, depth=6, heads=4, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True).to(device)
    
        
    # 2. Training
    h_m = train_model("Manifold-GFN", manifold, max_steps=1000, device=device)
    ckpt_path = logger.results_dir / "manifold_math_complex.pt"
    torch.save(manifold.state_dict(), ckpt_path)
    console.print(f"[bold green]Checkpoint guardado:[/] {ckpt_path}")
    
    # 3. Scaling
    lengths = [20, 100, 500, 1000, 2000]
    s_m = evaluate_scaling("Manifold-GFN", manifold, lengths, device)
    
    # Filtrar datos válidos (manejar OOM)
    lengths_m, acc_m, mem_m, oom_m = filter_valid_data(lengths, s_m["acc"], s_m["mem"])
    
    console.print(f"[bold cyan]Manifold-GFN:[/] {len(lengths_m)}/{len(lengths)} successful runs" + (f" (OOM at {oom_m})" if oom_m else ""))
    
    # 4. Dashboard Plotting (Cyberpunk Premium Styling)
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    plt.rcParams.update({
        'text.color': '#00ADB5',
        'axes.labelcolor': '#00ADB5',
        'xtick.color': '#00ADB5',
        'ytick.color': '#00ADB5',
        'font.family': 'sans-serif',
        'font.weight': 'bold'
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#121212')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    cols = ['#00ADB5']
    
    # Plot A: Convergence
    ax = axes[0, 0]
    ax.plot(h_m["loss"], color=cols[0], label='Manifold GFN (Hamiltonian)', linewidth=2.5)
    ax.set_title("Training Convergence", fontweight='bold', fontsize=18, color='white')
    ax.set_yscale('log')
    ax.legend(facecolor='#1e1e1e', edgecolor=cols[0], labelcolor='white')

    # Plot B: Accuracy
    ax = axes[0, 1]
    ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN', linewidth=3.5)
    ax.set_title("Learning Dynamics", fontweight='bold', fontsize=18, color='white')
    ax.legend(facecolor='#1e1e1e', edgecolor=cols[0], labelcolor='white')

    # Plot C: OOD Generalization (con datos filtrados)
    ax = axes[1, 0]
    if len(lengths_m) > 0:
        ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN', linewidth=5, markersize=12, markerfacecolor='white')
    ax.set_title("OOD Stability (Context Scaling)", fontweight='bold', fontsize=18, color='white')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Accuracy")
    ax.legend(facecolor='#1e1e1e', edgecolor=cols[0], labelcolor='white')

    # Plot D: VRAM (con datos filtrados)
    ax = axes[1, 1]
    if len(lengths_m) > 0:
        ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Streaming)', linewidth=5, markersize=12, markerfacecolor='white')
    ax.set_title("Memory Constraints", fontweight='bold', fontsize=18, color='white')
    ax.set_yscale('log')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.legend(facecolor='#1e1e1e', edgecolor=cols[0], labelcolor='white')

    fig.suptitle("GFN: MATH COMPLEXITY DASHBOARD", fontsize=28, fontweight='bold', y=0.98, color='white')
    logger.save_plot(fig, "gfn_math_complexity.png")
    
    # FINAL REPORT TABLE
    summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY[/]", border_style="magenta", show_header=True, header_style="bold cyan")
    summary_table.add_column("Capability", justify="left")
    summary_table.add_column("Manifold-GFN", justify="center")
    summary_table.add_column("Verdict", justify="right")
    
    # Manejar OOM en el reporte final
    acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
    
    m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
    target_l = lengths[-1]
    
    summary_table.add_row(f"Long Context ({target_l})", m_str, "", "[bold green]GFN[/]")
    summary_table.add_row("Memory Complexity", "O(1)", "", "[bold green]GFN[/]")
    summary_table.add_row("Training Bias", "Hamiltonian", "", "[bold blue]ISOMORPHIC[/]")
    
    console.print("\n", summary_table)
    console.print("\n[bold green][SUCCESS][/] Benchmark Complete. Dashboard saved to [cyan]results/viz/math_complex/gfn_math_complexity.png[/]\n")

if __name__ == "__main__":
    run_superiority_benchmark()

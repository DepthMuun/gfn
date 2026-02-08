"""
Professional Learning Dynamics Comparison (v2.6.5)
=================================================

Demonstrates HOW and HOW FAST GFN learns compared to Transformers.
Showdown across convergence speed, sample efficiency, and stability.
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
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss
from gfn.datasets.math import MathDataset
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

class LearningDynamicsComparison:
    """Head-to-head training comparison framework."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = ResultsLogger("learning_dynamics", category="core")
        
        self.history = {
            'Manifold': {'loss': [], 'acc': [], 'time': []},
            'Transformer': {'loss': [], 'acc': [], 'time': []}
        }
    
    def evaluate(self, model, dataset, num_samples=50):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                problem = dataset._generate_problem()
                parts = problem.split('=')
                prompt = parts[0] + '='
                target = parts[1].strip()
                
                ids = [dataset.char_to_id[c] for c in prompt]
                input_seq = torch.tensor([ids]).to(self.device)
                
                # Generate
                if isinstance(model, Manifold):
                    logits, state = model(input_seq)[:2]
                    generated = list(ids)
                    curr_token = torch.argmax(logits[:, -1, :], dim=-1)
                    generated.append(curr_token.item())
                    
                    for _ in range(len(target) + 1):
                        logits, state = model(curr_token.unsqueeze(0).unsqueeze(0), state=state)[:2]
                        curr_token = torch.argmax(logits[:, -1, :], dim=-1)
                        tok_id = curr_token.item()
                        if tok_id == dataset.char_to_id.get('<EOS>', -1): break
                        generated.append(tok_id)
                else: # GPT
                    generated = list(ids)
                    for _ in range(len(target) + 1):
                        inp = torch.tensor([generated]).to(self.device)
                        logits = model(inp)
                        curr_token = torch.argmax(logits[:, -1, :], dim=-1)
                        tok_id = curr_token.item()
                        if tok_id == dataset.char_to_id.get('<EOS>', -1): break
                        generated.append(tok_id)
                
                pred = dataset.decode(generated).split('=')[-1].strip()
                if pred == target: correct += 1
                total += 1
        
        return (correct / total) * 100

    def train_step_manifold(self, model, optimizer, inputs, targets, vocab_size):
        """Proper training step for Manifold models with toroidal loss."""
        optimizer.zero_grad()
        output = model(inputs)
        
        if isinstance(output, tuple):
            x_pred = output[0]
        else:
            x_pred = output
            
        # Convert targets to toroidal space for manifold loss
        targets_float = targets.float()
        targets_expanded = targets_float.unsqueeze(-1).expand_as(x_pred)
        
        criterion = ToroidalDistanceLoss()
        loss_val = criterion(x_pred, targets_expanded)
        
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
            return total_loss
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item()

    def run_showdown(self, epochs=20, dataset_size=500):
        dim = 256
        depth = 6  # Match reference architecture
        
        dataset = MathDataset(size=dataset_size, max_digits=2)
        vocab_size = dataset.vocab_size
        
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
        
        manifold = Manifold(
            vocab_size=vocab_size, 
            dim=dim, 
            depth=depth, 
            heads=4,
            integrator_type='leapfrog',
            physics_config=physics_config,
            impulse_scale=80.0,
            holographic=True
        ).to(self.device)
        gpt = MicroGPT(vocab_size=vocab_size, dim=dim, depth=depth, heads=4).to(self.device)
        
        m_opt = RiemannianAdam(manifold.parameters(), lr=1e-3)
        g_opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_id.get('<PAD>', -1))
        
        console.print(f"\n[bold]GFN LEARNING DYNAMICS AUDIT[/] (Manifold v2.6.5)\n")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            m_task = progress.add_task("Manifold-GFN   ", total=epochs)
            g_task = progress.add_task("Transformer    ", total=epochs)
            
            for epoch in range(epochs):
                # Training step
                manifold.train()
                gpt.train()
                
                # Fetch epoch batch
                problems = [dataset._generate_problem() for _ in range(32)]
                inputs, targets = [], []
                for p in problems:
                    ids = [dataset.char_to_id[c] for c in p + '<EOS>']
                    inputs.append(ids[:-1])
                    targets.append(ids[1:])
                
                max_len = max(len(s) for s in inputs)
                padded_in = torch.tensor([s + [0]*(max_len-len(s)) for s in inputs]).to(self.device)
                padded_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in targets]).to(self.device)
                
                # Manifold update with proper toroidal loss
                m_loss_item = self.train_step_manifold(manifold, m_opt, padded_in, padded_tg, vocab_size)
                m_loss = torch.tensor(m_loss_item) if not torch.isnan(torch.tensor(m_loss_item)) else torch.tensor(0.0)
                
                # GPT update
                g_opt.zero_grad()
                g_logits = gpt(padded_in)
                g_loss = criterion(g_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))
                g_loss.backward()
                g_opt.step()
                
                # Evaluation
                if epoch % 2 == 0 or epoch == epochs - 1:
                    m_acc = self.evaluate(manifold, dataset)
                    g_acc = self.evaluate(gpt, dataset)
                    
                    self.history['Manifold']['loss'].append(m_loss.item())
                    self.history['Manifold']['acc'].append(m_acc)
                    self.history['Transformer']['loss'].append(g_loss.item())
                    self.history['Transformer']['acc'].append(g_acc)
                    
                    progress.update(m_task, description=f"[cyan]Manifold A:{m_acc:.1f}%")
                    progress.update(g_task, description=f"[pink1]Transformer A:{g_acc:.1f}%")

                progress.update(m_task, advance=1)
                progress.update(g_task, advance=1)

        self.plot_results()

    def plot_results(self):
        sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#121212')
        
        # Loss
        ax1.set_xlabel('Epoch', fontsize=13)
        ax1.set_ylabel('Training Loss', fontsize=13)
        ax1.set_title('🔥 Learning Speed: Loss Convergence', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs_eval, self.history['Manifold']['acc'], 
                linewidth=2.5, marker='o', markersize=6, label='Manifold', color='#2A9D8F')
        ax2.plot(epochs_eval, self.history['Transformer']['acc'], 
                linewidth=2.5, marker='s', markersize=6, label='Transformer', color='#E76F51')
        ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
        ax2.set_xlabel('Epoch', fontsize=13)
        ax2.set_ylabel('Test Accuracy (%)', fontsize=13)
        ax2.set_title('🎯 Generalization: Test Accuracy', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "learning_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_comparison(self, milestones):
        """Bar chart of epochs needed to reach milestones."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = ['50%', '70%', '90%']
        gfn_epochs = [milestones['Manifold'][t] if milestones['Manifold'][t] is not None else 100 
                     for t in thresholds]
        gpt_epochs = [milestones['Transformer'][t] if milestones['Transformer'][t] is not None else 100 
                     for t in thresholds]
        
        x = np.arange(len(thresholds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gfn_epochs, width, label='GFN', color='#2A9D8F', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt_epochs, width, label='Transformer', color='#E76F51', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height < 100:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height - 5,
                           'Not reached', ha='center', va='top', fontsize=9, color='white')
        
        ax.set_xlabel('Accuracy Milestone', fontsize=13)
        ax.set_ylabel('Epochs Required', fontsize=13)
        ax.set_title('⚡ Convergence Speed Showdown', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "convergence_speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_metrics(self):
        """Plot training efficiency (time per epoch, etc)."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        gfn_avg_time = np.mean(self.history['Manifold']['time'])
        gpt_avg_time = np.mean(self.history['Transformer']['time'])
        
        gfn_final_acc = self.history['Manifold']['acc'][-1]
        gpt_final_acc = self.history['Transformer']['acc'][-1]
        
        # Efficiency = Final Accuracy / Avg Time per Epoch
        gfn_efficiency = gfn_final_acc / (gfn_avg_time + 1e-6)
        gpt_efficiency = gpt_final_acc / (gpt_avg_time + 1e-6)
        
        models = ['Manifold', 'Transformer']
        efficiencies = [gfn_efficiency, gpt_efficiency]
        colors = ['#2A9D8F', '#E76F51']
        
        bars = ax.bar(models, efficiencies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Efficiency (Accuracy % / Sec per Epoch)', fontsize=13)
        ax.set_title('💪 Training Efficiency Comparison', fontsize=15, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_learning_showdown():
    """Main entry point for learning dynamics comparison."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🎮 Device: {device}")
    
    comparator = LearningDynamicsComparison(device=device)
    
    # Run training comparison with adjusted params
    history, milestones = comparator.run_training_comparison(
        epochs=30,  # Fewer epochs but evaluate more often
        batch_size=32,  # Larger batch
        dataset_size=800  # More training data
    )
    
    table = Table(title="Learning Dynamics Summary", box=None)
    table.add_column("Architecture")
    table.add_column("Final Acc (%)", justify="right")
    table.add_column("Final Loss", justify="right")
    
    for arch in ['Manifold', 'Transformer']:
        table.add_row(
            arch,
            f"{history[arch]['acc'][-1]*100:.1f}",
            f"{history[arch]['loss'][-1]:.4f}"
        )
    console.print("\n", table)


if __name__ == "__main__":
    run_learning_showdown()

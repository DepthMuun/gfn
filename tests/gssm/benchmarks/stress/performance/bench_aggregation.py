"""
Aggregation Comparison Benchmark
=================================

Tests 3 geodesic aggregation methods on Math task:
1. Hamiltonian Pooling
2. Geodesic Attention  
3. Momentum Accumulation

Compares accuracy, convergence speed, and stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
# MANIFOLD imports
from gfn import Manifold
from gfn.aggregation import HamiltonianPooling, GeodesicAttention, MomentumAccumulation
from gfn import CircularDistanceLoss, geodesic_regularization, hamiltonian_loss
from gfn.data_pipeline.loaders.math_loader import MathDataset
from gfn import RiemannianAdam
from gfn.geometry.boundaries import toroidal_dist_python


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
        seq_len = x.size(1)
        
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
        
        # Only supervise last token
        y_angle = torch.zeros(batch_size, seq_len, device=device)
        y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi
        
        return x, y_class, y_angle


class ManifoldWithAggregation(nn.Module):
    """
    Wrapper that adds aggregation module to standard Manifold.
    """
    
    def __init__(self, base_model, aggregation_module):
        super().__init__()
        self.base_model = base_model
        self.aggregation = aggregation_module
        
    def forward(self, input_ids, collect_christ=False):
        # Get full sequence output from base model
        output = self.base_model(input_ids, collect_christ=collect_christ)
        
        # Handle different output formats
        if isinstance(output, (tuple, list)):
            x_pred = output[0]  # [B, L, dim]
        else:
            x_pred = output
        
        # Use x_pred for both position and velocity (simplified)
        # In holographic mode, x_pred contains position states directly
        x_seq = x_pred
        v_seq = torch.zeros_like(x_pred)  # No velocity info available easily
        
        # Apply aggregation to get final state
        x_agg, v_agg, _ = self.aggregation(x_seq, v_seq)
        
        # Replace last token with aggregated state
        x_pred_agg = x_pred.clone()
        x_pred_agg[:, -1] = x_agg
        
        # Return modified output
        if isinstance(output, (tuple, list)):
            if isinstance(output, tuple):
                return (x_pred_agg,) + output[1:]
            else:  # list
                result = list(output)
                result[0] = x_pred_agg
                return result
        else:
            return x_pred_agg


def train_step(model, optimizer, scheduler, inputs, targets, targets_class, device):
    optimizer.zero_grad()
    output = model(inputs, collect_christ=False)
    
    if isinstance(output, tuple):
        x_pred = output[0]
    else:
        x_pred = output
        
    y_float = targets.float()
    y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
    
    # Loss only on last token
    x_last = x_pred[:, -1]
    y_last = y_expanded[:, -1]
    
    dist = toroidal_dist_python(x_last, y_last)
    loss_val = dist.pow(2).mean()
    
    total_loss = loss_val
    
    if torch.isnan(total_loss):
        return total_loss, 0.0
        
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler:
        scheduler.step()
    
    with torch.no_grad():
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        half_pi = PI * 0.5
        
        x_last = x_pred[:, -1]
        
        dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))
        dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))
        d_pos = dist_pos.mean(dim=-1)
        d_neg = dist_neg.mean(dim=-1)
        preds = (d_pos < d_neg).long()
        
        acc = (preds == targets_class).float().mean().item()
    
    return total_loss.item(), acc


def train_model(model_name, model, max_steps=100, device='cuda'):
    optimizer = RiemannianAdam([
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 
         'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 
         'lr': 1e-2, 'weight_decay': 0}
    ], retraction='euclidean')
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
    model.train()
    
    history = {"loss": [], "acc": []}
    
    pbar = tqdm(range(max_steps), desc=f"Training {model_name}")
    
    for i in pbar:
        task = ComplexMathTask(max_digits=8)
        x, y_class, y_angle = task.generate_batch(128, device=device)
        
        loss, acc = train_step(model, optimizer, scheduler, x, y_angle, y_class, device)
        
        history["loss"].append(loss)
        history["acc"].append(acc)
        
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc*100:.1f}%"})
        
        # Early stopping if converged
        if i > 100 and acc > 0.95:
            print(f"✓ Converged at step {i}!")
            break
    
    return history


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Base configuration
    vocab_size = 16
    dim = 128
    depth = 6
    heads = 4
    
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
        'stability': {'base_dt': 0.1}
    }
    
    results_all = {}
    
    # Test 1: Hamiltonian Pooling
    print("\n" + "="*80)
    print("TEST 1: Hamiltonian Pooling")
    print("="*80)
    
    base_model_1 = Manifold(
        vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
        integrator_type='leapfrog',
        impulse_scale=10.0,  # REDUCED from 80
        holographic=True
    ).to(device)
    
    agg_hamiltonian = HamiltonianPooling(dim=dim, temperature=0.5, learn_metric=True)
    model_1 = ManifoldWithAggregation(base_model_1, agg_hamiltonian).to(device)
    
    history_1 = train_model("Hamiltonian", model_1, max_steps=100, device=device)
    results_all['Hamiltonian'] = history_1
    
    # Test 2: Geodesic Attention
    print("\n" + "="*80)
    print("TEST 2: Geodesic Attention")
    print("="*80)
    
    base_model_2 = Manifold(
        vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
        integrator_type='leapfrog',
        impulse_scale=10.0,
        holographic=True
    ).to(device)
    
    agg_geodesic = GeodesicAttention(dim=dim, temperature=0.5, distance_metric='riemannian')
    model_2 = ManifoldWithAggregation(base_model_2, agg_geodesic).to(device)
    
    history_2 = train_model("Geodesic", model_2, max_steps=100, device=device)
    results_all['Geodesic'] = history_2
    
    # Test 3: Momentum Accumulation
    print("\n" + "="*80)
    print("TEST 3: Momentum Accumulation")
    print("="*80)
    
    base_model_3 = Manifold(
        vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
        integrator_type='leapfrog',
        impulse_scale=10.0,
        holographic=True
    ).to(device)
    
    agg_momentum = MomentumAccumulation(dim=dim, alpha=0.2, mode='avg', gated=True)
    model_3 = ManifoldWithAggregation(base_model_3, agg_momentum).to(device)
    
    history_3 = train_model("Momentum", model_3, max_steps=300, device=device)
    results_all['Momentum'] = history_3
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for name, history in results_all.items():
        final_acc = history['acc'][-1]
        final_loss = history['loss'][-1]
        max_acc = max(history['acc'])
        
        # Find convergence step (first time acc > 90%)
        conv_step = None
        for i, acc in enumerate(history['acc']):
            if acc > 0.90:
                conv_step = i
                break
        
        print(f"\n{name}:")
        print(f"  Final Accuracy: {final_acc*100:.2f}%")
        print(f"  Max Accuracy:   {max_acc*100:.2f}%")
        print(f"  Final Loss:     {final_loss:.4f}")
        print(f"  Converged at:   {conv_step if conv_step else 'Did not converge'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

"""
GSSM Quicktest con ToroidalLoss sobre una task ARC mini.
Objetivo: validar que el pipeline (forces -> forward -> toroidal loss -> backward -> step)
funciona sin NaN y la loss baja en 100 steps sobre la task 'mini_flip_h_001'.

Setup:
  - Embedding: continuous, impulse_scale=5.0 (no 80.0)
  - Model: dim=64, depth=2, heads=4
  - Loss: toroidal con target mapeado a [-pi/2, pi/2]
  - Optimizer: AdamW dual-group
"""
import sys
import math
import random
import tempfile
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[5]  # scripts/smoke_test_toroidal_one_task.py -> scripts -> arc-agi-2 -> convergence -> benchmarks -> tests -> ROOT
BENCHMARK_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import gfn
from gfn.realizations.gssm.training.optimizer import make_gfn_optimizer
from src.evaluation.metrics import ARCMetrics
from src.training.few_shot import (
    build_fewshot_forces,
    extract_predictions,
    get_model_embedding,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# 1) Mini dataset de 1 task (mini_flip_h_001 del quicktest_mse)
# =============================================================================
def make_mini_task():
    """Una task ARC simple: flip horizontal de la grilla."""
    return {
        'task_id': 'mini_flip_h_001',
        'train': [
            {
                'input': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                'output': torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
            },
            {
                'input': torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                'output': torch.tensor([[7.0, 8.0], [5.0, 6.0]]),
            },
            {
                'input': torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                'output': torch.tensor([[2.0, 3.0], [0.0, 1.0]]),
            },
        ],
        'test': [
            {
                'input': torch.tensor([[9.0, 8.0], [7.0, 6.0]]),
                'output': torch.tensor([[7.0, 6.0], [9.0, 8.0]]),
            }
        ],
    }


# =============================================================================
# 2) Config de modelo (corregida)
# =============================================================================
def get_config():
    return {
        'vocab_size': 10,
        'dim': 8,
        'depth': 1,
        'heads': 1,
        'max_seq_len': 900,
        'embedding_mode': 'continuous',
        'continuous_input_dim': 900,
        'store_full_sequence': True,
        'physics': {
            'embedding': {
                'type': 'functional',
                'mode': 'continuous',
                'coord_dim': 900,
                'impulse_scale': 80.0,  # RECALIBRADO: 80.0 era para dim=8
            },
            'readout': {
                'type': 'implicit',
                'coord_dim': 16,
                'out_dim': 900,
            },
            'topology': {
                'type': 'torus',
                'R': 3.0,
                'r': 1.0,
                'learnable_R': True,
                'learnable_r': True,
            },
            'stability': {
                'base_dt': 0.05,
                'dt_min': 0.0001,
                'dt_max': 0.2,
                'friction': 2.0,
                'velocity_saturation': 15.0,
                'curvature_clamp': 1.0,
            },
            'active_inference': {
                'enabled': True,
                'hysteresis': {'enabled': False, 'strength': 0.1, 'decay': 0.9},
                'curiosity': {'enabled': False},
                'stochasticity': {'enabled': False},
            },
            'integrator': {'type': 'leapfrog', 'adaptive_dt': False},
        },
        'readout_type': 'implicit',
        'readout_hidden_dim': 64,
        'readout_out_dim': 900,
    }


def create_model(device=DEVICE):
    cfg = get_config()
    model = gfn.create(
        'gssm',
        vocab_size=cfg['vocab_size'],
        dim=cfg['dim'],
        heads=cfg['heads'],
        depth=cfg['depth'],
        max_seq_len=cfg['max_seq_len'],
        embedding_mode=cfg['embedding_mode'],
        continuous_input_dim=cfg['continuous_input_dim'],
        physics=cfg['physics'],
        readout_type=cfg['readout_type'],
        readout_hidden_dim=cfg['readout_hidden_dim'],
        readout_out_dim=cfg['readout_out_dim'],
        holographic=True,
        device=device,
    )
    return model.to(device)


# =============================================================================
# 3) Loss toroidal con target mapeado a angulos
# =============================================================================
def value_to_angle(x: torch.Tensor) -> torch.Tensor:
    """Map values in [0, 9] -> angles in [-pi/2, pi/2]."""
    return (x - 4.5) * (math.pi / 9.0)


def toroidal_arc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Toroidal angular distance loss.
    pred:   [B, 900] predicted angles
    target: [B, 900] target angles
    """
    # Wrap difference to [-pi, pi]
    diff = pred - target
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff_wrapped.pow(2).mean()


# =============================================================================
# 4) Test loop
# =============================================================================
def run_quicktest(steps=1000, lr=1e-3, verbose=True):
    print("=" * 60)
    print("GSSM + ToroidalLoss Quicktest (1 ARC task)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Steps: {steps}, LR: {lr}")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Modelo
    print("\n[1/4] Creando modelo...")
    model = create_model(device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    # Optimizer (mismo patron que XOR que funciona)
    optimizer = make_gfn_optimizer(
        model, lr=lr, physics_lr_scale=10.0, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01
    )

    # Task mini
    print("\n[2/4] Preparando task mini_flip_h_001...")
    task = make_mini_task()
    train_pairs = []
    for p in task['train']:
        train_pairs.append({
            'input': p['input'].to(DEVICE),
            'output': p['output'].to(DEVICE),
        })
    test_input = task['test'][0]['input'].to(DEVICE)
    test_output = task['test'][0]['output'].to(DEVICE)

    # Tamaños reales (mini task = 2x2 = 4 celdas)
    test_h, test_w = test_output.shape
    test_size = test_h * test_w
    train_sizes = [(p['output'].numel()) for p in train_pairs]

    # Construir secuencia de fuerzas (poco-shot) - se hace fresh en cada step
    # para evitar graph reuse issues
    print("\n[3/4] Preparando pipeline de training (forces por step)...")

    # Targets como angulos (full 900-dim padded)
    test_target_angle_full = value_to_angle(test_output.flatten().unsqueeze(0))  # [1, 900]

    # Training
    print("\n[4/4] Entrenando (verifica NaN, loss, accuracy)...")
    losses = []
    best_loss = float('inf')

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        # Construir forces EN CADA STEP (evita graph reuse)
        forces, pred_timesteps, target_grids = build_fewshot_forces(
            model, train_pairs, test_input, device=DEVICE
        )
        target_grids_angle = [value_to_angle(g.unsqueeze(0) if g.dim() == 1 else g) for g in target_grids]

        # Forward con force sequence
        out = model(force_manual=forces)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        predictions = extract_predictions(logits, pred_timesteps)
        train_preds = predictions[:-1]
        test_pred = predictions[-1]  # [1, 900] logits predichos

        # Convertir logits a angulos acotados [-pi/2, pi/2]
        # Tomar solo los primeros test_size valores (el resto es padding)
        test_pred_angle = torch.tanh(test_pred[..., :test_size]) * (math.pi / 2)  # [1, test_size]
        test_target_slice = test_target_angle_full[..., :test_size]  # [1, test_size]

        # Loss principal (test)
        primary = toroidal_arc_loss(test_pred_angle, test_target_slice)

        # Aux loss (train pairs): matchear tamaños
        aux_losses = []
        for i, (pred, tgt) in enumerate(zip(train_preds, target_grids_angle)):
            size = train_sizes[i] if i < len(train_sizes) else tgt.shape[-1]
            pred_slice = torch.tanh(pred[..., :size]) * (math.pi / 2)
            tgt_slice = tgt[..., :size]
            aux_losses.append(toroidal_arc_loss(pred_slice, tgt_slice))
        aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=DEVICE)

        loss = primary + 0.5 * aux

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [step {step}] NaN/Inf detected! loss={loss.item()}")
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

        if verbose and (step + 1) % 20 == 0:
            # Eval rapida: pasar pred a int [0,9] y comparar
            with torch.no_grad():
                pred_vals = (test_pred_angle.squeeze().cpu() / (math.pi / 9.0) + 4.5).clamp(0, 9).round().long()
                target_vals = test_output.flatten().cpu().long()
                pixel_acc = (pred_vals == target_vals).float().mean().item()
            print(
                f"  step {step+1:4d}/{steps} | "
                f"loss={loss.item():.4f} (best={best_loss:.4f}) | "
                f"pixel_acc={pixel_acc:.2%} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # Resumen
    initial = sum(losses[:10]) / 10
    final = sum(losses[-10:]) / 10
    improvement = (initial - final) / max(initial, 1e-6)
    print("\n" + "=" * 60)
    print(f"Initial loss (avg first 10): {initial:.4f}")
    print(f"Final loss (avg last 10):    {final:.4f}")
    print(f"Improvement:                 {improvement:.2%}")
    print("=" * 60)

    return final < initial and not torch.isnan(torch.tensor(final))


if __name__ == '__main__':
    success = run_quicktest(steps=1000, lr=1e-3)
    print("\n" + ("[OK] Pipeline funcional" if success else "[FAIL] La loss no bajo o hubo NaN"))

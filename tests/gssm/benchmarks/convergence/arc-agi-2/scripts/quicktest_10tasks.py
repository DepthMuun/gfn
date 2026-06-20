"""
GSSM Quicktest 10-Task: entrena 10 tasks reales de ARC-AGI-2 simultáneamente
hasta 100% acc en train, luego evalua en validacion.

Setup (validado en smoke_test_toroidal_one_task.py):
  - GSSM con FunctionalEmbedding + impulse_scale=5.0
  - ToroidalLoss con value_to_angle
  - make_gfn_optimizer dual-group
  - Inner-loop con early-exit

Entrenamiento:
  - 10 tasks del train set
  - Pool de tasks: cada step samplea 1 task al azar (no secuencial)
  - Continua hasta que TODAS las tasks alcancen inner_max_acc
  - O hasta max_steps

Evaluacion:
  - Sobre 20 tasks del val set
  - Reporta task_acc, pixel_acc, num_tasks_perfect
"""
import sys
import math
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import argparse

# Paths
HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[5]  # scripts -> arc-agi-2 -> convergence -> benchmarks -> tests -> ROOT
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import gfn
from gfn.realizations.gssm.training.optimizer import make_gfn_optimizer

from src.data.arc_dataset import ARCAGI2Dataset
from src.training.few_shot import build_fewshot_forces, extract_predictions

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# 1) Loss (mismo patron que quicktest anterior)
# =============================================================================
VALUE_MAX = 9.0

parser = argparse.ArgumentParser(description="Train tgfn on ARC-AGI-2")
parser.add_argument(
    "--full_dataset",
    action="store_true",
    help="Train on the full dataset"
)
parser.add_argument("--num_train_tasks", type=int, default=10, help="Number of training tasks")
parser.add_argument("--num_val_tasks", type=int, default=20, help="Number of validation tasks")
parser.add_argument("--output_stats", type= str,  default="quicktest_results.jsonl", help="File to save the results")
parser.add_argument("--output_model", type= str,  default="quicktest_models.pth", help="File to save the results")

args = parser.parse_args()


def value_to_angle(x: torch.Tensor) -> torch.Tensor:
    return (x - VALUE_MAX / 2.0) * (math.pi / VALUE_MAX)


def angle_to_value(angle: torch.Tensor) -> torch.Tensor:
    return angle * (VALUE_MAX / math.pi) + VALUE_MAX / 2.0


def toroidal_arc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff_wrapped.pow(2).mean()


# =============================================================================
# 2) Modelo
# =============================================================================
def get_config():
    return {
        'vocab_size': 10,
        'dim': 64,
        'depth': 2,
        'heads': 4,
        'rank': 128,
        'max_seq_len': 900,
        'embedding_mode': 'continuous',
        'continuous_input_dim': 900,
        'store_full_sequence': True,
        'physics': {
            'embedding': {
                'type': 'functional',
                'mode': 'continuous',
                'coord_dim': 900,
                'impulse_scale': 5.0,
            },
            'readout': {
                'type': 'implicit',
                'coord_dim': 64,
                'out_dim': 900,
            },
            'topology': {
                'type': 'torus',
                'riemannian_type': 'low_rank',
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
                'integrator_type': 'leapfrog',
            },
            'active_inference': {
                'enabled': True,
                'hysteresis': {'enabled': False, 'strength': 0.1, 'decay': 0.9},
                'curiosity': {'enabled': False},
                'stochasticity': {'enabled': False},
            },
        },
        'readout_type': 'implicit',
        'readout_hidden_dim': 64,
        'readout_out_dim': 900,
    }


def create_model():
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
        device=DEVICE,
    )
    return model.to(DEVICE)


# =============================================================================
# 3) Train step sobre 1 task (con tracking de pixel_acc y task_acc)
# =============================================================================
def train_step(model, optimizer, batch, pad_to=900, aux_weight=0.5, return_preds=False):
    """
    Un step de training sobre 1 task. Retorna (loss, pixel_acc, task_acc, info).
    """
    device = DEVICE
    train_pairs, test_input, test_output = [], None, None
    for p in batch['train_pairs']:
        train_pairs.append({
            'input': p['input'].to(device),
            'output': p['output'].to(device),
            'input_size': p.get('input_size', None),
            'output_size': p.get('output_size', None),
        })
    test_input = batch['test_input'].to(device)
    test_input_size = batch.get('test_input_size', None)
    test_output_size = batch.get('test_output_size', None)
    if 'test_output' in batch and batch['test_output'] is not None:
        test_output = batch['test_output'].to(device)

    if test_output is None:
        return None

    if test_output_size is not None:
        test_h, test_w = int(test_output_size[0]), int(test_output_size[1])
    else:
        test_h, test_w = test_output.shape[-2], test_output.shape[-1]
    test_size = test_h * test_w

    # Build forces (fresh cada step)
    forces, pred_timesteps, target_grids = build_fewshot_forces(
        model, train_pairs, test_input, device=device, test_input_size=test_input_size
    )

    # Forward
    out = model(force_manual=forces)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    # Extract predictions
    predictions = extract_predictions(logits, pred_timesteps)
    test_pred = predictions[-1]  # [1, 900]

    # Targets en angulos
    test_out_grid = test_output
    if test_out_grid.dim() == 3:
        test_out_grid = test_out_grid[0]
    test_out_grid = test_out_grid[:test_h, :test_w]
    test_target_flat = test_out_grid.flatten()
    if test_target_flat.numel() < pad_to:
        test_target_flat = F.pad(test_target_flat, (0, pad_to - test_target_flat.numel()), value=0)
    test_target_angle = value_to_angle(test_target_flat.unsqueeze(0))

    target_grids_angle = []
    for g in target_grids:
        if g.dim() == 1:
            if g.numel() < pad_to:
                g = F.pad(g, (0, pad_to - g.numel()), value=0)
            target_grids_angle.append(value_to_angle(g.unsqueeze(0)))
        else:
            target_grids_angle.append(value_to_angle(g))

    # Loss principal
    test_pred_angle = torch.tanh(test_pred[..., :test_size]) * (math.pi / 2)
    primary = toroidal_arc_loss(test_pred_angle, test_target_angle[..., :test_size])

    # Loss auxiliar (train pairs)
    aux_losses = []
    train_preds = predictions[:-1]
    for i, (pred, tgt) in enumerate(zip(train_preds, target_grids_angle)):
        out_size = train_pairs[i].get('output_size', None) if i < len(train_pairs) else None
        if out_size is not None:
            size = int(out_size[0]) * int(out_size[1])
        else:
            size = tgt.shape[-1] if tgt.dim() > 1 else tgt.shape[0]
        pred_slice = torch.tanh(pred[..., :size]) * (math.pi / 2)
        tgt_slice = tgt[..., :size]
        aux_losses.append(toroidal_arc_loss(pred_slice, tgt_slice))
    aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)

    loss = primary + aux_weight * aux

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Metricas
    with torch.no_grad():
        pred_vals = (test_pred_angle.squeeze().cpu() / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0)
        pred_vals = pred_vals.clamp(0, VALUE_MAX).round().long().numpy()
        target_vals = test_target_angle[..., :test_size].squeeze().cpu().numpy()
        target_vals = (target_vals / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0).round().astype(np.int64).clip(0, 9)

        pixel_acc = float((pred_vals == target_vals).mean())
        task_acc = float(np.array_equal(pred_vals, target_vals))

    info = {
        'loss': loss.item(),
        'primary': primary.item(),
        'aux': aux.item() if aux_losses else 0.0,
        'pixel_acc': pixel_acc,
        'task_acc': task_acc,
        'task_id': batch.get('task_id', 'unknown'),
    }
    return info


def evaluate(model, val_dataset, n_eval=20):
    """Evalua el modelo en N tasks de validacion. Retorna metricas agregadas."""
    model.eval()
    results = []

    indices = list(range(min(n_eval, len(val_dataset))))
    if not indices:
        return {'task_accuracy': 0.0, 'pixel_accuracy': 0.0, 'num_tasks': 0}

    for idx in indices:
        batch = val_dataset[idx]
        if 'test_output' not in batch or batch['test_output'] is None:
            continue
        device = DEVICE
        train_pairs, test_input, test_output = [], None, None
        for p in batch['train_pairs']:
            train_pairs.append({
                'input': p['input'].to(device),
                'output': p['output'].to(device),
                'input_size': p.get('input_size', None),
                'output_size': p.get('output_size', None),
            })
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)
        test_input_size = batch.get('test_input_size', None)
        test_output_size = batch.get('test_output_size', None)

        if test_output_size is not None:
            test_h, test_w = int(test_output_size[0]), int(test_output_size[1])
        else:
            test_h, test_w = test_output.shape[-2], test_output.shape[-1]
        test_size = test_h * test_w

        with torch.no_grad():
            forces, pred_timesteps, target_grids = build_fewshot_forces(
                model, train_pairs, test_input, device=device, test_input_size=test_input_size
            )
            out = model(force_manual=forces)
            logits = out[0] if isinstance(out, tuple) else out
            predictions = extract_predictions(logits, pred_timesteps)
            test_pred = predictions[-1]

            test_target_flat = test_output.flatten()
            if test_target_flat.numel() < 900:
                test_target_flat = F.pad(test_target_flat, (0, 900 - test_target_flat.numel()), value=0)
            test_target_angle = value_to_angle(test_target_flat.unsqueeze(0))

            test_pred_angle = torch.tanh(test_pred[..., :test_size]) * (math.pi / 2)
            pred_vals = (test_pred_angle.squeeze().cpu() / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0)
            pred_vals = pred_vals.clamp(0, VALUE_MAX).round().long().numpy()
            target_vals = test_target_angle[..., :test_size].squeeze().cpu().numpy()
            target_vals = (target_vals / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0).round().astype(np.int64).clip(0, 9)

            pixel_acc = float((pred_vals == target_vals).mean())
            task_acc = float(np.array_equal(pred_vals, target_vals))

            results.append({
                'loss': float('nan'),
                'primary': float('nan'),
                'aux': float('nan'),
                'pixel_acc': pixel_acc,
                'task_acc': task_acc,
                'task_id': batch.get('task_id', 'unknown'),
            })

    model.train()

    if not results:
        return {'task_accuracy': 0.0, 'pixel_accuracy': 0.0, 'num_tasks': 0}

    task_acc = sum(r['task_acc'] for r in results) / len(results)
    pixel_acc = sum(r['pixel_acc'] for r in results) / len(results)
    return {
        'task_accuracy': task_acc,
        'pixel_accuracy': pixel_acc,
        'num_tasks': len(results),
        'tasks_perfect': sum(1 for r in results if r['task_acc'] == 1.0),
    }


# =============================================================================
# 4) Main: train 10 tasks, eval val
# =============================================================================
def main():
    print("=" * 60)
    print("GSSM Quicktest: 10 ARC Tasks Simultaneas")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Datasets
    print("\n[1/3] Cargando datasets...")
    data_path = BENCHMARK_ROOT / "data" / "processed" / "splits"
    train_ds = ARCAGI2Dataset(data_path=str(data_path), split='train', max_train_pairs=3, shuffle_pairs=True)
    val_ds = ARCAGI2Dataset(data_path=str(data_path), split='val', max_train_pairs=3, shuffle_pairs=False)
    print(f"  Train: {len(train_ds)} tasks")
    print(f"  Val:   {len(val_ds)} tasks")

    # Seleccionar 10 tasks de train (las primeras 10, fijas para reproducibilidad)
    if args.full_dataset:
        n_train_tasks = len(train_ds)
    else:
        n_train_tasks = args.num_train_tasks
    train_task_ids = list(range(n_train_tasks))
    print(f"  Usando {n_train_tasks} tasks de train: {train_task_ids}")

    # Modelo
    print("\n[2/3] Creando modelo...")
    model = create_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Optimizer
    optimizer = make_gfn_optimizer(
        model,
        lr=1e-3,
        physics_lr_scale=10.0,
        weight_decay=1e-4,
        physics_param_names=frozenset({'x0', 'v0', 'impulse_scale'}),
    )

    # Tracking por task
    best_pixel_acc = {tid: 0.0 for tid in train_task_ids}
    best_task_acc = {tid: 0.0 for tid in train_task_ids}
    history = defaultdict(list)

    # Hyperparametros del entrenamiento
    max_steps = 10000000
    early_exit_acc = 1  # pixel_acc threshold

    print("\n[3/3] Entrenando (max %d pasos, early-exit @ %.0f%% pixel_acc)..." % (max_steps, early_exit_acc * 100))
    print("=" * 60)

    t_start = time.time()
    step = 0
    all_reached = False

    while step < max_steps and not all_reached:
        # Samplear 1 task al azar de las 10
        tid = random.choice(train_task_ids)
        batch = train_ds[tid]

        try:
            info = train_step(model, optimizer, batch)
        except Exception as e:
            print(f"  [step {step}] Error en task {tid}: {str(e)[:50]}")
            step += 1
            continue

        if info is None:
            step += 1
            continue

        if torch.isnan(torch.tensor(info['loss'])):
            print(f"  [step {step}] NaN loss, abort")
            break

        # Track best
        if info['pixel_acc'] > best_pixel_acc[tid]:
            best_pixel_acc[tid] = info['pixel_acc']
        if info['task_acc'] > best_task_acc[tid]:
            best_task_acc[tid] = info['task_acc']

        history[tid].append(info)

        step += 1

        # Check si todas las tasks llegaron al umbral
        if step % 20 == 0:
            n_reached = sum(1 for v in best_pixel_acc.values() if v >= early_exit_acc)
            n_perfect = sum(1 for v in best_task_acc.values() if v >= 1.0)
            avg_loss = np.mean([history[t][-1]['loss'] for t in train_task_ids if history[t]])
            avg_pix = np.mean(list(best_pixel_acc.values()))
            print(
                f"  step {step:4d} | "
                f"avg_loss={avg_loss:.4f} | "
                f"avg_pix={avg_pix:.2%} | "
                f"tasks@{early_exit_acc:.0%}={n_reached}/{n_train_tasks} | "
                f"perfect={n_perfect}/{n_train_tasks}"
            )

            if n_reached == n_train_tasks:
                all_reached = True
                print(f"\n  *** Todas las {n_train_tasks} tasks llegaron a {early_exit_acc:.0%} pixel_acc ***")

    train_time = time.time() - t_start
    print(f"\nEntrenamiento: {train_time:.1f}s en {step} pasos")

    # Resumen train
    print("\n" + "=" * 60)
    print("RESULTADOS TRAIN")
    print("=" * 60)
    for tid in train_task_ids:
        if history[tid]:
            last = history[tid][-1]
            best = max(h['pixel_acc'] for h in history[tid])
            perfect = any(h['task_acc'] == 1.0 for h in history[tid])
            print(
                f"  task {tid:2d} (id={last['task_id'][:30]:30s}) | "
                f"last_pix={last['pixel_acc']:.2%} | "
                f"best_pix={best:.2%} | "
                f"perfect={'YES' if perfect else 'no'}"
            )

    n_perfect = sum(1 for v in best_task_acc.values() if v >= 1.0)
    avg_pix = np.mean(list(best_pixel_acc.values()))
    print(f"\n  Promedio pixel_acc: {avg_pix:.2%}")
    print(f"  Tasks perfectas (100%): {n_perfect}/{n_train_tasks}")

    # Evaluacion en val
    print("\n" + "=" * 60)
    print("EVALUACION EN VALIDATION (20 tasks)")
    print("=" * 60)
    val_results = evaluate(model, val_ds, n_eval=20)
    print(f"  Tasks evaluadas: {val_results['num_tasks']}")
    print(f"  Task accuracy:   {val_results['task_accuracy']:.2%}")
    print(f"  Pixel accuracy:  {val_results['pixel_accuracy']:.2%}")
    print(f"  Tasks perfectas: {val_results.get('tasks_perfect', 0)}/{val_results['num_tasks']}")

# Guardar modelo
    model_path = HERE / args.output_model

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "n_params": n_params,
        "train_time_sec": train_time,
    }, model_path)

    print(f"\nModelo guardado en: {model_path}")

# Guardar resumen
    output = {
        'device': DEVICE,
        'n_params': n_params,
        'train_time_sec': train_time,
        'n_steps': step,
        'early_exit_acc': early_exit_acc,
        'all_reached_early_exit': all_reached,
        'train': {
            'avg_pixel_acc': float(avg_pix),
            'n_perfect': int(n_perfect),
            'n_total': n_train_tasks,
        },
        'val': {
            'task_accuracy': val_results['task_accuracy'],
            'pixel_accuracy': val_results['pixel_accuracy'],
            'num_tasks': val_results['num_tasks'],
            'tasks_perfect': val_results.get('tasks_perfect', 0),
        },
    }
    out_path = HERE / args.output_stats
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()

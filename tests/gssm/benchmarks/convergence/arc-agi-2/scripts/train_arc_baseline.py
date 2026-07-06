"""
GSSM Training completo para ARC-AGI-2 con ToroidalLoss.

Setup corregido basado en smoke_test_toroidal_one_task.py:
  - Embedding: continuous + impulse_scale=5.0 (recalibrado)
  - Loss: toroidal angular distance (targets en [-pi/2, pi/2])
  - Forces construidas fresh en cada step (evita graph reuse)
  - Optimizer: make_gfn_optimizer (dual-group LR)
  - Few-shot: condiciona en train pairs, predice test

Pipeline:
  1. Crea modelo GSSM
  2. Carga datos ARC
  3. Loop: build forces (fresh) -> forward -> toroidal loss -> backward -> step
  4. Validation: mismo flujo sin grad, calcula pixel_acc y task_acc

Uso:
  python train_arc_baseline.py --data_path ../data/processed --epochs 20
  python train_arc_baseline.py --data_path ../data/processed --config small --epochs 50
"""
import sys
import math
import json
import argparse
import random
import time
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]  # scripts/train_2.py -> scripts -> arc-agi-2 -> convergence -> benchmarks -> tests -> ROOT
BENCHMARK_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gfn
from gfn.realizations.gssm.training.optimizer import make_gfn_optimizer

from src.evaluation.metrics import ARCMetrics
from src.training.few_shot import (
    build_fewshot_forces,
    extract_predictions,
    compute_forces,
)


# =============================================================================
# 1) Configs
# =============================================================================
def get_tiny_config():
    """Config minimo (166K params) - buena para ARC-AGI-2 pequeno."""
    return {
        'vocab_size': 10,
        'dim': 64,
        'depth': 1,
        'heads': 4,
        'max_seq_len': 900,
        'embedding_mode': 'continuous',
        'continuous_input_dim': 900,
        'store_full_sequence': True,
        'physics': {
            'embedding': {
                'type': 'functional',
                'mode': 'continuous',
                'coord_dim': 900,
                'impulse_scale': 5.0,  # RECALIBRADO: 80.0 era para dim=8
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


def get_small_config():
    """Config mas grande (2-3x tiny) - para escalar despues."""
    cfg = get_tiny_config()
    cfg['dim'] = 96
    cfg['depth'] = 3
    cfg['heads'] = 6
    return cfg


def get_config(name: str = 'tiny'):
    if name == 'small':
        return get_small_config()
    return get_tiny_config()


# =============================================================================
# 2) Modelo
# =============================================================================
def create_model(config_name: str = 'tiny', device: str = 'cpu'):
    cfg = get_config(config_name)
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
# 3) Loss: toroidal angular distance
# =============================================================================
VALUE_MAX = 9.0


def value_to_angle(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 9] -> [-pi/2, pi/2] centered at 4.5."""
    return (x - VALUE_MAX / 2.0) * (math.pi / VALUE_MAX)


def angle_to_value(angle: torch.Tensor) -> torch.Tensor:
    """Map [-pi/2, pi/2] -> [0, 9]."""
    return angle * (VALUE_MAX / math.pi) + VALUE_MAX / 2.0


def toroidal_arc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Toroidal angular distance loss.
    Both pred and target in [-pi, pi].
    """
    diff = pred - target
    # Wrap to [-pi, pi]
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff_wrapped.pow(2).mean()


# =============================================================================
# 4) Train / Val functions
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_pairs(batch: dict, device: str):
    """Extrae train pairs y test del batch."""
    train_pairs = []
    for p in batch['train_pairs']:
        train_pairs.append({
            'input': p['input'].to(device),
            'output': p['output'].to(device),
        })
    test_input = batch['test_input'].to(device)
    test_output = batch.get('test_output', None)
    if test_output is not None:
        test_output = test_output.to(device)
    return train_pairs, test_input, test_output


def compute_losses(
    predictions: list,
    target_grids_angle: list,
    test_pred: torch.Tensor,
    test_target_angle: torch.Tensor,
    test_size: int,
    train_sizes: list,
    auxiliary_weight: float,
    device: str,
):
    """
    Compute primary (test) + auxiliary (train pairs) toroidal losses.

    Returns: (total_loss, primary_loss, aux_loss, pixel_acc)
    """
    # Primary: test prediction
    # Slice to actual size (rest is padding)
    test_pred_angle = torch.tanh(test_pred[..., :test_size]) * (math.pi / 2)
    primary = toroidal_arc_loss(test_pred_angle, test_target_angle[..., :test_size])

    # Aux: train pairs
    aux_losses = []
    for i, (pred, tgt) in enumerate(zip(predictions, target_grids_angle)):
        size = train_sizes[i] if i < len(train_sizes) else tgt.shape[-1]
        pred_slice = torch.tanh(pred[..., :size]) * (math.pi / 2)
        tgt_slice = tgt[..., :size]
        aux_losses.append(toroidal_arc_loss(pred_slice, tgt_slice))
    aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)

    total = primary + auxiliary_weight * aux

    # Pixel acc (debug)
    with torch.no_grad():
        pred_vals = (test_pred_angle.squeeze().cpu() / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0)
        pred_vals = pred_vals.clamp(0, VALUE_MAX).round().long()
        target_vals = test_target_angle[..., :test_size].squeeze().cpu()
        target_vals = (target_vals / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0).round().long()
        pixel_acc = (pred_vals == target_vals).float().mean().item()

    return total, primary, aux, pixel_acc


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    auxiliary_weight: float = 0.5,
    grad_clip: float = 1.0,
    writer: SummaryWriter = None,
    pad_to: int = 900,
    inner_steps: int = 1,
    inner_max_acc: float = 0.0,
) -> dict:
    """
    Train epoch with few-shot forces (built fresh each step).

    Inner-loop: each task is shown `inner_steps` times. If `inner_max_acc > 0`,
    early-exit when pixel_acc >= inner_max_acc for the test prediction.

    This is critical for few-shot learning: the model needs multiple gradient
    steps on the same task to internalize the transformation, otherwise gradients
    are too noisy and the model never converges to the invariant.
    """
    model.train()

    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    total_pixel_acc = 0.0
    num_batches = 0
    num_inner_steps = 0
    num_tasks_completed = 0  # tasks that reached inner_max_acc (if set)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        if 'test_output' not in batch:
            continue

        try:
            train_pairs, test_input, test_output = prepare_pairs(batch, device)
        except Exception as e:
            continue

        if test_output is None:
            continue

        # Tamaños reales
        test_h, test_w = test_output.shape[-2], test_output.shape[-1]
        test_size = test_h * test_w
        train_sizes = []
        for p in train_pairs:
            out = p['output']
            if out.dim() == 2:
                train_sizes.append(out.numel())
            else:
                train_sizes.append(out.shape[-1])

        # Targets en angulos (computados una sola vez por task)
        test_target_flat = test_output.flatten()
        if test_target_flat.numel() < pad_to:
            test_target_flat = F.pad(test_target_flat, (0, pad_to - test_target_flat.numel()), value=0)
        test_target_angle = value_to_angle(test_target_flat.unsqueeze(0))

        # Inner loop: ver el mismo task K veces
        task_completed = False
        last_pixel_acc = 0.0
        last_loss = 0.0

        for inner_step in range(inner_steps):
            num_inner_steps += 1
            optimizer.zero_grad()

            # Construir forces EN CADA STEP (evita graph reuse)
            try:
                forces, pred_timesteps, target_grids = build_fewshot_forces(
                    model, train_pairs, test_input, device=device
                )
            except Exception as e:
                pbar.set_postfix({'error': f'forces: {str(e)[:30]}'})
                break

            # Forward
            try:
                out = model(force_manual=forces)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
            except Exception as e:
                pbar.set_postfix({'error': f'fwd: {str(e)[:30]}'})
                break

            # Extract predictions
            predictions = extract_predictions(logits, pred_timesteps)
            train_preds = predictions[:-1]
            test_pred = predictions[-1]

            # Target en angulos para los train pairs
            target_grids_angle = []
            for g in target_grids:
                if g.dim() == 1:
                    if g.numel() < pad_to:
                        g = F.pad(g, (0, pad_to - g.numel()), value=0)
                    target_grids_angle.append(value_to_angle(g.unsqueeze(0)))
                else:
                    target_grids_angle.append(value_to_angle(g))

            # Loss
            try:
                loss, primary, aux, pixel_acc = compute_losses(
                    train_preds, target_grids_angle,
                    test_pred, test_target_angle,
                    test_size, train_sizes, auxiliary_weight, device
                )
            except Exception as e:
                pbar.set_postfix({'error': f'loss: {str(e)[:30]}'})
                break

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({'error': 'NaN'})
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_primary += primary.item()
            total_aux += aux.item()
            total_pixel_acc += pixel_acc
            num_batches += 1
            last_pixel_acc = pixel_acc
            last_loss = loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pix': f'{pixel_acc:.2%}',
                'inner': f'{inner_step+1}/{inner_steps}',
            })

            # Early-exit si alcanzamos el umbral
            if inner_max_acc > 0 and pixel_acc >= inner_max_acc:
                num_tasks_completed += 1
                task_completed = True
                break

        # Si el modelo alcanzo el umbral en algun inner step, lo contamos
        if task_completed and not (inner_max_acc > 0 and last_pixel_acc >= inner_max_acc):
            num_tasks_completed += 1

    metrics = {
        'train_loss': total_loss / max(num_batches, 1),
        'train_primary': total_primary / max(num_batches, 1),
        'train_aux': total_aux / max(num_batches, 1),
        'train_pixel_acc': total_pixel_acc / max(num_batches, 1),
        'num_train_tasks': num_batches // max(inner_steps, 1),
        'num_inner_steps': num_inner_steps,
        'num_tasks_completed': num_tasks_completed,
        'epoch': epoch,
    }

    if writer:
        writer.add_scalar('Train/Loss', metrics['train_loss'], epoch)
        writer.add_scalar('Train/Primary', metrics['train_primary'], epoch)
        writer.add_scalar('Train/Aux', metrics['train_aux'], epoch)
        writer.add_scalar('Train/PixelAcc', metrics['train_pixel_acc'], epoch)
        writer.add_scalar('Train/InnerSteps', num_inner_steps, epoch)
        writer.add_scalar('Train/TasksCompleted', num_tasks_completed, epoch)

    return metrics


def validate(
    model,
    dataloader,
    device: str,
    epoch: int,
    pad_to: int = 900,
    writer: SummaryWriter = None,
) -> dict:
    """Validation: few-shot forces + toroidal loss + ARC metrics."""
    model.eval()

    task_results = []
    total_loss = 0.0
    num_tasks = 0

    pbar = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for batch in pbar:
            if 'test_output' not in batch:
                continue

            try:
                train_pairs, test_input, test_output = prepare_pairs(batch, device)
            except Exception:
                continue

            if test_output is None:
                continue

            # Tamaños
            test_h, test_w = test_output.shape[-2], test_output.shape[-1]
            test_size = test_h * test_w
            train_sizes = []
            for p in train_pairs:
                out = p['output']
                if out.dim() == 2:
                    train_sizes.append(out.numel())
                else:
                    train_sizes.append(out.shape[-1])

            # Build forces (no_grad context)
            try:
                forces, pred_timesteps, target_grids = build_fewshot_forces(
                    model, train_pairs, test_input, device=device
                )
            except Exception:
                continue

            # Forward
            try:
                out = model(force_manual=forces)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
            except Exception:
                continue

            predictions = extract_predictions(logits, pred_timesteps)
            test_pred = predictions[-1]

            # Loss (mismo que train)
            test_target_flat = test_output.flatten()
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

            try:
                loss, primary, aux, _ = compute_losses(
                    predictions, target_grids_angle,
                    test_pred, test_target_angle,
                    test_size, train_sizes, 0.0, device  # no aux in val
                )
            except Exception:
                continue

            total_loss += loss.item()
            num_tasks += 1

            # ARC metrics: convertir angulos a enteros [0, 9]
            with torch.no_grad():
                pred_angle_slice = test_pred[..., :test_size]
                pred_vals = (pred_angle_slice.squeeze().cpu() / (math.pi / VALUE_MAX) + VALUE_MAX / 2.0)
                pred_vals = pred_vals.clamp(0, VALUE_MAX).round().long().numpy()
                pred_2d = pred_vals.reshape(test_h, test_w)

                gt = test_output.cpu().numpy().astype(np.int64)
                if gt.ndim == 3:
                    gt = gt[0]
                gt_2d = gt.reshape(test_h, test_w)

                metrics = ARCMetrics.evaluate_task(
                    pred_2d, gt_2d,
                    pred_size=(test_h, test_w),
                    true_size=(test_h, test_w),
                )
                task_results.append(metrics)

    aggregated = ARCMetrics.aggregate_metrics(task_results) if task_results else {
        'task_accuracy': 0.0,
        'mean_pixel_accuracy': 0.0,
        'size_accuracy': 0.0,
    }

    results = {
        'val_loss': total_loss / max(num_tasks, 1),
        'val_task_accuracy': aggregated['task_accuracy'],
        'val_mean_pixel_accuracy': aggregated['mean_pixel_accuracy'],
        'val_size_accuracy': aggregated['size_accuracy'],
        'num_val_tasks': num_tasks,
        'epoch': epoch,
    }

    if writer:
        writer.add_scalar('Val/Loss', results['val_loss'], epoch)
        writer.add_scalar('Val/TaskAccuracy', results['val_task_accuracy'], epoch)
        writer.add_scalar('Val/PixelAccuracy', results['val_mean_pixel_accuracy'], epoch)

    return results


# =============================================================================
# 5) Utils
# =============================================================================
def save_checkpoint(model, optimizer, epoch, metrics, model_config, output_dir, is_best=False):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model_config,
    }
    out = Path(output_dir) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out / f"checkpoint_epoch_{epoch}.pt")
    if is_best:
        torch.save(ckpt, out / "best_model.pt")
        print(f"  → Best model saved (task_acc={metrics.get('val_task_accuracy', 0):.2%})")


# =============================================================================
# 6) CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description='ARC-AGI-2 GSSM Training (ToroidalLoss)')
    p.add_argument('--data_path', type=str, default='../data/processed/splits')
    p.add_argument('--output_dir', type=str, default='../outputs/gssm_arc')
    p.add_argument('--config', type=str, default='tiny', choices=['tiny', 'small'])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--max_train_pairs', type=int, default=4)
    p.add_argument('--auxiliary_weight', type=float, default=0.5)
    p.add_argument('--physics_lr_scale', type=float, default=10.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--inner_steps', type=int, default=3,
                   help='K veces que se muestra cada task por epoch (inner-loop few-shot)')
    p.add_argument('--inner_max_acc', type=float, default=1.00,
                   help='Early-exit si pixel_acc >= este umbral (0 = desactivar)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=1)
    return p.parse_args()


# =============================================================================
# 7) Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "logs")

    config = get_config(args.config)
    with open(output_dir / "config.json", 'w') as f:
        json.dump({'args': vars(args), 'config': config}, f, indent=2)

    print("=" * 60)
    print(f"ARC-AGI-2 GSSM Training (ToroidalLoss + FunctionalEmbedding)")
    print("=" * 60)
    print(f"Config: {args.config} ({config['dim']}d, {config['depth']}L, {config['heads']}H)")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Aux: {args.auxiliary_weight}")
    print(f"Physics LR scale: {args.physics_lr_scale}")
    print(f"Inner-loop: {args.inner_steps} pasos/task, early-exit @ {args.inner_max_acc:.0%}")
    print("=" * 60)

    # Modelo
    print("\n[1/3] Creando modelo...")
    model = create_model(args.config, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Data
    print("\n[2/3] Cargando datos...")
    from src.data.arc_dataset import create_arc_dataloader
    train_loader = create_arc_dataloader(
        args.data_path, split="train",
        batch_size=args.batch_size, max_train_pairs=args.max_train_pairs,
    )
    val_loader = None
    val_path = Path(args.data_path) / "splits" / "val"
    if val_path.exists():
        val_loader = create_arc_dataloader(
            args.data_path, split="val",
            batch_size=1, max_train_pairs=args.max_train_pairs,
            shuffle_pairs=False,
        )
    print(f"  Train tasks: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Val tasks: {len(val_loader.dataset)}")

    # Optimizer (dual-group, mismo patron que XOR que funciona)
    optimizer = make_gfn_optimizer(
        model, lr=args.lr,
        physics_lr_scale=args.physics_lr_scale,
        weight_decay=1e-4,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # Loop
    print("\n[3/3] Entrenando...")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device=args.device, epoch=epoch,
            auxiliary_weight=args.auxiliary_weight,
            grad_clip=args.grad_clip, writer=writer,
            inner_steps=args.inner_steps,
            inner_max_acc=args.inner_max_acc,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} [{elapsed:.1f}s] | "
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"(pri={train_metrics['train_primary']:.4f}, aux={train_metrics['train_aux']:.4f}) | "
            f"Pixel Acc: {train_metrics['train_pixel_acc']:.2%} | "
            f"Tasks: {train_metrics['num_train_tasks']} | "
            f"Inner: {train_metrics['num_inner_steps']} | "
            f"@100%: {train_metrics['num_tasks_completed']}"
        )

        if val_loader:
            val_metrics = validate(
                model, val_loader, device=args.device, epoch=epoch, writer=writer
            )
            print(
                f"  Val   | Loss: {val_metrics['val_loss']:.4f} | "
                f"Task Acc: {val_metrics['val_task_accuracy']:.2%} | "
                f"Pixel Acc: {val_metrics['val_mean_pixel_accuracy']:.2%} | "
                f"Tasks: {val_metrics['num_val_tasks']}"
            )

            is_best = val_metrics['val_task_accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['val_task_accuracy']
                best_epoch = epoch

            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, config, output_dir, is_best=is_best
                )
        else:
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    model, optimizer, epoch, train_metrics, config, output_dir
                )

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best val_task_accuracy: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Outputs: {output_dir}")
    print("=" * 60)

    writer.close()


if __name__ == "__main__":
    main()

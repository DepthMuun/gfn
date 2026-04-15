"""
ARC-AGI-2 Training Script
Entrenamiento few-shot de GSSM en ARC-AGI-2.

Few-shot approach:
- Build a force sequence: [input_1, output_1, input_2, output_2, ..., test_input]
- Single forward pass through the model
- Readout at input positions should predict corresponding outputs (auxiliary loss)
- Readout at final position is the test prediction (primary loss)

All values are in [0, 9] range (ARC colors). No normalization to [0, 1].
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.parent
BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from src.data.arc_dataset import create_arc_dataloader, ARCAGI2Dataset
from src.models.gssm_config import create_arc_agi2_model, get_arc_agi2_config
from src.evaluation.metrics import ARCMetrics, crop_to_original
from src.training.few_shot import (
    build_fewshot_forces,
    extract_predictions,
    fewshot_loss,
    grid_loss,
    prediction_to_grid,
    get_model_embedding,
)
import gfn
# make_gfn_optimizer is in gssm training submodule
from gfn.realizations.gssm.training.optimizer import make_gfn_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GSSM on ARC-AGI-2")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ARC-AGI-2 data")
    parser.add_argument("--config", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_pairs", type=int, default=3, help="Max training examples per task")
    parser.add_argument("--auxiliary_weight", type=float, default=0.5, help="Weight for auxiliary train-pair loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    dataloader,
    optimizer,
    device: str,
    epoch: int,
    auxiliary_weight: float = 0.5,
    writer: SummaryWriter = None,
    scheduler = None
) -> dict:
    """
    Entrena una época con few-shot learning real.

    Para cada task, construye una secuencia de fuerzas:
    [input_1, output_1, input_2, output_2, ..., test_input]
    El modelo condiciona en los train pairs y predice el test output.
    """
    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        # Check that we have train pairs and test output
        train_pairs = batch['train_pairs']
        if not train_pairs:
            continue

        # Move batch to device
        test_input = batch['test_input'].to(device)
        test_output = batch.get('test_output', None)
        if test_output is not None:
            test_output = test_output.to(device)
        
        # Move train pairs to device
        for pair in train_pairs:
            pair['input'] = pair['input'].to(device)
            pair['output'] = pair['output'].to(device)

        # Build few-shot force sequence
        try:
            forces, pred_timesteps, target_grids = build_fewshot_forces(
                model, train_pairs, test_input, device=device
            )
        except Exception as e:
            print(f"  Warning: skipping task {batch.get('task_id', '?')}: {e}")
            continue

        # Forward pass with the force sequence
        logits, state, info = model(force_manual=forces)

        # Extract predictions at relevant timesteps
        predictions = extract_predictions(logits, pred_timesteps)

        # Train pair predictions (auxiliary) and test prediction (primary)
        train_preds = predictions[:-1]  # All except last
        test_pred = predictions[-1]     # Last one is test prediction

        # Compute loss
        if test_output is not None and len(target_grids) > 0:
            test_target = test_output.to(device)
            if test_target.dim() == 2:
                test_target_flat = test_target.flatten().to(device)
            else:
                test_target_flat = test_target.to(device)

            loss = fewshot_loss(
                predictions=train_preds,
                targets=target_grids,
                test_prediction=test_pred,
                test_target=test_target_flat,
                auxiliary_weight=auxiliary_weight
            )
        else:
            # No test output available - only auxiliary loss on train pairs
            if train_preds and target_grids:
                aux_losses = []
                for pred, target in zip(train_preds, target_grids):
                    target_batch = target.unsqueeze(0).to(device)
                    min_dim = min(pred.shape[-1], target_batch.shape[-1])
                    aux_losses.append(grid_loss(pred[..., :min_dim], target_batch[..., :min_dim]))
                loss = torch.stack(aux_losses).mean()
            else:
                continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    metrics = {
        'train_loss': avg_loss,
        'epoch': epoch
    }

    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)

    return metrics


def validate(
    model,
    dataloader,
    device: str,
    epoch: int,
    writer: SummaryWriter = None
) -> dict:
    """
    Validación con few-shot learning: condiciona en train pairs, predice test output.
    Evalúa sobre la región original del grid (sin padding).
    """
    model.eval()

    task_results = []
    total_loss = 0.0
    num_tasks = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if 'test_output' not in batch:
                continue

            train_pairs = batch['train_pairs']
            if not train_pairs:
                continue

            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            test_output_size = batch.get('test_output_size', None)
            test_input_size = batch.get('test_input_size', None)

            # Build few-shot force sequence
            try:
                forces, pred_timesteps, target_grids = build_fewshot_forces(
                    model, train_pairs, test_input, device=device
                )
            except Exception:
                continue

            # Forward
            logits, state, info = model(force_manual=forces)

            # Extract test prediction (last timestep)
            predictions = extract_predictions(logits, pred_timesteps)
            test_pred = predictions[-1]

            # Convert prediction to grid [0, 9]
            pred_flat = test_pred.squeeze().cpu()
            pred_grid = pred_flat.round().clamp(0, 9).to(torch.int64).numpy()

            # Determine original sizes
            if test_output_size is not None:
                out_h, out_w = test_output_size
            elif test_output.dim() == 2:
                out_h, out_w = test_output.shape
            else:
                out_h, out_w = 30, 30

            # Reshape prediction to 2D and crop to original size
            pred_2d = pred_grid.reshape(30, 30) if len(pred_grid) == 900 else pred_grid.reshape(out_h, out_w)
            pred_cropped = pred_2d[:out_h, :out_w]

            # Ground truth: crop to original size
            gt_np = test_output.cpu().numpy().round().astype(np.int64)
            gt_2d = gt_np.reshape(30, 30) if gt_np.size == 900 else gt_np.reshape(out_h, out_w)
            gt_cropped = gt_2d[:out_h, :out_w]

            # Loss
            gt_flat = test_output.flatten()
            if gt_flat.dim() == 0:
                gt_flat = gt_flat.unsqueeze(0)
            min_dim = min(test_pred.shape[-1], gt_flat.shape[-1])
            loss = grid_loss(test_pred[..., :min_dim].squeeze(), gt_flat[..., :min_dim].squeeze())
            total_loss += loss.item()

            # Evaluate with ARC metrics on original region
            metrics = ARCMetrics.evaluate_task(
                pred_cropped, gt_cropped,
                pred_size=(out_h, out_w),
                true_size=(out_h, out_w)
            )
            task_results.append(metrics)
            num_tasks += 1

    aggregated = ARCMetrics.aggregate_metrics(task_results)
    avg_loss = total_loss / num_tasks if num_tasks > 0 else 0.0

    results = {
        'val_loss': avg_loss,
        'val_task_accuracy': aggregated['task_accuracy'],
        'val_mean_pixel_accuracy': aggregated['mean_pixel_accuracy'],
        'val_size_accuracy': aggregated['size_accuracy'],
        'num_val_tasks': num_tasks,
        'epoch': epoch
    }

    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/TaskAccuracy', aggregated['task_accuracy'], epoch)
        writer.add_scalar('Val/PixelAccuracy', aggregated['mean_pixel_accuracy'], epoch)

    return results


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, output_dir: Path, is_best: bool = False):
    """Guarda checkpoint del modelo."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config if hasattr(model, 'config') else {}
    }

    checkpoint_path = output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save as best if this is the best model so far
    if is_best:
        best_path = output_dir / "checkpoints" / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path}")


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Guardar config
    config_dict = {
        'args': vars(args),
        'model_config': get_arc_agi2_config()
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 60)
    print("ARC-AGI-2 GSSM Training (Few-Shot)")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Max train pairs: {args.max_train_pairs}")
    print(f"Auxiliary weight: {args.auxiliary_weight}")
    print("=" * 60)

    # Crear modelo con config preset
    print("\nCreating model...")
    from src.models.gssm_config import get_config
    model_config = get_config(args.config)
    model = create_arc_agi2_model(config=model_config, device=args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: {model_config['dim']}d, {model_config['depth']}L, {model_config['heads']}H")

    # Data loaders (before optimizer/scheduler)
    print("\nLoading data...")
    train_loader = create_arc_dataloader(
        args.data_path,
        split="train",
        batch_size=args.batch_size,
        max_train_pairs=args.max_train_pairs
    )

    val_loader = None
    if (Path(args.data_path) / "eval").exists():
        val_loader = create_arc_dataloader(
            args.data_path,
            split="eval",
            batch_size=1,
            max_train_pairs=args.max_train_pairs,
            shuffle_pairs=False
        )

    print(f"Train tasks: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val tasks: {len(val_loader.dataset)}")

    # Optimizer
    optimizer = make_gfn_optimizer(
        model,
        lr=args.lr,
        physics_lr_scale=10.0,
        weight_decay=1e-4
    )

    # Scheduler (XOR-proven: OneCycleLR)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            auxiliary_weight=args.auxiliary_weight, writer=writer,
            scheduler=scheduler
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, args.device, epoch, writer)
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Task Accuracy: {val_metrics['val_task_accuracy']:.2%}")
            print(f"  Val Pixel Accuracy: {val_metrics['val_mean_pixel_accuracy']:.2%}")

            # Track best model by maximum accuracy (not arbitrary threshold)
            is_best = val_metrics['val_task_accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['val_task_accuracy']

            if epoch % args.save_every == 0 or is_best:
                save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, is_best=is_best)
        else:
            if epoch % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, train_metrics, output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print("=" * 60)

    writer.close()


if __name__ == "__main__":
    main()

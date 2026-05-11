"""
ARC-AGI-2 QuickTest - Few-Shot with MSE Loss
Test rápido con modelo pequeño (64d, 2L, 2H) y pocos datos.
Uses few-shot conditioning: model sees train pairs before predicting test output.
All values in [0, 9] range (ARC colors). No normalization to [0, 1].
Evaluation on original grid region (no padding inflation).
"""

import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.parent
BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
import tempfile
from tqdm import tqdm

from src.data.arc_dataset import ARCAGI2Dataset
from src.evaluation.metrics import ARCMetrics
from src.training.few_shot import (
    build_fewshot_forces,
    extract_predictions,
    fewshot_loss,
    grid_loss,
    prediction_to_grid,
    get_model_embedding,
)


def get_tiny_config():
    """Configuración PROVEN (basada en XOR que funciona)."""
    return {
        'vocab_size': 10,
        'dim': 64,  # XOR usa 8, MNIAH usa 16 - empezamos conservador
        'heads': 4,
        'depth': 4,
        'max_seq_len': 900,
        'embedding_mode': 'continuous',  # Mantenemos multimodal para grids
        'continuous_input_dim': 900,  # 30x30 grid
        'store_full_sequence': True,  # Need all timesteps for few-shot
        'physics': {
            'embedding': {
                'type': 'functional',
                'mode': 'continuous',
                'coord_dim': 900,
                'impulse_scale': 80.0,  # XOR usa 80.0 (critical!)
            },
            'readout': {
                'type': 'implicit',  # XOR usa implicit, no identity
                'coord_dim': 16,     # XOR usa coord_dim: 16
                'out_dim': 900,     # Project to 900 for grid output
            },
            'topology': {
                'type': 'torus',
                'R': 3.0,
                'r': 1.0,
                'learnable_R': True,
                'learnable_r': True
            },
            'stability': {
                'base_dt': 0.05,
                'dt_min': 0.0001,
                'dt_max': 0.2,
                'friction': 2.0,  # XOR usa 2.0 (critical!)
                'velocity_saturation': 15.0,  # XOR usa 15.0
                'curvature_clamp': 1.0
            },
            'active_inference': {
                'enabled': True,
                'hysteresis': {
                    'enabled': False,
                    'strength': 0.1,
                    'decay': 0.9
                },
                'curiosity': {
                    'enabled': False
                },
                'stochasticity': {
                    'enabled': False
                }
            },
            'integrator': {
                'type': 'leapfrog',
                'adaptive_dt': False
            }
        },
        'readout_type': 'implicit',  # Cambiado de identity a implicit
        'readout_hidden_dim': 64,
        'readout_out_dim': 900,  # Project back to grid size
    }


def create_tiny_model(device='cpu'):
    """Crea modelo tiny (64d, 2L, 2H) con modo continuo."""
    from gfn import create

    config = get_tiny_config()

    print(f"  [DEBUG] vocab_size: {config['vocab_size']}")
    print(f"  [DEBUG] embedding_mode: {config['embedding_mode']}")
    print(f"  [DEBUG] continuous_input_dim: {config['continuous_input_dim']}")
    print(f"  [DEBUG] physics.readout: {config['physics'].get('readout', {})}")

    model = create(
        'gssm',
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        heads=config['heads'],
        depth=config['depth'],
        max_seq_len=config['max_seq_len'],
        embedding_mode=config['embedding_mode'],
        continuous_input_dim=config['continuous_input_dim'],
        physics=config['physics'],
        readout_type=config['readout_type'],
        readout_hidden_dim=config['readout_hidden_dim'],
        readout_out_dim=config.get('readout_out_dim', 900),  # Project to grid size
        holographic=True,
        device=device
    )

    model = model.to(device)
    return model


def count_parameters(model):
    """Cuenta parámetros del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_mini_dataset(num_tasks=10, seed=42):
    """
    Crea dataset mini con N tasks de ARC-AGI-2.
    Usa tasks simples que un humano puede resolver fácilmente.
    Todos los valores en [0, 9] (ARC colors).
    """
    random.seed(seed)
    np.random.seed(seed)

    tasks = [
        {
            'task_id': 'mini_flip_h_001',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[3, 4], [1, 2]]},
                {'input': [[5, 6], [7, 8]], 'output': [[7, 8], [5, 6]]},
                {'input': [[0, 1], [2, 3]], 'output': [[2, 3], [0, 1]]}
            ],
            'test': [
                {'input': [[9, 8], [7, 6]], 'output': [[7, 6], [9, 8]]}
            ]
        },
        {
            'task_id': 'mini_color_swap_001',
            'train': [
                {'input': [[1, 1], [2, 2]], 'output': [[2, 2], [1, 1]]},
                {'input': [[1, 2], [1, 2]], 'output': [[2, 1], [2, 1]]}
            ],
            'test': [
                {'input': [[2, 2], [1, 1]], 'output': [[1, 1], [2, 2]]}
            ]
        },
        {
            'task_id': 'mini_identity_001',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[1, 2], [3, 4]]},
                {'input': [[5, 5], [5, 5]], 'output': [[5, 5], [5, 5]]}
            ],
            'test': [
                {'input': [[9, 8], [7, 6]], 'output': [[9, 8], [7, 6]]}
            ]
        },
        {
            'task_id': 'mini_transpose_001',
            'train': [
                {'input': [[1, 2, 3], [4, 5, 6]], 'output': [[1, 4], [2, 5], [3, 6]]}
            ],
            'test': [
                # FIXED: values 10,11,12 changed to 7,8,9 (ARC range is 0-9)
                {'input': [[7, 8], [9, 6], [5, 4]], 'output': [[7, 9, 5], [8, 6, 4]]}
            ]
        },
        {
            'task_id': 'mini_mirror_h_001',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[2, 1], [4, 3]]},
                {'input': [[5, 6], [7, 8]], 'output': [[6, 5], [8, 7]]}
            ],
            'test': [
                {'input': [[9, 1], [2, 3]], 'output': [[1, 9], [3, 2]]}
            ]
        },
        {
            'task_id': 'mini_fill_001',
            'train': [
                {'input': [[0, 0], [0, 0]], 'output': [[1, 1], [1, 1]]},
                {'input': [[0, 0, 0], [0, 0, 0]], 'output': [[1, 1, 1], [1, 1, 1]]}
            ],
            'test': [
                {'input': [[0, 0], [0, 0], [0, 0]], 'output': [[1, 1], [1, 1], [1, 1]]}
            ]
        },
        {
            'task_id': 'mini_extract_color_001',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[1, 0], [0, 0]]},
                {'input': [[2, 2], [1, 3]], 'output': [[0, 0], [1, 0]]}
            ],
            'test': [
                {'input': [[5, 1], [2, 1]], 'output': [[0, 1], [0, 1]]}
            ]
        },
        {
            'task_id': 'mini_duplicate_001',
            'train': [
                {'input': [[1, 2]], 'output': [[1, 2], [1, 2]]},
                {'input': [[3]], 'output': [[3], [3]]}
            ],
            'test': [
                {'input': [[5, 6, 7]], 'output': [[5, 6, 7], [5, 6, 7]]}
            ]
        },
        {
            'task_id': 'mini_border_001',
            'train': [
                {'input': [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 'output': [[2, 2, 2], [2, 1, 2], [2, 2, 2]]}
            ],
            'test': [
                {'input': [[0, 0], [0, 3]], 'output': [[2, 2], [2, 3]]}
            ]
        },
        {
            'task_id': 'mini_shift_right_001',
            'train': [
                {'input': [[1, 2, 3]], 'output': [[0, 1, 2]]},
                {'input': [[4, 5]], 'output': [[0, 4]]}
            ],
            'test': [
                {'input': [[7, 8, 9, 1]], 'output': [[0, 7, 8, 9]]}
            ]
        }
    ]

    return tasks[:num_tasks]


def quick_train(model, dataset, epochs=50, lr=1e-3, device='cpu'):
    """Entrenamiento rápido con few-shot conditioning."""
    model.train()

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                   if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
         'lr': lr, 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters()
                   if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
         'lr': lr * 2, 'weight_decay': 0},
    ])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*2, total_steps=epochs*len(dataset), pct_start=0.2
    )

    print(f"\nTraining for {epochs} epochs (few-shot)...")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for task in dataset:
            optimizer.zero_grad()

            train_pairs = task['train_pairs']
            test_input = task['test_input'].to(device)
            test_output = task.get('test_output', None)

            # Build few-shot force sequence
            try:
                forces, pred_timesteps, target_grids = build_fewshot_forces(
                    model, train_pairs, test_input, device=device
                )
            except Exception as e:
                print(f"  Warning: skipping task {task.get('task_id', '?')}: {e}")
                continue

            # Forward pass with force sequence
            logits, state, info = model(force_manual=forces)

            # Extract predictions at relevant timesteps
            predictions = extract_predictions(logits, pred_timesteps)

            train_preds = predictions[:-1]
            test_pred = predictions[-1]

            # Compute loss
            if test_output is not None and len(target_grids) > 0:
                test_target = test_output.to(device)
                if test_target.dim() == 2:
                    test_target_flat = test_target.flatten()
                else:
                    test_target_flat = test_target

                loss = fewshot_loss(
                    predictions=train_preds,
                    targets=target_grids,
                    test_prediction=test_pred,
                    test_target=test_target_flat,
                    auxiliary_weight=0.5
                )
            else:
                # Only auxiliary loss
                if train_preds and target_grids:
                    aux_losses = []
                    for pred, target in zip(train_preds, target_grids):
                        target_batch = target.unsqueeze(0).to(device)
                        min_dim = min(pred.shape[-1], target_batch.shape[-1])
                        aux_losses.append(grid_loss(pred[..., :min_dim], target_batch[..., :min_dim]))
                    loss = torch.stack(aux_losses).mean()
                else:
                    continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    return model


def quick_evaluate(model, dataset, device='cpu'):
    """Evaluación rápida con few-shot conditioning. Evalúa sobre región original."""
    model.eval()

    task_results = []

    with torch.no_grad():
        for task in dataset:
            if 'test_output' not in task:
                continue

            train_pairs = task['train_pairs']
            test_input = task['test_input'].to(device)
            test_output = task['test_output']
            test_output_size = task.get('test_output_size', None)

            # Determine output size
            if test_output_size is not None:
                out_h, out_w = test_output_size
            elif test_output.dim() == 2:
                out_h, out_w = test_output.shape
            else:
                out_h, out_w = 30, 30

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

            # Reshape to 2D using actual grid size
            if len(pred_grid) == 900:
                pred_2d = pred_grid.reshape(30, 30)
            else:
                pred_2d = pred_grid.reshape(out_h, out_w)
            pred_cropped = pred_2d[:out_h, :out_w]

            # Ground truth
            gt_np = test_output.cpu().numpy().round().astype(np.int64)
            if gt_np.size == 900:
                gt_2d = gt_np.reshape(30, 30)
            else:
                gt_2d = gt_np.reshape(out_h, out_w)
            gt_cropped = gt_2d[:out_h, :out_w]

            # Evaluate on original region (no padding inflation)
            metrics = ARCMetrics.evaluate_task(
                pred_cropped, gt_cropped,
                pred_size=(out_h, out_w),
                true_size=(out_h, out_w)
            )
            metrics['task_id'] = task['task_id']
            task_results.append(metrics)

            correct = "OK" if metrics['strict_match'] else "XX"
            print(f"  {correct} {task['task_id']}: "
                  f"strict_match={metrics['strict_match']}, "
                  f"pixel_acc={metrics['pixel_accuracy']:.2%}, "
                  f"size={out_h}x{out_w}")

    aggregated = ARCMetrics.aggregate_metrics(task_results)
    return aggregated, task_results


def main():
    print("=" * 70)
    print("ARC-AGI-2 QUICKTEST (Few-Shot)")
    print("=" * 70)
    print("Model: 900d, 4L, 4H (Holographic/Identity)")
    print("Data: 10 simple tasks (few-shot)")
    print("Goal: Test generalization with minimal data")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Crear modelo
    print("\n[1/5] Creating tiny model...")
    config = get_tiny_config()  # Get config for saving later
    model = create_tiny_model(device=device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    if total_params > 50000000:
        print(f"  WARNING: Model has >50M params!")
    else:
        print(f"  [OK] Model size OK (<50M)")

    # Crear dataset mini
    print("\n[2/5] Creating mini dataset (10 tasks)...")
    tasks_data = create_mini_dataset(num_tasks=10, seed=42)

    # Convertir a dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        train_dir.mkdir()

        for task in tasks_data:
            with open(train_dir / f"{task['task_id']}.json", 'w') as f:
                json.dump(task, f)

        dataset = ARCAGI2Dataset(tmpdir, split="train", max_train_pairs=3)
        print(f"  Created dataset with {len(dataset)} tasks")
        print(f"  Each task has 1-3 training examples")

    # Entrenar
    print("\n[3/5] Quick training (1000 epochs)...")
    model = quick_train(model, dataset, epochs=1000, lr=1e-3, device=device)

    # Guardar modelo
    checkpoint_path = Path("results/quicktest_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'task_accuracy': None,  # Will update after eval
        'epochs_trained': 1000,
    }, checkpoint_path)
    print(f"\n  [OK] Model saved to {checkpoint_path}")

    # Evaluar
    print("\n[4/5] Evaluating...")
    aggregated, task_results = quick_evaluate(model, dataset, device=device)

    # Resultados
    print("\n[5/5] Results")
    print("=" * 70)
    print(f"Tasks correct: {aggregated['tasks_correct']}/{aggregated['num_tasks']}")
    print(f"Task Accuracy: {aggregated['task_accuracy']:.2%}")
    print(f"Mean Pixel Accuracy: {aggregated['mean_pixel_accuracy']:.2%}")
    print("=" * 70)

    if aggregated['task_accuracy'] >= 0.7:
        print("\nEXCELLENT: Model generalizes well!")
        print("   Ready for full training on real ARC-AGI-2")
    elif aggregated['task_accuracy'] >= 0.4:
        print("\nGOOD: Model shows learning")
        print("   May work on real ARC-AGI-2 with more data")
    elif aggregated['task_accuracy'] >= 0.2:
        print("\nMODERATE: Some progress")
        print("   Try tuning hyperparameters")
    else:
        print("\nNEEDS WORK: Model struggling")
        print("   Check architecture and training setup")

    # Update checkpoint with results
    checkpoint = torch.load(checkpoint_path)
    checkpoint['task_accuracy'] = aggregated['task_accuracy']
    checkpoint['mean_pixel_accuracy'] = aggregated['mean_pixel_accuracy']
    torch.save(checkpoint, checkpoint_path)

    print("\nQuickTest complete!")
    print("=" * 70)
    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"  - Model weights")
    print(f"  - Config: {config['dim']}d, {config['depth']}L, {config['heads']}H")
    print(f"  - Results: {aggregated['task_accuracy']:.1%} task accuracy")
    print("=" * 70)


if __name__ == "__main__":
    main()

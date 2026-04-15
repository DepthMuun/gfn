"""
ARC-AGI-2 Evaluation Script
Evaluación verificable de GSSM en ARC-AGI-2 con métricas correctas.

Few-shot approach: conditions on train pairs before predicting test output.
All values in [0, 9] range. Evaluation on original grid region (no padding inflation).
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.parent
BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from src.data.arc_dataset import create_arc_dataloader
from src.models.gssm_config import create_arc_agi2_model
from src.evaluation.metrics import ARCMetrics, save_predictions, load_predictions
from src.training.few_shot import (
    build_fewshot_forces,
    extract_predictions,
    prediction_to_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSSM on ARC-AGI-2")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ARC-AGI-2 data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "eval", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/predictions")
    parser.add_argument("--max_train_pairs", type=int, default=3)
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Carga modelo desde checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_arc_agi2_model(device=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return model, checkpoint


def evaluate_model(
    model,
    dataloader,
    device: str,
    split: str = "test",
    save_preds: bool = False,
    output_dir: Path = None
) -> dict:
    """
    Evalúa el modelo en el dataset usando few-shot conditioning.
    Evalúa sobre la región original del grid (sin padding).

    Returns:
        Dict con métricas y resultados detallados.
    """
    model.eval()

    task_results = []
    predictions = {}
    original_sizes = {}

    print(f"\nEvaluating on {len(dataloader.dataset)} tasks...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            task_id = batch['task_id']

            train_pairs = batch['train_pairs']
            test_input = batch['test_input'].to(device)
            test_input_size = batch.get('test_input_size', None)
            test_output_size = batch.get('test_output_size', None)

            # Determine original output size
            if test_output_size is not None:
                out_h, out_w = test_output_size
            elif 'test_output' in batch:
                out_h, out_w = batch['test_output'].shape
            else:
                out_h, out_w = 30, 30

            # Build few-shot force sequence
            try:
                forces, pred_timesteps, target_grids = build_fewshot_forces(
                    model, train_pairs, test_input, device=device
                )
            except Exception as e:
                print(f"  Warning: skipping task {task_id}: {e}")
                continue

            # Forward pass
            logits, state, info = model(force_manual=forces)

            # Extract test prediction (last timestep)
            predictions_list = extract_predictions(logits, pred_timesteps)
            test_pred = predictions_list[-1]

            # Convert prediction to grid [0, 9]
            pred_flat = test_pred.squeeze().cpu()
            pred_grid = pred_flat.round().clamp(0, 9).to(torch.int64).numpy()

            # Reshape and crop to original size
            if len(pred_grid) == 900:
                pred_2d = pred_grid.reshape(30, 30)
            else:
                pred_2d = pred_grid.reshape(out_h, out_w)
            pred_cropped = pred_2d[:out_h, :out_w]

            # Save prediction with original size
            predictions[task_id] = pred_2d
            original_sizes[task_id] = (out_h, out_w)

            # If ground truth available, evaluate
            if 'test_output' in batch:
                test_output = batch['test_output'].cpu().numpy().round().astype(np.int64)
                if test_output.size == 900:
                    gt_2d = test_output.reshape(30, 30)
                else:
                    gt_2d = test_output.reshape(out_h, out_w)
                gt_cropped = gt_2d[:out_h, :out_w]

                metrics = ARCMetrics.evaluate_task(
                    prediction=pred_cropped,
                    ground_truth=gt_cropped,
                    pred_size=(out_h, out_w),
                    true_size=(out_h, out_w)
                )
                metrics['task_id'] = task_id
                task_results.append(metrics)

    # Aggregate metrics
    if task_results:
        aggregated = ARCMetrics.aggregate_metrics(task_results)
    else:
        aggregated = {
            'task_accuracy': 0.0,
            'tasks_correct': 0,
            'mean_pixel_accuracy': 0.0,
            'size_accuracy': 0.0,
            'num_tasks': len(predictions)
        }

    results = {
        'metrics': aggregated,
        'task_results': task_results,
        'num_tasks': len(predictions)
    }

    # Save predictions
    if save_preds and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = output_dir / f"predictions_{split}.json"
        save_predictions(predictions, str(predictions_path), original_sizes)
        print(f"\nPredictions saved to: {predictions_path}")

        metrics_path = output_dir / f"metrics_{split}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'aggregated': aggregated,
                'per_task': task_results
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    return results


def print_results(results: dict):
    """Imprime resultados de evaluación."""
    metrics = results['metrics']

    print("\n" + "=" * 60)
    print("ARC-AGI-2 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of tasks: {results['num_tasks']}")
    print("")
    print("METRICS (Official)")
    print("-" * 60)
    print(f"Task Accuracy:       {metrics['task_accuracy']:.2%} ({metrics['tasks_correct']}/{metrics['num_tasks']})")
    print("")
    print("METRICS (Informative)")
    print("-" * 60)
    print(f"Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.2%}")
    print(f"Size Accuracy:       {metrics['size_accuracy']:.2%}")
    print("=" * 60)

    if metrics['task_accuracy'] >= 0.5:
        print("EXCELLENT: Model solved >50% of tasks!")
    elif metrics['task_accuracy'] >= 0.2:
        print("GOOD: Model shows strong learning")
    elif metrics['task_accuracy'] >= 0.05:
        print("MODERATE: Some progress, room for improvement")
    else:
        print("NEEDS WORK: Model struggling with ARC tasks")

    print("=" * 60)


def verify_predictions(predictions_path: str, ground_truth_path: str):
    """
    Verifica que las predicciones sean correctas comparando con ground truth.
    Util para auditoría externa.
    """
    print("\n" + "=" * 60)
    print("VERIFYING PREDICTIONS")
    print("=" * 60)

    with open(predictions_path, 'r') as f:
        pred_data = json.load(f)

    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    correct_count = 0
    total_count = 0

    for task_id in pred_data:
        if task_id not in gt_data:
            print(f"Warning: Task {task_id} not in ground truth")
            continue

        pred_grid = np.array(pred_data[task_id]['prediction'])
        gt_grid = np.array(gt_data[task_id]['output'])

        strict_match = ARCMetrics.strict_match(pred_grid, gt_grid)

        if strict_match:
            correct_count += 1

        total_count += 1

        if not strict_match:
            print(f"Task {task_id}: Mismatch")
            print(f"  Pred shape: {pred_grid.shape}, GT shape: {gt_grid.shape}")
            if pred_grid.shape == gt_grid.shape:
                diff_pixels = np.sum(pred_grid != gt_grid)
                print(f"  Different pixels: {diff_pixels}")

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"\nVerification complete:")
    print(f"  Tasks correct: {correct_count}/{total_count}")
    print(f"  Accuracy: {accuracy:.2%}")
    print("=" * 60)

    return accuracy


def main():
    args = parse_args()

    print("=" * 60)
    print("ARC-AGI-2 GSSM Evaluation")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Cargar modelo
    model, checkpoint = load_model(args.checkpoint, args.device)

    # Crear dataloader
    dataloader = create_arc_dataloader(
        args.data_path,
        split=args.split,
        batch_size=1,
        max_train_pairs=args.max_train_pairs,
        shuffle_pairs=False
    )

    # Evaluar (pass split explicitly, no global args)
    results = evaluate_model(
        model,
        dataloader,
        args.device,
        split=args.split,
        save_preds=args.save_predictions,
        output_dir=Path(args.output_dir)
    )

    # Imprimir resultados
    print_results(results)

    # Guardar resultados completos
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"summary_{args.split}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'split': args.split,
            'num_tasks': results['num_tasks'],
            'metrics': results['metrics'],
            'task_results': results['task_results']
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

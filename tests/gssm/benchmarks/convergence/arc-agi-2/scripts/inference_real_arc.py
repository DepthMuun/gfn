"""
Inference script for real ARC-AGI-2 data
Loads a trained checkpoint and evaluates on real ARC-AGI-2 tasks
"""
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch

# Paths
HERE = Path(__file__).parent
BENCHMARK_ROOT = HERE.parent
PROJECT_ROOT = BENCHMARK_ROOT.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from src.data.arc_dataset import ARCAGI2Dataset, create_arc_dataloader
from src.evaluation.metrics import ARCMetrics
# Import gfn public API only
import gfn
from src.training.few_shot import build_fewshot_forces, extract_predictions


def find_arc_dataset():
    """Find ARC-AGI-2 dataset in common locations."""
    possible_paths = [
        BENCHMARK_ROOT / "arc_agi_2_data",
        BENCHMARK_ROOT / "data" / "arc_agi_2",
        BENCHMARK_ROOT / ".." / ".." / "arc_agi_2_data",
        Path("C:/Users/joaqu/Downloads/arc-agi-2"),
        Path("C:/Users/joaqu/Downloads/arc-agi-2-master"),
        Path("D:/arc-agi-2"),
    ]
    
    for path in possible_paths:
        if not path.is_dir():
            continue
        # Check for training subdirectory
        train_dir = path / "data" / "training"
        if train_dir.exists() and any(train_dir.glob("*.json")):
            return path
        # Fallback to train subdirectory
        train_dir = path / "train"
        if train_dir.exists() and any(train_dir.glob("*.json")):
            return path
    
    return None


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    from gfn import create
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint (required)
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint missing 'config'. Please use a checkpoint saved with quicktest.py")
    config = checkpoint['config']
    
    # Create model with same config
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
        readout_hidden_dim=config.get('readout_hidden_dim', 64),
        readout_out_dim=config.get('readout_out_dim', 900),
        holographic=True,
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Config: {config['dim']}d, {config['depth']}L, {config['heads']}H")
    print(f"  Readout: {config['physics']['readout']['type']}")
    
    return model, config


def inference_task(model, task_data, device='cpu'):
    """Run inference on a single task."""
    train_pairs = task_data['train_pairs']
    test_input = task_data['test_input'].to(device)
    test_output = task_data.get('test_output', None)
    
    # Build few-shot forces
    from src.training.few_shot import build_fewshot_forces
    forces, pred_timesteps, _ = build_fewshot_forces(
        model, train_pairs, test_input, device=device
    )
    
    # Forward pass
    with torch.no_grad():
        logits, state, info = model(force_manual=forces)
    
    # Extract test prediction
    predictions = extract_predictions(logits, pred_timesteps)
    test_pred = predictions[-1]
    
    # Convert to grid
    pred_grid = test_pred.squeeze().cpu().numpy()
    pred_grid = pred_grid.round().clip(0, 9).astype(np.int64)
    
    # Reshape if needed (model outputs 900, may need to reshape to actual size)
    if len(pred_grid) == 900:
        # Try to infer actual grid size from test_output
        if test_output is not None:
            target_shape = test_output.shape
            if len(target_shape) == 2:
                h, w = target_shape
                if h * w == 900:
                    pred_grid = pred_grid.reshape(h, w)
                else:
                    # Pad or crop
                    pred_grid_2d = pred_grid.reshape(30, 30)
                    pred_grid = pred_grid_2d[:h, :w]
            else:
                pred_grid = pred_grid.reshape(30, 30)
        else:
            pred_grid = pred_grid.reshape(30, 30)
    
    return pred_grid


def evaluate_on_real_data(checkpoint_path: str, data_path: str = None, 
                           max_tasks: int = None, device: str = 'cpu'):
    """Evaluate model on real ARC-AGI-2 data."""
    
    # Load model
    model, config = load_checkpoint(checkpoint_path, device)
    
    # Find or use provided data path
    if data_path is None:
        data_path = find_arc_dataset()
        if data_path is None:
            print("ERROR: Could not find ARC-AGI-2 dataset")
            print("Please provide path with: --data_path /path/to/arc-agi-2")
            print("Expected structure: /path/to/arc-agi-2/train/*.json")
            return None
    
    data_path = Path(data_path)
    print(f"\nLoading ARC-AGI-2 data from: {data_path}")
    
    # Load dataset - handle both 'train' and 'data/training' structures
    try:
        # Try default structure first (train/ subdir)
        dataset = ARCAGI2Dataset(data_path, split="train", max_train_pairs=5)
    except Exception:
        # Fallback to data/training structure
        try:
            from src.data.arc_dataset import ARCAGI2Dataset as ArcDatasetRaw
            # Load directly from data/training
            training_path = data_path / "data" / "training"
            if training_path.exists():
                # Create custom dataset loader for this structure
                import json
                import os
                from torch.utils.data import Dataset
                
                class SimpleARCDataset(Dataset):
                    def __init__(self, training_path, max_train_pairs=5):
                        self.training_path = Path(training_path)
                        self.max_train_pairs = max_train_pairs
                        self.task_files = sorted(self.training_path.glob("*.json"))
                    
                    def __len__(self):
                        return len(self.task_files)
                    
                    def __getitem__(self, idx):
                        import numpy as np
                        task_file = self.task_files[idx]
                        task_id = task_file.stem
                        
                        with open(task_file, 'r') as f:
                            task_data = json.load(f)
                        
                        # Parse train pairs
                        train_pairs = []
                        for pair in task_data.get('train', [])[:self.max_train_pairs]:
                            input_grid = np.array(pair['input'])
                            output_grid = np.array(pair['output'])
                            # Normalize to [0,9]
                            input_grid = np.clip(input_grid, 0, 9)
                            output_grid = np.clip(output_grid, 0, 9)
                            train_pairs.append({
                                'input': torch.tensor(input_grid, dtype=torch.float32),
                                'output': torch.tensor(output_grid, dtype=torch.float32)
                            })
                        
                        # Parse test (first test sample)
                        test_data = task_data.get('test', [{}])[0]
                        test_input = np.array(test_data.get('input', [[]]))
                        test_output = np.array(test_data.get('output', [[]]))
                        test_input = np.clip(test_input, 0, 9)
                        test_output = np.clip(test_output, 0, 9)
                        
                        return {
                            'task_id': task_id,
                            'train_pairs': train_pairs,
                            'test_input': torch.tensor(test_input, dtype=torch.float32),
                            'test_output': torch.tensor(test_output, dtype=torch.float32),
                            'num_train_pairs': len(train_pairs)
                        }
                
                dataset = SimpleARCDataset(training_path, max_train_pairs=5)
            else:
                raise ValueError(f"No training data found in {data_path}")
        except Exception as e2:
            print(f"ERROR loading dataset: {e2}")
            return None
    
    if len(dataset) == 0:
        print("ERROR: No tasks found in dataset")
        return None
    
    print(f"Found {len(dataset)} tasks")
    if max_tasks:
        print(f"Evaluating on first {max_tasks} tasks")
    
    # Evaluate
    results = []
    model.eval()
    
    with torch.no_grad():
        for i, task in enumerate(dataset):
            if max_tasks and i >= max_tasks:
                break
            
            task_id = task.get('task_id', f'task_{i:03d}')
            
            try:
                # Run inference
                pred_grid = inference_task(model, task, device)
                
                # Get ground truth
                test_output = task.get('test_output')
                if test_output is None:
                    print(f"  {task_id}: No test output (skipping)")
                    continue
                
                target = test_output.numpy() if hasattr(test_output, 'numpy') else test_output
                
                # Evaluate
                metrics = ARCMetrics.evaluate_task(pred_grid, target)
                
                result = {
                    'task_id': task_id,
                    'strict_match': metrics['strict_match'],
                    'pixel_accuracy': metrics['pixel_accuracy'],
                    'size_correct': metrics['size_correct'],
                    'task_size': f"{target.shape[0]}x{target.shape[1]}"
                }
                results.append(result)
                
                status = "✓" if metrics['strict_match'] else "✗"
                print(f"  {status} {task_id}: pixel_acc={metrics['pixel_accuracy']:.1%}, "
                      f"strict_match={metrics['strict_match']}, size={target.shape[0]}x{target.shape[1]}")
                
            except Exception as e:
                print(f"  ✗ {task_id}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate results
    if not results:
        print("\nNo results to aggregate")
        return None
    
    strict_matches = sum(1 for r in results if r['strict_match'])
    mean_pixel_acc = np.mean([r['pixel_accuracy'] for r in results])
    
    print("\n" + "=" * 70)
    print("REAL ARC-AGI-2 INFERENCE RESULTS")
    print("=" * 70)
    print(f"Tasks evaluated: {len(results)}")
    print(f"Task Accuracy: {strict_matches}/{len(results)} = {strict_matches/len(results):.1%}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.1%}")
    print("=" * 70)
    
    return {
        'num_tasks': len(results),
        'task_accuracy': strict_matches / len(results),
        'mean_pixel_accuracy': mean_pixel_acc,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='ARC-AGI-2 Inference on Real Data')
    parser.add_argument('--checkpoint', type=str, 
                        default='results/quicktest_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to ARC-AGI-2 data directory (contains train/ subdir)')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Maximum number of tasks to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ARC-AGI-2 REAL DATA INFERENCE")
    print("=" * 70)
    
    results = evaluate_on_real_data(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        max_tasks=args.max_tasks,
        device=args.device
    )
    
    if results:
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / "real_arc_inference_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        return 0
    else:
        print("\nInference failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
ARC-AGI-2 Data Preparation
Descarga y prepara el dataset ARC-AGI-2 para entrenamiento.
"""

import sys
from pathlib import Path
import argparse
import json
import subprocess
import shutil
from typing import List, Dict
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI-2 dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for data")
    parser.add_argument("--download", action="store_true", help="Download from GitHub")
    parser.add_argument("--repo_url", type=str, default="https://github.com/arcprize/ARC-AGI-2.git")
    parser.add_argument("--split_ratios", type=str, default="0.8/0.1/0.1", help="Train/val/test ratios")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_dataset(repo_url: str, output_dir: Path) -> bool:
    """Descarga dataset desde GitHub."""
    print(f"Downloading ARC-AGI-2 from {repo_url}...")
    
    arc_dir = output_dir / "arc_agi_2_data"
    
    if arc_dir.exists():
        print(f"Directory {arc_dir} already exists. Remove it to re-download.")
        return True
    
    try:
        result = subprocess.run(
            ["git", "clone", repo_url, str(arc_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print("Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: git not found. Please install git.")
        return False


def load_arc_tasks(data_dir: Path) -> Dict[str, List[Dict]]:
    """Carga todas las tasks ARC."""
    tasks = {'train': [], 'eval': [], 'test': []}

    for split in ['train', 'eval', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue

        for json_file in split_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                task = json.load(f)
                task['task_id'] = json_file.stem
                # Fix: Handle test split without outputs (no labels available)
                if split == 'test' and 'test' in task:
                    for test_pair in task['test']:
                        # Test split may not have output labels
                        if 'output' not in test_pair:
                            test_pair['output'] = None  # Mark as unavailable
                tasks[split].append(task)

        print(f"Loaded {len(tasks[split])} tasks from {split}")

    return tasks


def create_custom_split(tasks: Dict, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Crea split personalizado de tasks."""
    random.seed(seed)
    
    # Combinar todas las tasks disponibles
    all_tasks = tasks.get('train', []) + tasks.get('eval', [])
    
    if not all_tasks:
        print("No tasks found to split")
        return {'train': [], 'val': [], 'test': []}
    
    # Shuffle
    random.shuffle(all_tasks)
    
    # Calcular índices
    n = len(all_tasks)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test = n - n_train - n_val
    
    split = {
        'train': all_tasks[:n_train],
        'val': all_tasks[n_train:n_train + n_val],
        'test': all_tasks[n_train + n_val:]
    }
    
    print(f"\nCustom split created:")
    print(f"  Train: {len(split['train'])} tasks ({len(split['train'])/n:.1%})")
    print(f"  Val: {len(split['val'])} tasks ({len(split['val'])/n:.1%})")
    print(f"  Test: {len(split['test'])} tasks ({len(split['test'])/n:.1%})")
    
    return split


def save_split(tasks: List[Dict], output_dir: Path, split_name: str):
    """Guarda split en directorio."""
    split_dir = output_dir / "splits" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for task in tasks:
        task_file = split_dir / f"{task['task_id']}.json"
        
        # Guardar solo datos necesarios (no metadata interna)
        task_data = {
            'train': task.get('train', []),
            'test': task.get('test', [])
        }
        
        with open(task_file, 'w') as f:
            json.dump(task_data, f)
    
    print(f"Saved {len(tasks)} tasks to {split_dir}")


def analyze_dataset(tasks: List[Dict]):
    """Analiza estadísticas del dataset."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Contar tasks
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")
    
    # Analizar tamaños de grid
    input_sizes = []
    output_sizes = []
    num_train_pairs = []
    
    for task in tasks:
        # Tamaños de train pairs
        for pair in task.get('train', []):
            inp = pair['input']
            out = pair['output']
            input_sizes.append((len(inp), len(inp[0]) if inp else 0))
            output_sizes.append((len(out), len(out[0]) if out else 0))
        
        # Número de train pairs
        num_train_pairs.append(len(task.get('train', [])))
    
    # Estadísticas
    if input_sizes:
        max_h = max(s[0] for s in input_sizes)
        max_w = max(s[1] for s in input_sizes)
        min_h = min(s[0] for s in input_sizes)
        min_w = min(s[1] for s in input_sizes)
        
        print(f"\nGrid size statistics:")
        print(f"  Max: {max_h}x{max_w}")
        print(f"  Min: {min_h}x{min_w}")
    
    if num_train_pairs:
        avg_pairs = sum(num_train_pairs) / len(num_train_pairs)
        max_pairs = max(num_train_pairs)
        min_pairs = min(num_train_pairs)
        
        print(f"\nTrain pairs per task:")
        print(f"  Average: {avg_pairs:.1f}")
        print(f"  Range: {min_pairs} - {max_pairs}")
    
    # Colores (0-9)
    print(f"\nColor values: 0-9 (10 colors)")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ARC-AGI-2 DATA PREPARATION")
    print("=" * 60)
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar si se solicita
    if args.download:
        success = download_dataset(args.repo_url, data_dir)
        if not success:
            print("Download failed. Exiting.")
            return
    
    # Verificar que existen los datos
    arc_data_dir = data_dir / "arc_agi_2_data"
    if not arc_data_dir.exists():
        print(f"Error: Data not found at {arc_data_dir}")
        print("Run with --download to download the dataset.")
        return
    
    # Cargar tasks
    print("\nLoading tasks...")
    tasks = load_arc_tasks(arc_data_dir)
    
    all_tasks = tasks.get('train', []) + tasks.get('eval', [])
    
    # Analizar
    analyze_dataset(all_tasks)
    
    # Parse split ratios
    ratios = args.split_ratios.split('/')
    train_ratio = float(ratios[0])
    val_ratio = float(ratios[1])
    test_ratio = float(ratios[2]) if len(ratios) > 2 else 1.0 - train_ratio - val_ratio
    
    print(f"\nCreating custom split: {train_ratio}/{val_ratio}/{test_ratio}")
    custom_split = create_custom_split(
        tasks, train_ratio, val_ratio, test_ratio, args.seed
    )
    
    # Guardar splits
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving splits...")
    save_split(custom_split['train'], processed_dir, 'train')
    save_split(custom_split['val'], processed_dir, 'val')
    save_split(custom_split['test'], processed_dir, 'test')
    
    # También guardar metadata
    metadata = {
        'source': str(arc_data_dir),
        'num_tasks_total': len(all_tasks),
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'seed': args.seed,
        'splits': {
            'train': len(custom_split['train']),
            'val': len(custom_split['val']),
            'test': len(custom_split['test'])
        }
    }
    
    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nProcessed data saved to: {processed_dir}")
    print("\nYou can now train with:")
    print(f"  python scripts/train.py --data_path {processed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

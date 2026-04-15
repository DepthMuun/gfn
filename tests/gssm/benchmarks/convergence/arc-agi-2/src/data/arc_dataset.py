"""
ARC-AGI-2 Data Loader
Carga y preprocesa el dataset ARC-AGI-2 para GSSM.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class ARCAGI2Dataset(Dataset):
    """
    Dataset ARC-AGI-2 para entrenamiento few-shot.
    
    Cada item contiene:
    - task_id: identificador único
    - train_pairs: lista de (input, output) ejemplos
    - test_input: grid de test
    - test_output: ground truth (solo en evaluación)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_train_pairs: int = 3,
        max_grid_size: int = 30,
        color_values: int = 10,  # 0-9 colores ARC
        shuffle_pairs: bool = True
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_train_pairs = max_train_pairs
        self.max_grid_size = max_grid_size
        self.color_values = color_values
        self.shuffle_pairs = shuffle_pairs
        
        self.tasks = self._load_tasks()
        
    def _load_tasks(self) -> List[Dict]:
        """Carga todas las tasks del split especificado."""
        tasks = []
        
        # ARC-AGI-2 tiene formatos: train/ o data/training/
        split_dir = self.data_path / self.split
        if not split_dir.exists():
            # Try alternative structure: data/training/
            alt_map = {'train': 'training', 'eval': 'evaluation', 'test': 'test'}
            alt_name = alt_map.get(self.split, self.split)
            split_dir = self.data_path / "data" / alt_name
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Cargar todos los archivos JSON
        for json_file in split_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                task_data = json.load(f)
                task_data['task_id'] = json_file.stem
                tasks.append(task_data)
        
        print(f"Loaded {len(tasks)} tasks from {split_dir}")
        return tasks
    
    def _pad_grid(self, grid: np.ndarray, target_size: int = 30) -> np.ndarray:
        """Pad grid to target_size x target_size."""
        h, w = grid.shape
        padded = np.zeros((target_size, target_size), dtype=np.int64)
        padded[:h, :w] = grid
        return padded
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convierte grid numpy a tensor (valores 0-9 sin normalizar)."""
        return torch.from_numpy(grid.astype(np.float32))
    
    def _tensor_to_grid(self, tensor: torch.Tensor) -> np.ndarray:
        """Convierte tensor a grid numpy (redondea a enteros)."""
        return tensor.round().clamp(0, 9).numpy().astype(np.int64)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict:
        task = self.tasks[idx]
        
        # Seleccionar subset de train pairs (few-shot)
        train_pairs = task.get('train', [])
        if len(train_pairs) > self.max_train_pairs:
            if self.shuffle_pairs:
                selected_pairs = random.sample(train_pairs, self.max_train_pairs)
            else:
                selected_pairs = train_pairs[:self.max_train_pairs]
        else:
            selected_pairs = train_pairs
        
        # Procesar train pairs
        processed_pairs = []
        for pair in selected_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Pad a tamaño máximo
            input_padded = self._pad_grid(input_grid, self.max_grid_size)
            output_padded = self._pad_grid(output_grid, self.max_grid_size)
            
            processed_pairs.append({
                'input': self._grid_to_tensor(input_padded),
                'output': self._grid_to_tensor(output_padded),
                'input_size': input_grid.shape,
                'output_size': output_grid.shape
            })
        
        # Procesar test input
        test_input = np.array(task['test'][0]['input'])
        test_input_padded = self._pad_grid(test_input, self.max_grid_size)
        
        result = {
            'task_id': task['task_id'],
            'train_pairs': processed_pairs,
            'test_input': self._grid_to_tensor(test_input_padded),
            'test_input_size': test_input.shape,
            'num_train_pairs': len(selected_pairs)
        }
        
        # Incluir test output si está disponible (train/eval splits)
        if 'test' in task and len(task['test']) > 0 and 'output' in task['test'][0]:
            test_output = np.array(task['test'][0]['output'])
            test_output_padded = self._pad_grid(test_output, self.max_grid_size)
            result['test_output'] = self._grid_to_tensor(test_output_padded)
            result['test_output_size'] = test_output.shape
        
        return result


def collate_arc_batch(batch: List[Dict]) -> Dict:
    """
    Collate function para DataLoader.
    Como cada task tiene tamaños diferentes, usamos batch_size=1.
    """
    assert len(batch) == 1, "ARC-AGI-2 usa batch_size=1 por task"
    return batch[0]


def create_arc_dataloader(
    data_path: str,
    split: str = "train",
    batch_size: int = 1,  # Siempre 1 para ARC
    max_train_pairs: int = 3,
    **kwargs
) -> DataLoader:
    """Crea DataLoader para ARC-AGI-2."""
    dataset = ARCAGI2Dataset(
        data_path=data_path,
        split=split,
        max_train_pairs=max_train_pairs,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_arc_batch,
        num_workers=0  # Single-threaded para reproducibilidad
    )


if __name__ == "__main__":
    # Test del data loader
    print("Testing ARC-AGI-2 DataLoader...")
    
    # Crear dataset dummy para test
    dummy_data = {
        "train": [
            {
                "input": [[0, 0], [0, 1]],
                "output": [[1, 1], [1, 0]]
            },
            {
                "input": [[1, 1], [1, 0]],
                "output": [[0, 0], [0, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]]
            }
        ]
    }
    
    import tempfile
    import os
    
    # Guardar dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        train_dir.mkdir()
        
        with open(train_dir / "test_task.json", 'w') as f:
            json.dump(dummy_data, f)
        
        # Probar dataset
        dataset = ARCAGI2Dataset(tmpdir, split="train", max_train_pairs=2)
        print(f"Dataset length: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Task ID: {sample['task_id']}")
        print(f"Num train pairs: {sample['num_train_pairs']}")
        print(f"Test input shape: {sample['test_input'].shape}")
        print(f"Test output shape: {sample['test_output'].shape}")
        
        print("\nDataLoader test passed!")

"""
data/loader.py — GFN V5
DataLoaders y datasets para tareas GFN.
WATCHOUT: El DataLoader se crea UNA vez fuera del loop de entrenamiento.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple, Optional
from ..data.dataset import SequenceDataset


def create_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Crea train y validation DataLoaders desde tensores.
    IMPORTANTE: Crear los DataLoaders UNA VEZ fuera del loop — no dentro.

    Args:
        x, y:        Tensores de entrada y objetivo
        batch_size:  Tamaño de batch
        val_split:   Fracción de datos para validación (0 = sin validación)
        shuffle:     Mezclar datos de entrenamiento
        num_workers: Workers para carga de datos
        seed:        Semilla para reproducibilidad del split

    Returns:
        (train_loader, val_loader) — val_loader es None si val_split=0
    """
    dataset = SequenceDataset(x, y)

    if val_split > 0:
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    return train_loader, None

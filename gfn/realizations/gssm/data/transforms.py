"""
data/transforms.py — GFN V5
Transformaciones de datos para secuencias.
"""

import torch
from typing import Tuple, Optional


def shift_targets(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crea pares (input, target) desplazados for language modeling.
    input  = x[:, :-1]
    target = x[:, 1:]
    """
    return x[:, :-1], x[:, 1:]


def add_bos_token(x: torch.Tensor, bos_id: int = 0) -> torch.Tensor:
    """Añade token BOS al inicio de cada secuencia."""
    bos = torch.full((x.size(0), 1), bos_id, dtype=x.dtype, device=x.device)
    return torch.cat([bos, x], dim=1)


def pad_sequences(sequences, max_len: int, pad_id: int = 0) -> torch.Tensor:
    """Padea una lista de secuencias de longitud variable."""
    result = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        result[i, :length] = torch.tensor(seq[:length])
    return result


def create_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Crea attention mask desde longitudes de secuencia.
    Returns: [B, max_len] con True donde hay datos válidos.
    """
    indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return indices < lengths.unsqueeze(1)

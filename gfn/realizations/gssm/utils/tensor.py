"""
Utilidades de tensores — GFN V5
Funciones auxiliares para manipulación de tensores PyTorch.
"""

import torch
from typing import Optional, Tuple, Union


def flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, D] → [B, H*D]"""
    if x.dim() == 3:
        return x.flatten(1)
    return x


def unflatten_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[B, D] → [B, H, D//H]"""
    if x.dim() == 2:
        B, D = x.shape
        return x.view(B, heads, D // heads)
    return x


def merge_batch_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, D] → [B*H, D]"""
    if x.dim() == 3:
        B, H, D = x.shape
        return x.reshape(B * H, D)
    return x


def split_batch_heads(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """[B*H, D] → [B, H, D]"""
    if x.dim() == 2:
        BH, D = x.shape
        H = BH // batch_size
        return x.reshape(batch_size, H, D)
    return x


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Máscara causal de atención: True donde se puede atender.
    WATCHOUT: verificar causal masking antes de publicar resultados.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()


def shift_right(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """
    Desplaza la secuencia un paso a la derecha (para language modeling).
    Input:  [B, S]
    Output: [B, S] con padding en posición 0
    """
    shifted = torch.zeros_like(x)
    shifted[:, 1:] = x[:, :-1]
    shifted[:, 0] = pad_value
    return shifted


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Media enmascarada a lo largo de una dimensión."""
    mask = mask.float()
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def nan_to_num(x: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    """Reemplaza NaNs e Infs con un valor numérico."""
    return torch.nan_to_num(x, nan=replacement, posinf=replacement, neginf=replacement)


def count_parameters(model: torch.nn.Module) -> int:
    """Cuenta parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

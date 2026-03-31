"""
gfn/math/stability.py — GFN V5
Numerical stability and entropy functions.
"""
import torch
import torch.nn.functional as F

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))

def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)

def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over logits distribution [*, V]."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * safe_log(probs)).sum(dim=-1)

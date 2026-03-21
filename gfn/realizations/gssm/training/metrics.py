"""
Métricas de evaluación — GFN V5
Calcula métricas de evaluación para modelos de secuencia.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math


def accuracy(logits: torch.Tensor, targets: torch.Tensor,
             ignore_index: int = -100) -> float:
    """
    Token-level accuracy.
    logits: [B, S, V]
    targets: [B, S]
    """
    preds = logits.argmax(dim=-1)  # [B, S]
    mask = (targets != ignore_index)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def perplexity(logits: torch.Tensor, targets: torch.Tensor,
               ignore_index: int = -100) -> float:
    """
    Perplexidad de lenguaje = exp(cross_entropy).
    """
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index
    )
    return math.exp(min(ce.item(), 100))  # clamp para evitar overflow


def last_token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Accuracy solo sobre el último token.
    Usada en XOR y tareas de razonamiento donde importa el token final.
    WATCHOUT: evaluar solo último token en XOR.
    """
    pred_last = logits[:, -1, :].argmax(dim=-1)  # [B]
    target_last = targets[:, -1]                  # [B]
    return (pred_last == target_last).float().mean().item()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor,
                    task: str = 'lm') -> Dict[str, float]:
    """
    Computa todas las métricas relevantes para una tarea.

    task:
    - 'lm':     Token-level accuracy + perplexity
    - 'xor':    Last-token accuracy
    - 'niah':   Last-token accuracy + perplexity
    """
    metrics: Dict[str, float] = {}

    if task == 'xor':
        metrics['acc'] = last_token_accuracy(logits, targets)
    elif task == 'lm':
        metrics['acc'] = accuracy(logits, targets)
        metrics['ppl'] = perplexity(logits, targets)
    elif task == 'niah':
        metrics['acc'] = last_token_accuracy(logits, targets)
        metrics['ppl'] = perplexity(logits, targets)
    else:
        metrics['acc'] = accuracy(logits, targets)

    return metrics

"""
Schedulers — GFN V5
Learning rate schedulers para entrenamiento de GFN.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import math


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup lineal + decaída coseno.
    Estándar para entrenar transformers y modelos de secuencia.
    """

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            alpha = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return [max(self.min_lr, base_lr * alpha) for base_lr in self.base_lrs]


class StepScheduler(torch.optim.lr_scheduler.StepLR):
    """Wrapper semántico sobre StepLR estándar."""
    pass


class ReduceOnPlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce LR cuando la pérdida de validación se estanca."""
    pass


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any],
                     steps_per_epoch: int = 1) -> Optional[object]:
    """
    Factory para crear schedulers desde configuración.

    config keys:
    - 'type':         'cosine_warmup' | 'step' | 'plateau' | 'none'
    - 'warmup_steps': pasos de calentamiento (si aplica)
    - 'total_steps':  pasos totales (si aplica)
    - 'min_lr':       lr mínimo
    """
    sched_type = config.get('type', 'none').lower()
    if sched_type == 'none':
        return None

    epochs = config.get('epochs', 10)
    total_steps = config.get('total_steps', epochs * steps_per_epoch)
    warmup_steps = config.get('warmup_steps', max(1, total_steps // 10))
    min_lr = config.get('min_lr', 1e-6)

    if sched_type == 'cosine_warmup':
        return WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)
    elif sched_type == 'step':
        step_size = config.get('step_size', epochs // 3)
        gamma = config.get('gamma', 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'plateau':
        patience = config.get('patience', 5)
        factor = config.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)

    return None

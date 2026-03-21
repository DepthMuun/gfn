"""
CheckpointCallback — GFN V5
Guarda el modelo periódicamente o cuando mejora la métrica de referencia.
"""

import torch
import os
from typing import Optional, Dict, Any
from ...training.callbacks import Callback


class CheckpointCallback(Callback):
    """
    Guarda checkpoints del modelo.

    Config:
    - save_dir:      Directorio donde guardar los checkpoints.
    - monitor:       Métrica a monitorear (default: 'loss').
    - mode:          'min' o 'max'.
    - save_every:    Guardar cada N épocas (default: None = solo best).
    """

    def __init__(self, save_dir: str = './checkpoints', monitor: str = 'loss',
                 mode: str = 'min', save_every: Optional[int] = None):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.best_score = float('inf') if mode == 'min' else float('-inf')

        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer=None):
        score = metrics.get(self.monitor, None)

        # Periodic save
        if self.save_every and (epoch + 1) % self.save_every == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            self._save(trainer, path, epoch, metrics)

        # Best model save
        if score is not None:
            is_better = (score < self.best_score) if self.mode == 'min' else (score > self.best_score)
            if is_better:
                self.best_score = score
                path = os.path.join(self.save_dir, 'best_model.pt')
                self._save(trainer, path, epoch, metrics)
                print(f"[Checkpoint] Best {self.monitor}={score:.4f} → saved to {path}")

    def _save(self, trainer, path: str, epoch: int, metrics: Dict[str, float]):
        if trainer is None:
            return
        torch.save({
            'epoch': epoch,
            'model_state': trainer.model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'metrics': metrics
        }, path)

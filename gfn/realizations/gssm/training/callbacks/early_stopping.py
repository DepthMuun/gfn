"""
EarlyStoppingCallback — GFN V5
Detiene el entrenamiento si la métrica monitoreada no mejora.
"""

from typing import Dict
from ...training.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """
    Para el entrenamiento cuando la métrica no mejora por `patience` épocas.

    Config:
    - monitor:   Métrica a monitorear (default: 'loss').
    - patience:  Épocas sin mejora antes de parar (default: 5).
    - min_delta: Mejora mínima para considerar como mejora real.
    - mode:      'min' o 'max'.
    """

    def __init__(self, monitor: str = 'loss', patience: int = 5,
                 min_delta: float = 1e-4, mode: str = 'min'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer=None):
        score = metrics.get(self.monitor, None)
        if score is None:
            return

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EarlyStopping] Stopping at epoch {epoch+1}.")
                if hasattr(trainer, '_stop_training'):
                    trainer._stop_training = True

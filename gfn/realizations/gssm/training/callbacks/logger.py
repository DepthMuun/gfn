"""
LoggerCallback — GFN V5
Logging de métricas de entrenamiento a consola y/o archivo.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
from ...training.callbacks import Callback


class LoggerCallback(Callback):
    """
    Registra métricas de entrenamiento en consola y opcionalmente en archivo JSON.

    Config:
    - log_every:  Cada cuántos pasos logear (default: 100).
    - log_file:   Path al archivo de log (default: None = solo consola).
    - run_name:   Nombre del experimento.
    """

    def __init__(self, log_every: int = 100, log_file: Optional[str] = None,
                 run_name: str = 'gfn_run'):
        self.log_every = log_every
        self.log_file = log_file
        self.run_name = run_name
        self._step_log: list[Dict[str, Any]] = []
        self._epoch_log: list[Dict[str, Any]] = []
        self._step = 0

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    def on_train_start(self, trainer=None):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{self.run_name}] Training started at {ts}")

    def on_batch_end(self, step: int, loss: float, trainer=None):
        self._step += 1
        if self._step % self.log_every == 0:
            print(f"  step={self._step:6d}  loss={loss:.4f}")
            self._step_log.append({'step': self._step, 'loss': loss})

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer=None):
        metrics_str = '  '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        print(f"[Epoch {epoch+1}] {metrics_str}")
        self._epoch_log.append({'epoch': epoch + 1, **metrics})

        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump({'epochs': self._epoch_log, 'steps': self._step_log}, f, indent=2)

    def on_train_end(self, trainer=None):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{self.run_name}] Training ended at {ts}")

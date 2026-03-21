"""
GFNTrainer — GFN V5
Trainer completo con soporte de callbacks, scheduler, y métricas.
Reemplaza/enriquece el trainer básico existente.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm

from ..interfaces.model import GFNModel
from ..interfaces.loss import Loss
from .callbacks import Callback
from .metrics import compute_metrics


class GFNTrainer:
    """
    Trainer estándar para GFN V5.

    Soporta:
    - Callbacks: checkpoint, early stopping, logger
    - Scheduler: paso automático por época
    - Métricas: accuracy, perplexity, last-token para tareas XOR/NIAH
    - Gradient clipping configurable
    - State info passthrough para pérdidas físicas
    """

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler=None,
                 callbacks: Optional[List[Callback]] = None,
                 device: Optional[str] = None,
                 grad_clip: float = 1.0,
                 task: str = 'lm'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.grad_clip = grad_clip
        self.task = task
        self._stop_training = False

    # ─── Hooks de ciclo de vida ──────────────────────────────────────────────

    def _call(self, event: str, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(trainer=self, **kwargs)

    # ─── Paso de entrenamiento ───────────────────────────────────────────────

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        logits, states, info = self.model(x)
        
        # Asegurar que info contenga toda la información del estado
        # para pérdidas físicas y toroidales
        state_info = info if info is not None else {}
        
        # Si la pérdida necesita state_info, aseguramos que tenga x_seq y v_seq
        if 'x_seq' not in state_info and hasattr(states, '__iter__'):
            # states es (x_final, v_final) - intentar obtener secuencias
            pass
        
        loss = self.loss_fn(logits, y, state_info=state_info)

        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss.item()

    # ─── Loop de entrenamiento ───────────────────────────────────────────────

    def fit(self, train_loader, epochs: int = 1,
            val_loader=None) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'loss': []}
        self._call('on_train_start')

        for epoch in range(epochs):
            if self._stop_training:
                break

            self._call('on_epoch_start', epoch=epoch)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            n_batches = 0

            for step, (x, y) in enumerate(pbar):
                self._call('on_batch_start', step=step)
                x, y = x.to(self.device), y.to(self.device)
                loss_val = self.train_step(x, y)
                history['loss'].append(loss_val)
                epoch_loss += loss_val
                n_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                self._call('on_batch_end', step=step, loss=loss_val)

            epoch_metrics: Dict[str, float] = {'loss': epoch_loss / max(1, n_batches)}

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate_metrics(val_loader)
                epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            self._call('on_epoch_end', epoch=epoch, metrics=epoch_metrics)

            # Scheduler step (per epoch)
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    monitor_val = epoch_metrics.get('val_loss', epoch_metrics.get('loss'))
                    try:
                        self.scheduler.step(monitor_val)
                    except TypeError:
                        self.scheduler.step()

        self._call('on_train_end')
        return history

    # ─── Evaluación ─────────────────────────────────────────────────────────

    def evaluate(self, val_loader) -> float:
        """Retorna accuracy (token-level o last-token según task)."""
        metrics = self.evaluate_metrics(val_loader)
        return metrics.get('acc', 0.0)

    def evaluate_metrics(self, val_loader) -> Dict[str, float]:
        """Retorna dict completo de métricas."""
        self.model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _ = self.model(x)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())

        if not all_logits:
             return {}

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return compute_metrics(logits, targets, task=self.task)

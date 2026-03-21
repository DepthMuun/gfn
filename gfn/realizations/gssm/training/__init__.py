"""
gfn/training/__init__.py
Public API del módulo training — GFN V5
"""

from ..training.trainer import GFNTrainer
from ..training.optimizer import (
    RiemannianAdam, RiemannianSGD, create_optimizer,
    make_gfn_optimizer, all_parameters,
)
from ..training.scheduler import WarmupCosineScheduler, create_scheduler
from ..training.metrics import accuracy, perplexity, last_token_accuracy, compute_metrics
from ..training.callbacks import Callback
from ..training.callbacks.checkpoint import CheckpointCallback
from ..training.callbacks.early_stopping import EarlyStoppingCallback
from ..training.callbacks.logger import LoggerCallback
from ..training.checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "GFNTrainer",
    # Optimizers
    "RiemannianAdam", "RiemannianSGD", "create_optimizer",
    "make_gfn_optimizer", "all_parameters",
    # Schedulers
    "WarmupCosineScheduler", "create_scheduler",
    # Metrics
    "accuracy", "perplexity", "last_token_accuracy", "compute_metrics",
    # Callbacks
    "Callback", "CheckpointCallback", "EarlyStoppingCallback", "LoggerCallback",
    # Checkpoint
    "save_checkpoint", "load_checkpoint",
]

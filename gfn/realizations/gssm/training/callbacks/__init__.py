"""
Training callback base class and standard callback implementations.
"""

import torch
import os
from typing import Optional, Dict, Any


class Callback:
    """
    Base interface for training callbacks.
    """
    def on_epoch_start(self, epoch: int, trainer=None): pass
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer=None): pass
    def on_batch_start(self, step: int, trainer=None): pass
    def on_batch_end(self, step: int, loss: float, trainer=None): pass
    def on_train_start(self, trainer=None): pass
    def on_train_end(self, trainer=None): pass

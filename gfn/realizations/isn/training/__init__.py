"""Training components for ISN Model."""

# Import sub-packages to trigger registration
from . import strategies
from .trainer import Trainer

__all__ = ["strategies", "Trainer"]

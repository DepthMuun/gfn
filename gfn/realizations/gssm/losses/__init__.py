"""
gfn/losses/__init__.py
Public API del módulo losses — GFN V5
"""

from .base import BaseLoss
from .factory import LossFactory
from .generative import ManifoldGenerativeLoss
from .physics import PhysicsLoss, PhysicsInformedLoss
from .toroidal import ToroidalLoss, ToroidalVelocityLoss, ToroidalDistanceLoss
from .regularization import NoetherSymmetryLoss, DynamicLossBalancer
from .detection import GIoULoss, IoULoss, giou_loss, iou_loss

__all__ = [
    "BaseLoss",
    "LossFactory",
    "ManifoldGenerativeLoss",
    "PhysicsLoss",
    "PhysicsInformedLoss",
    "ToroidalLoss",
    "ToroidalVelocityLoss",
    "ToroidalDistanceLoss",
    "NoetherSymmetryLoss",
    "DynamicLossBalancer",
    # Detection
    "GIoULoss", "IoULoss", "giou_loss", "iou_loss",
]

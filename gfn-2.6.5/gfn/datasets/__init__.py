"""
Datasets
========

Dataset implementations for GFN training.

Available Datasets:
    - MathDataset: Mathematical operations dataset
    - MixedHFDataset: Mixed tasks dataset
"""

from .math import MathDataset
from .mixed import MixedHFDataset

__all__ = [
    'MathDataset',
    'MixedHFDataset',
]

"""
gfn/utils/__init__.py
Public API del módulo utils — GFN V5
"""

from ..utils.tensor import (
    flatten_heads,
    unflatten_heads,
    merge_batch_heads,
    split_batch_heads,
    causal_mask,
    shift_right,
    masked_mean,
    nan_to_num,
    count_parameters,
)
from ..utils.coords import (
    box_to_torus,
    torus_to_box,
    wrap_angles,
    angle_to_unit,
)

__all__ = [
    # Tensor
    "flatten_heads", "unflatten_heads", "merge_batch_heads", "split_batch_heads",
    "causal_mask", "shift_right", "masked_mean", "nan_to_num", "count_parameters",
    # Coords
    "box_to_torus", "torus_to_box", "wrap_angles", "angle_to_unit",
]

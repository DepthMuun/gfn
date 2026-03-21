"""
core/types.py — GFN V5
Tipos y type aliases del framework.
"""

import torch
from typing import Dict, Any, Tuple, Optional, List, Union

# ─── State types ─────────────────────────────────────────────────────────────
# (position, velocity) pair
ManifoldState = Tuple[torch.Tensor, torch.Tensor]

# Trajectory: list of (x, v) states over time
Trajectory = List[ManifoldState]

# Force tensor (same shape as x, v)
Force = torch.Tensor

# Integration step result
StepResult = Dict[str, torch.Tensor]  # {'x': ..., 'v': ...}

# ─── Config types ─────────────────────────────────────────────────────────────
ConfigDict = Dict[str, Any]

# ─── Forward pass outputs ─────────────────────────────────────────────────────
# (logits, state, info_dict)
ModelOutput = Tuple[torch.Tensor, ManifoldState, Dict[str, Any]]

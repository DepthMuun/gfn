import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
from .base import BaseModel
from .manifold_layer import ManifoldLayer
from ..registry import register_model

@register_model('manifold')
class ManifoldModel(BaseModel):
    """
    Concrete implementation of the GFN Manifold Model.
    Orchestrates layers, embeddings, and readout plugins.
    """
    def __init__(self, 
                 layers: nn.ModuleList, 
                 embedding: nn.Module, 
                 x0: nn.Parameter, 
                 v0: nn.Parameter, 
                 holographic: bool = False,
                 config: Optional[Any] = None):
        super().__init__(layers, embedding, x0, v0, holographic, config=config)
        
    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                force_manual: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Evolution loop through sequence and layers.
        """
        return super().forward(input_ids, attention_mask, state, force_manual, **kwargs)

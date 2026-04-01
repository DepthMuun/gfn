"""
ManifoldGenerativeLoss — GFN V5
Generative loss for the Manifold model.

ARCHITECTURE: The model can use different output strategies:
1. Holographic: The final state is used directly as logits
2. Readout: Linear projection of the state to vocabulary
3. Toroidal: Angular coordinates for toroidal space

Options:
- 'nll':        CrossEntropy over logits (default for categorical readout)
- 'mse':        L2 over output space (for continuous representations)
- 'cosine':     Cosine distance (normalized embeddings)
- 'toroidal':   Geodesic angular distance (for toroidal manifold)
- 'hybrid':     Combines NLL with toroidal regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from ..losses.base import BaseLoss
from ..registry import register_loss
from ..constants import EPS


@register_loss('generative')
class ManifoldGenerativeLoss(BaseLoss):
    """
    Generative loss for GFN V5.
    
    Handles multiple manifold output modes:
    - 'nll':      CrossEntropy over projected logits
    - 'mse':      MSE over continuous vectors
    - 'cosine':   Cosine distance
    - 'toroidal': Geodesic angular distance
    - 'hybrid':   Combines NLL + toroidal
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mode = self.config.get('mode', 'nll')
        self.entropy_coef = self.config.get('entropy_coef', 0.0)
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        
        # Parameters for toroidal mode
        self.toroidal_scale = self.config.get('toroidal_scale', 1.0)
        self.toroidal_weight = self.config.get('toroidal_weight', 0.3)
        
        # Parameters for hybrid mode
        self.hybrid_nll_weight = self.config.get('hybrid_nll_weight', 0.7)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                state_info: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x_pred:   Logits or output vectors from readout [B, S, V] or [B, S, D]
            x_target: Target token IDs [B, S]
            state_info: State information for additional losses
        Returns:
            Combined loss value
        """
        if self.mode == 'mse':
            return self._mse_loss(x_pred, x_target)
        
        elif self.mode == 'cosine':
            return self._cosine_loss(x_pred, x_target)
        
        elif self.mode == 'toroidal':
            return self._toroidal_loss(x_pred, x_target)
        
        elif self.mode == 'hybrid':
            return self._hybrid_loss(x_pred, x_target, state_info)
        
        else:
            # mode == 'nll' (default)
            return self._nll(x_pred, x_target)

    def _nll(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        NLL loss with label smoothing.
        
        Args:
            logits: Model logits [B, S, V]
            targets: Token IDs [B, S]
        Returns:
            Cross-entropy loss
        """
        if logits.dim() == 2:
            # Case where it's already flattened or single step
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
            
        if logits.dim() == 4:
            # Multi-head holographic case [B, S, H, HD] -> flatten heads for loss
            B, S, H, HD = logits.shape
            logits = logits.reshape(B, S, H * HD)

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            label_smoothing=self.label_smoothing
        )

        if self.entropy_coef > 0:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + EPS)).sum(dim=-1).mean()
            loss = loss - self.entropy_coef * entropy

        return loss

    def _mse_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """L2 loss over continuous vectors."""
        # Ensure targets are valid indices to allow gradient flow
        y = x_target.float()
        
        # If pred is [B, S, V] and target is [B, S], average logits if no readout
        if x_pred.dim() == 3 and y.dim() == 2:
             # Channel regression case: average or adjust
             # For now, if user requests MSE over logits, average the vocab
             y = y.unsqueeze(-1)
        
        return F.mse_loss(x_pred, y)

    def _cosine_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Cosine distance."""
        if x_target.dtype in (torch.long, torch.int):
            return self._nll(x_pred, x_target)
        
        x_pred_n = F.normalize(x_pred, dim=-1)
        x_tgt_n = F.normalize(x_target.float(), dim=-1)
        return (1 - (x_pred_n * x_tgt_n).sum(dim=-1)).mean()

    def _toroidal_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Toroidal loss: geodesic angular distance.
        Requires x_pred to be angular coordinates.
        """
        # If x_pred are logits, convert to coordinates
        if x_pred.dim() == 3 and x_pred.shape[-1] > 1:
            # Logits -> angular coordinates
            probs = F.softmax(x_pred, dim=-1)
            # Assume vocabulary evenly spaced in [0, 2π]
            num_classes = x_pred.shape[-1]
            angles = torch.linspace(0, 2 * torch.pi, num_classes, 
                                   device=x_pred.device).unsqueeze(0)
            x_pred_coords = torch.sum(probs * angles.unsqueeze(0), dim=-1)
        else:
            x_pred_coords = x_pred.squeeze(-1) if x_pred.dim() > 2 else x_pred
        
        # Convert targets to angular coordinates
        if x_target.dtype in (torch.long, torch.int):
            num_classes = x_pred.shape[-1] if x_pred.dim() == 3 else 100
            x_target_coords = 2 * torch.pi * x_target.float() / num_classes
        else:
            x_target_coords = x_target
        
        # Compute angular distance
        diff = x_pred_coords - x_target_coords
        diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return self.toroidal_scale * (diff_wrapped ** 2).mean()

    def _hybrid_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                    state_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Hybrid loss: combines NLL with toroidal regularization.
        Useful when the model produces logits but the latent space is toroidal.
        """
        # Component NLL
        nll = self._nll(x_pred, x_target)
        
        # Clamp to avoid out-of-bounds (if we have state information)
        toroidal_loss = torch.tensor(0.0, device=x_pred.device)
        
        if state_info is not None and 'x_seq' in state_info:
            x_seq = state_info['x_seq']
            
            # Get coordinates of the last state
            if x_seq.dim() == 4:
                # [B, S, H, D] -> get final state
                x_final = x_seq[:, -1, :, :]  # [B, H, D]
            else:
                x_final = x_seq
            
            # Compute negative log-likelihood to target
            if x_target.dtype in (torch.long, torch.int):
                num_classes = x_pred.shape[-1]
                tgt_coords = 2 * torch.pi * x_target.float() / num_classes
            else:
                tgt_coords = x_target.float()
            
            # Geodesic distance
            # Distancia geodésica
            diff = x_final - tgt_coords.unsqueeze(-1) if tgt_coords.dim() > 1 else x_final - tgt_coords
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            toroidal_loss = (diff_wrapped ** 2).mean()
        
        # Combine
        return self.hybrid_nll_weight * nll + self.toroidal_weight * toroidal_loss

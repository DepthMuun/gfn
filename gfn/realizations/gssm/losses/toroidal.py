"""
ToroidalLoss — GFN V5
Loss specific for toroidal geometries.
Toroidal loss works with angular distances in the manifold.
Improved version for compatibility with training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from ..losses.base import BaseLoss
from ..registry import register_loss
from ..constants import EPS
from ..cuda.ops import CUDA_AVAILABLE, toroidal_loss_fwd, toroidal_loss_bwd


class ToroidalDistanceLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        y_pred_c = y_pred.contiguous()
        y_true_c = y_true.contiguous()
        ctx.save_for_backward(y_pred_c, y_true_c)
        out = toroidal_loss_fwd(y_pred_c, y_true_c)
        return out.view(y_pred.shape)

    @staticmethod
    def backward(ctx, grad_output):
        y_pred_c, y_true_c = ctx.saved_tensors
        if grad_output is None:
            return None, None
        grad_output_c = grad_output.contiguous()
        grad_pred = toroidal_loss_bwd(grad_output_c, y_pred_c, y_true_c)
        return grad_pred.view(y_pred_c.shape), None


@register_loss('toroidal')
@register_loss('toroidal_distance')
class ToroidalLoss(BaseLoss):
    """
    Angular MSE loss for toroidal geometries.
    Uses CUDA-accelerated implementation when available.
    Available modes:
    - 'circular': Circular distance with wrapping (default)
    - 'mse': Standard MSE on angular coordinates
    - 'hybrid': Combines circular distance with vector consistency
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reduction = self.config.get('reduction', 'mean')
        self.scale = self.config.get('scale', 1.0)
        self.power = self.config.get('power', 2.0)
        self.mode = self.config.get('mode', 'circular')
        # Parameters for Riemannian metric
        self.R = self.config.get('R', 2.0)
        self.r = self.config.get('r', 1.0)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x_pred: Predicted angular position [B, S, D], [B, S, H, D], or [B, D]
            x_target: Target angular position, same shape
        Returns:
            Scalar loss — mean geodesic angular deviation.
        """
        # Ensure target is float for calculations
        if x_target.dtype in (torch.long, torch.int):
            # If target is token IDs, convert to angular coordinates
            # Use a simple transformation: token_id -> angle
            x_target = x_target.float()
        
        # Compute difference
        diff = x_pred - x_target
        
        if self.mode == 'circular':
            # Fast CUDA path if available, power=2.0 and inputs are on CUDA
            if CUDA_AVAILABLE and x_pred.is_cuda and toroidal_loss_fwd is not None and self.power == 2.0:
                dist = ToroidalDistanceLossFunction.apply(x_pred, x_target)
            else:
                # Wrap to [-π, π] using atan2(sin, cos)
                diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
                dist = diff_wrapped.pow(self.power)
        elif self.mode == 'mse':
            # Simple MSE without wrapping
            dist = diff.pow(self.power)
        elif self.mode == 'riemannian':
            # Métrica del toro: ds² = r² dθ² + (R + r cos θ)² dφ²
            # diff = x_pred - x_target
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            
            # We assume dimensions come in pairs (θ, φ)
            # x_pred shape: [..., D]
            D = diff_wrapped.shape[-1]
            if D % 2 != 0:
                # Fallback to circular if not even
                dist = diff_wrapped.pow(self.power)
            else:
                # θ: even indices (0, 2, ...), φ: odd indices (1, 3, ...)
                d_theta = diff_wrapped[..., 0::2]
                d_phi = diff_wrapped[..., 1::2]
                theta_pred = x_pred[..., 0::2]
                
                # ds² = r² Δθ² + (R + r cos θ)² Δφ²
                g_phi = (self.R + self.r * torch.cos(theta_pred))**2
                dist_sq = (self.r**2) * d_theta.pow(2) + g_phi * d_phi.pow(2)
                
                # Re-assemble or apply power
                if self.power == 2.0:
                    dist = dist_sq
                else:
                    dist = dist_sq.pow(self.power / 2.0)
        elif self.mode == 'hybrid':
            # Combine circular distance with vector penalty
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            
            # Circular component
            circ_dist = diff_wrapped.pow(self.power)
            
            # Vector component (direction consistency)
            pred_sin = torch.sin(x_pred)
            pred_cos = torch.cos(x_pred)
            tgt_sin = torch.sin(x_target)
            tgt_cos = torch.cos(x_target)
            
            # Cosine distance between unit vectors
            dot_product = pred_sin * tgt_sin + pred_cos * tgt_cos
            vec_dist = 1.0 - dot_product
            
            # Combine components
            dist = 0.7 * circ_dist + 0.3 * vec_dist
        elif self.mode == 'phase':
            # FIX: Add 'phase' mode per 01_HYPER_TORUS.md Section 3.2
            # L_phase = 1 - cos(x_pred - x_target)
            # Provides smooth gradients for periodic objectives
            diff_cos = torch.cos(diff)
            dist = 1.0 - diff_cos  # [0, 2] range, 0 = perfect match
        else:
            # Default: circular
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            dist = diff_wrapped.pow(self.power)

        if self.reduction == 'mean':
            return self.scale * dist.mean()
        elif self.reduction == 'sum':
            return self.scale * dist.sum()
        return self.scale * dist


@register_loss('toroidal_categorical')
class ToroidalCategoricalLoss(BaseLoss):
    """
    Loss for when the model produces categorical logits but works on a toroidal manifold.
    
    This loss:
    1. Converts logits to angular coordinates
    2. Converts token targets to angular coordinates
    3. Calculates geodesic distance between both
    
    Useful when the readout produces logits but the latent space is toroidal.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.scale = self.config.get('scale', 1.0)
        # Map tokens to angles - learnable or fixed
        self.learnable_tokens = self.config.get('learnable_tokens', True)
        self.vocab_size = self.config.get('vocab_size', 100)
        
        if self.learnable_tokens:
            # Tokens as angular embeddings evenly spaced
            # Use register_parameter to make it learnable
            angles = torch.linspace(0, 2 * torch.pi, self.vocab_size + 1)[:-1]  # Avoid duplicate at 2π
            self.register_parameter('token_angles', nn.Parameter(angles.unsqueeze(0)))

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x_pred: Model logits [B, S, V] or coordinates [B, S, D]
            x_target: Target token IDs [B, S]
        Returns:
            Scalar loss
        """
        B, S = x_target.shape
        
        # Get target angles
        if hasattr(self, 'token_angles') and self.learnable_tokens:
            # Use learnable angles
            tgt_angles = self.token_angles.to(x_target.device)
            # Map tokens to angles: [B, S] -> [B, S]
            x_target_rad = tgt_angles[0:1, :self.vocab_size].expand(B, -1)
            # Select angle for each token
            x_target_rad = torch.gather(
                x_target_rad.expand(B, -1), 
                1, 
                x_target.unsqueeze(-1)
            ).squeeze(-1)
        else:
            # Use fixed mapping: token_id -> 2*pi * token_id / vocab_size
            x_target_rad = 2 * torch.pi * x_target.float() / self.vocab_size
        
        # Get prediction angles
        if x_pred.dim() == 3 and x_pred.shape[-1] > 1:
            # x_pred are logits -> convert to angular coordinates
            # Use weighted average of angles using softmax
            probs = F.softmax(x_pred, dim=-1)  # [B, S, V]
            
            if hasattr(self, 'token_angles') and self.learnable_tokens:
                token_angles = self.token_angles.to(x_pred.device)
            else:
                token_angles = torch.linspace(0, 2 * torch.pi, x_pred.shape[-1], 
                                              device=x_pred.device).unsqueeze(0)
            
            # Weighted average of angles
            x_pred_rad = torch.sum(probs * token_angles.unsqueeze(0), dim=-1)  # [B, S]
        else:
            # x_pred are already coordinates
            x_pred_rad = x_pred.squeeze(-1) if x_pred.dim() > 2 else x_pred
        
        # Ensure same shape
        if x_pred_rad.dim() == 1:
            x_pred_rad = x_pred_rad.unsqueeze(0)
        if x_target_rad.dim() == 1:
            x_target_rad = x_target_rad.unsqueeze(0)
        
        # Compute angular distance
        diff = x_pred_rad - x_target_rad
        diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
        dist = diff_wrapped.pow(2)
        
        return self.scale * dist.mean()


@register_loss('toroidal_velocity')
class ToroidalVelocityLoss(BaseLoss):
    """
    Loss that penalizes excessive angular velocities in toroidal geometry.
    Regularization to prevent uncontrolled acceleration in toroidal space.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lambda_v = self.config.get('lambda_v', 0.01)
        self.max_velocity = self.config.get('max_velocity', 1.0)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                state_info: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        if state_info is None or 'v_seq' not in state_info:
            return torch.zeros(1, device=x_pred.device, requires_grad=True)

        v_seq = state_info['v_seq']  # [B, S, H, D]
        
        # For toroidal space, normalize velocities to tangent space
        # Angular velocity should not exceed π per timestep
        v_magnitude = torch.norm(v_seq, dim=-1)
        
        # Penalize velocities exceeding max_velocity
        excess = F.relu(v_magnitude - self.max_velocity)
        return self.lambda_v * excess.mean()


# Alias for export
ToroidalDistanceLoss = ToroidalLoss

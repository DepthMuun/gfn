"""
Geodesic Curvature Regularization
==================================

Regularization based on Christoffel symbols.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. PROBLEM WITH ORIGINAL IMPLEMENTATION:
   Penalizing ||Gamma||^2 can force the manifold toward flat metrics
   (Gamma = 0), which limits the model's capacity to represent complex
   geodesic flows.
   
2. SOLUTION - STRUCTURED REGULARIZATION MODES:
   - 'magnitude': Penalize ||Gamma||^2 (can flatten manifold, limit capacity)
   - 'structural': Penalize RAPID CHANGES in curvature (preserves functionality)
   - 'normalized': Penalize relative to learned curvature scale

3. TEMPORAL STRUCTURE FIX (2026-02-07):
   The structural mode now properly handles temporal ordering of curvatures
   to avoid mixing curvatures from different layers/timesteps incorrectly.

4. RECOMMENDATION:
   Use mode='structural' for most tasks. It preserves useful curvature
   while preventing numerical artifacts.
"""

import torch
from ..constants import GEODESIC_FUSED_SCALE


def geodesic_regularization(christoffel_outputs: list,
                           velocities: list = None,
                           lambda_g: float = 0.001,
                           mode: str = 'structural') -> torch.Tensor:
    """
    Geodesic Curvature Regularization.

    IMPORTANT (Auditoria 2026-02-06): This regularization now has CONFIGURED MODES
    to avoid forcing the manifold toward flat metrics.

    Modes:
    - 'magnitude': Penalize ||Gamma||^2 (can flatten manifold, limit capacity)
    - 'structural': Penalize ||dGamma/dx|| (preserves curvature, prevents artifacts)
    - 'normalized': Penalize relative to batch statistics (adaptive)

    Args:
        christoffel_outputs: List of Christoffel symbol tensors
        velocities: List of velocity tensors (used for structural mode)
        lambda_g: Regularization coefficient
        mode: Regularization mode ('magnitude', 'structural', 'normalized')

    Returns:
        Curvature regularization loss scalar
    """
    if not christoffel_outputs:
        return torch.tensor(0.0)

    # Check if this is a fused regulation tensor (single tensor in list)
    if len(christoffel_outputs) == 1 and christoffel_outputs[0].dim() == 1:
        # Fused case: christoffel_outputs[0] is sum(||Gamma||^2) per batch item
        fused_tensor = christoffel_outputs[0]

        if mode == 'normalized':
            # AUDIT FIX: Normalized mode - use relative scale
            mean_gamma = fused_tensor.mean() + GEODESIC_FUSED_SCALE
            return lambda_g * (fused_tensor / mean_gamma).mean()
        else:
            # Default behavior for fused tensors
            return lambda_g * fused_tensor.mean() / GEODESIC_FUSED_SCALE

    # Standard case: list of tensors
    # AUDIT FIX: Properly handle temporal structure
    # The christoffel_outputs can come from multiple layers and timesteps
    # We need to identify the temporal dimension correctly

    # Stack all curvatures
    all_curvatures = torch.stack(christoffel_outputs)  # [N, ...] where N = total outputs

    if mode == 'magnitude':
        # Original: penalize absolute magnitude
        curvature_norms = all_curvatures.pow(2).mean()

    elif mode == 'structural':
        # AUDIT FIX: Structural mode - penalize changes in curvature
        # This preserves useful curvature while preventing numerical artifacts

        # Handle temporal structure properly
        # We need to identify which dimension is temporal

        if all_curvatures.dim() >= 3:
            # Shape is likely [seq_len, batch, dim] or similar
            # Compute change along the first dimension (temporal)

            # AUDIT FIX: Use proper temporal differencing
            # curvature_change[i] = curv[i+1] - curv[i] for all i
            curvature_diff = all_curvatures[1:] - all_curvatures[:-1]
            curvature_change_norm = curvature_diff.pow(2).mean()

            # Also penalize magnitude but with smaller weight
            # This prevents the manifold from becoming completely flat
            magnitude_norm = all_curvatures.pow(2).mean()

            # Combine: emphasize structural regularization
            # Using 0.85/0.15 split to preserve more curvature
            curvature_norms = 0.85 * curvature_change_norm + 0.15 * magnitude_norm

        elif all_curvatures.dim() == 2:
            # Shape is [N, dim] - variance along N dimension
            # This happens when we have multiple layers at same timestep
            curvature_var = all_curvatures.var(dim=0)  # [dim]
            curvature_norms = curvature_var.mean()

        else:
            # Fallback: simple magnitude
            curvature_norms = all_curvatures.pow(2).mean()

    elif mode == 'normalized':
        # AUDIT FIX: Normalized mode - penalize relative to statistics
        batch_mean = all_curvatures.mean()
        batch_std = all_curvatures.std() + 1e-6

        # Normalized curvature
        normalized = (all_curvatures - batch_mean) / batch_std
        curvature_norms = normalized.pow(2).mean()

    else:
        # Fallback to magnitude
        curvature_norms = all_curvatures.pow(2).mean()

    return lambda_g * curvature_norms


def dynamic_loss_balancing(loss_components: list, target_ratio: float = 1.0) -> list:
    """
    Dynamically balance loss components based on gradient magnitudes.
    
    This prevents any single loss component from dominating optimization.
    
    Args:
        loss_components: List of loss tensors to balance
        target_ratio: Target ratio between gradient magnitudes
        
    Returns:
        List of scaled loss tensors
    """
    if len(loss_components) <= 1:
        return loss_components
    
    # Compute gradients for each component
    grads = []
    for i, loss in enumerate(loss_components):
        if loss.requires_grad:
            try:
                grad = torch.autograd.grad(loss, loss_components[i].parameters() if hasattr(loss_components[i], 'parameters') else None, 
                                          retain_graph=True, create_graph=True)
                if grad:
                    grad_norm = sum(g.norm() for g in grad if g is not None)
                    grads.append(grad_norm)
                else:
                    grads.append(torch.tensor(1.0))
            except:
                grads.append(torch.tensor(1.0))
        else:
            grads.append(torch.tensor(1.0))
    
    # Compute scale factors to equalize gradient magnitudes
    grad_norms = torch.stack([g.detach() if isinstance(g, torch.Tensor) else g for g in grads])
    mean_norm = grad_norms.mean()
    
    scale_factors = []
    for norm in grad_norms:
        if norm > 0:
            scale = target_ratio * mean_norm / norm
        else:
            scale = 1.0
        scale_factors.append(scale)
    
    # Apply scaling
    scaled_losses = []
    for loss, scale in zip(loss_components, scale_factors):
        scaled_losses.append(loss * scale)
    
    return scaled_losses

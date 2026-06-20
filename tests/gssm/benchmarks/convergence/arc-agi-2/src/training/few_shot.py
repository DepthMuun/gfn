"""
Few-Shot Utilities for ARC-AGI-2 GSSM Benchmark

Implements proper few-shot learning where the model conditions on train pairs
before predicting the test output.

Approach: Sequential force conditioning
- Build a force sequence: [input_1, output_1, input_2, output_2, ..., test_input]
- Single forward pass through the model
- Readout at input positions should predict corresponding outputs (auxiliary loss)
- Readout at final position is the test prediction (primary loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


def get_model_embedding(model):
    """Get the embedding module from a model (handles wrappers)."""
    if hasattr(model, 'embedding'):
        return model.embedding
    elif hasattr(model, 'model') and hasattr(model.model, 'embedding'):
        return model.model.embedding
    else:
        raise AttributeError(
            f"Cannot find embedding module in {type(model).__name__}. "
            f"Model must have .embedding or .model.embedding attribute."
        )


def compute_forces(model, grid_flat: torch.Tensor, pad_to: int = 900) -> torch.Tensor:
    """
    Compute forces from a flattened grid using the model's embedding.

    Args:
        model: GSSM model (BaseModel or wrapper)
        grid_flat: [B, H*W] flattened grid values in [0, 9]
        pad_to: Target dimension (default 900 for 30x30 grid)

    Returns:
        forces: [B, 1, D] force vectors
    """
    embedding = get_model_embedding(model)
    
    # Ensure grid_flat is on same device as embedding
    device = next(embedding.parameters()).device
    grid_flat = grid_flat.to(device)
    
    # Pad variable-size grids to fixed dimension
    if grid_flat.shape[-1] < pad_to:
        pad_size = pad_to - grid_flat.shape[-1]
        grid_flat = torch.nn.functional.pad(grid_flat, (0, pad_size), value=0)
    elif grid_flat.shape[-1] > pad_to:
        grid_flat = grid_flat[..., :pad_to]
    # continuous_input expects [B, T, D_in] where T=1 for a single grid
    forces = embedding(continuous_input=grid_flat.unsqueeze(1))  # [B, 1, D]
    return forces


def _crop_and_flatten_grid(grid: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    if grid.dim() == 2:
        if size is not None:
            h, w = int(size[0]), int(size[1])
            grid = grid[:h, :w]
        return grid.flatten()
    if grid.dim() == 1:
        if size is not None:
            h, w = int(size[0]), int(size[1])
            n = h * w
            grid = grid[:n]
        return grid
    return grid.reshape(-1)


def build_fewshot_forces(
    model,
    train_pairs: List[Dict],
    test_input: torch.Tensor,
    device: str = 'cpu',
    test_input_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[int], List[torch.Tensor]]:
    """
    Build the force sequence for few-shot learning.

    Sequence layout: [input_1, output_1, input_2, output_2, ..., test_input]
    - At even indices (0, 2, 4, ...): input forces → readout should predict output
    - At odd indices (1, 3, 5, ...): output forces → context for next pair
    - At final index: test input forces → readout is the test prediction

    Args:
        model: GSSM model
        train_pairs: List of dicts with 'input' and 'output' tensors [H, W] or [H*W]
        test_input: Test input tensor [H, W] or [H*W]
        device: Device string

    Returns:
        forces: [1, T, D] stacked force sequence
        prediction_timesteps: List of timestep indices where predictions should match targets
        target_grids: List of target grids (flattened) corresponding to prediction_timesteps
    """
    force_list = []
    prediction_timesteps = []
    target_grids = []

    for i, pair in enumerate(train_pairs):
        # Input force
        input_grid = pair['input'].to(device)
        input_flat_1d = _crop_and_flatten_grid(input_grid, pair.get('input_size'))
        input_flat = input_flat_1d.unsqueeze(0)

        input_forces = compute_forces(model, input_flat)  # [1, 1, D]
        force_list.append(input_forces.squeeze(1))  # [1, D]

        # Record: readout at this timestep should predict the output
        input_timestep = i * 2
        prediction_timesteps.append(input_timestep)

        output_grid = pair['output'].to(device)
        output_flat = _crop_and_flatten_grid(output_grid, pair.get('output_size')).to(device)
        target_grids.append(output_flat)

        # Output force (as context for the model to learn the transformation)
        output_forces = compute_forces(model, output_flat.unsqueeze(0))  # [1, 1, D]
        force_list.append(output_forces.squeeze(1))  # [1, D]

    # Test input force
    test_grid = test_input.to(device)
    test_flat_1d = _crop_and_flatten_grid(test_grid, test_input_size)
    test_flat = test_flat_1d.unsqueeze(0)

    test_forces = compute_forces(model, test_flat)  # [1, 1, D]
    force_list.append(test_forces.squeeze(1))  # [1, D]

    # Final timestep is the test prediction
    test_timestep = len(train_pairs) * 2
    prediction_timesteps.append(test_timestep)

    # Stack into sequence: [1, T, D]
    forces = torch.stack(force_list, dim=1)

    return forces, prediction_timesteps, target_grids


def extract_predictions(
    logits: torch.Tensor,
    prediction_timesteps: List[int]
) -> List[torch.Tensor]:
    """
    Extract readout predictions at specified timesteps.

    Args:
        logits: [B, T, out_dim] readout outputs from model
        prediction_timesteps: List of timestep indices

    Returns:
        List of prediction tensors, each [B, out_dim]
    """
    predictions = []
    for t in prediction_timesteps:
        if t < logits.shape[1]:
            predictions.append(logits[:, t, :])
        else:
            # Fallback: use last timestep
            predictions.append(logits[:, -1, :])
    return predictions


def fewshot_loss(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    test_prediction: torch.Tensor,
    test_target: torch.Tensor,
    auxiliary_weight: float = 0.5
) -> torch.Tensor:
    """
    Compute few-shot loss combining auxiliary (train pair) and primary (test) losses.

    Args:
        predictions: List of train pair predictions [B, out_dim]
        targets: List of train pair targets [H*W]
        test_prediction: Test prediction [B, out_dim]
        test_target: Test target [H*W]
        auxiliary_weight: Weight for auxiliary loss relative to primary loss

    Returns:
        Combined loss scalar
    """
    # Primary loss: test prediction
    test_target_batch = test_target.unsqueeze(0)  # [1, H*W]
    primary_loss = grid_loss(test_prediction, test_target_batch)

    # Auxiliary loss: train pair predictions
    if predictions and auxiliary_weight > 0:
        aux_losses = []
        for pred, target in zip(predictions, targets):
            target_batch = target.unsqueeze(0)  # [1, H*W]
            # Match dimensions: prediction may differ from target
            min_dim = min(pred.shape[-1], target_batch.shape[-1])
            aux_losses.append(grid_loss(
                pred[..., :min_dim],
                target_batch[..., :min_dim]
            ))
        aux_loss = torch.stack(aux_losses).mean()
    else:
        aux_loss = torch.tensor(0.0, device=test_prediction.device)

    # Ensure all losses on same device
    device = test_prediction.device
    primary_loss = primary_loss.to(device)
    aux_loss = aux_loss.to(device)
    
    return primary_loss + auxiliary_weight * aux_loss


def grid_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss function for grid prediction in [0, 9] value space.

    Combines:
    - MSE for continuous values
    - Range penalty to keep predictions in [0, 9]
    - Integer penalty to push predictions toward exact integer values

    Args:
        prediction: [B, N] predicted grid values
        target: [B, N] target grid values in [0, 9]

    Returns:
        Scalar loss
    """
    # Ensure both tensors on same device
    if target.device != prediction.device:
        target = target.to(prediction.device)
    
    # MSE base loss
    mse = F.mse_loss(prediction, target)

    # Range penalty: penalize values outside [0, 9]
    range_penalty = F.relu(prediction - 9.0).mean() + F.relu(-prediction).mean()

    # Integer penalty: push predictions toward nearest integer
    nearest_int = prediction.round().detach()
    distance_to_int = (prediction - nearest_int).abs().mean()
    integer_penalty = distance_to_int * 0.5

    return mse + 0.1 * range_penalty + integer_penalty


def prediction_to_grid(
    pred_tensor: torch.Tensor,
    original_size: Tuple[int, int] = None,
    max_grid_size: int = 30
) -> torch.Tensor:
    """
    Convert raw prediction tensor to integer grid values.

    Args:
        pred_tensor: [N] or [H, W] prediction tensor
        original_size: (H, W) of original grid (before padding)
        max_grid_size: Maximum grid size used for padding

    Returns:
        Integer grid tensor [H, W] with values in [0, 9]
    """
    if pred_tensor.dim() == 1:
        # Flatten prediction → reshape to 2D
        grid = pred_tensor.reshape(max_grid_size, max_grid_size)
    elif pred_tensor.dim() == 2:
        grid = pred_tensor
    else:
        grid = pred_tensor.squeeze()

    # Round and clip to valid ARC range
    grid = grid.round().clamp(0, 9).long()

    # Crop to original size if provided
    if original_size is not None:
        h, w = original_size
        grid = grid[:h, :w]

    return grid

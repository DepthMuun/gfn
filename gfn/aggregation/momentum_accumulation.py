"""
Momentum Accumulation
=====================

Trajectory-based integration of momentum over geodesic flow.

Accumulates momentum (velocity) from entire trajectory and adds
it to final position state.

Physics Motivation:
- Total impulse = ∫ force dt determines final momentum
- Trajectory history influences final state
- Natural for sequential processing where each token "pushes" state
"""

import torch
import torch.nn as nn


class MomentumAccumulation(nn.Module):
    """
    Accumulate states from geodesic trajectory to final position.
    
    Final state receives contribution from accumulated trajectory:
        x_final = x_last + alpha * accumulated_states
    
    This module aggregates the trajectory history and adds it to the final position,
    allowing the model to incorporate information from the entire forward pass.
    
    Note on v_seq parameter: This parameter is accepted for API compatibility but
    is NOT used. The module only accumulates position states (x_seq), not velocities.
    The returned v_final is always zeros, representing no velocity information.
    
    Args:
        dim: State dimension
        alpha: Trajectory contribution weight (default: 0.1)
        mode: 'sum' or 'avg' for accumulation (default: 'avg')
            - 'sum': Total states ∑x_t
            - 'avg': Average states (1/L)∑x_t
        gated: If True, learn gating for trajectory contribution (default: False)
    
    Example:
        >>> momentum_agg = MomentumAccumulation(dim=128, alpha=0.2, mode='avg', gated=True)
        >>> x_final, v_final, accumulated = momentum_agg(x_seq)  # v_seq is ignored
        >>> # x_final includes trajectory contribution from entire sequence
    """
    
    def __init__(self, dim, alpha=0.1, mode='avg', gated=False):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.mode = mode
        self.gated = gated
        
        if gated:
            # Learnable gating for momentum contribution
            # Gate depends on final state (x, v)
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),  # Input: [x_final, v_final] concatenated
                nn.Tanh(),
                nn.Linear(dim, 1),
                nn.Sigmoid()  # Output: scalar gate in [0, 1]
            )
    
    def forward(self, x_seq, v_seq=None):
        """
        Aggregate by accumulating states over trajectory.
        
        Note: v_seq parameter is accepted for API compatibility but is NOT used.
        This module only accumulates position states (x_seq).
        
        Args:
            x_seq: Position sequence [B, L, dim]
            v_seq: Velocity sequence [B, L, dim] (optional, NOT USED)
        
        Returns:
            x_final: Final position with state accumulation [B, dim]
            v_final: Final velocity (zeros) [B, dim] - always zeros
            accumulated: Accumulated states [B, dim] for interpretability
        """
        B, L, dim = x_seq.shape
        
        # Accumulate STATES over trajectory (not velocity)
        # This represents: "how much did the trajectory explore the manifold"
        if self.mode == 'sum':
            accumulated_states = x_seq.sum(dim=1)  # [B, dim]
        else:  # 'avg'
            accumulated_states = x_seq.mean(dim=1)  # [B, dim]
        
        # Final state (last token)
        x_last = x_seq[:, -1]  # [B, dim]
        v_final = torch.zeros_like(x_last)  # No velocity available
        
        # Compute gating if enabled
        if self.gated:
            # Gate based on final state and accumulated trajectory
            gate_input = torch.cat([x_last, accumulated_states], dim=-1)  # [B, 2*dim]
            gate_value = self.gate(gate_input)  # [B, 1]
            effective_alpha = self.alpha * gate_value
        else:
            effective_alpha = self.alpha
        
        # Add trajectory contribution to final position
        # This is: "where you are" + alpha * "where you've been on average"
        x_final = x_last + effective_alpha * accumulated_states
        
        return x_final, v_final, accumulated_states

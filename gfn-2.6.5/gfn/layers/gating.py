import torch
import torch.nn as nn

class RiemannianGating(nn.Module):
    """
    Computes a scalar curvature-based gating mechanism.
    If curvature is high, dt should be small (complex region).
    If curvature is low (flat), dt can be large (skip connection behavior).
    """
    def __init__(self, dim, topology=0):
        super().__init__()
        self.topology = topology
        # For Torus, we use sin(x) and cos(x) as inputs, so input_dim = 2*dim
        input_dim = 2 * dim if topology == 1 else dim
        self.curvature_net = nn.Sequential(
            nn.Linear(input_dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid() # Range [0, 1]
        )
        
        # Level 31: KICKSTART GATING
        # Initialize bias high so sigmoid(bias) is close to 1.0
        # This prevents the initial dt from being throttled to 0.5 (sigmoid(0))
        with torch.no_grad():
            nn.init.constant_(self.curvature_net[2].bias, 2.0)
            nn.init.xavier_uniform_(self.curvature_net[0].weight, gain=0.1)
            nn.init.xavier_uniform_(self.curvature_net[2].weight, gain=0.1)
        
    def forward(self, x):
        """
        Returns a scaling factor for dt.
        """
        if self.topology == 1:
             # Map to periodic space
             x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
             
        # Try CUDA path
        try:
            from gfn.cuda.ops import dynamic_gating_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and x.is_cuda:
                W1 = self.curvature_net[0].weight  # [dim/4, dim]
                b1 = self.curvature_net[0].bias    # [dim/4]
                W2 = self.curvature_net[2].weight  # [1, dim/4]
                b2 = self.curvature_net[2].bias    # [1]
                return dynamic_gating_fused(x, W1, b1, W2, b2)
        except Exception:
            pass
        
        # Fallback PyTorch
        return self.curvature_net(x)

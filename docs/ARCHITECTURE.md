# Manifold Architecture Reference

## Overview

This document provides a complete technical reference for the Manifold architecture, describing each component, its inputs and outputs, internal structure, and configuration options. The architecture is presented as a layered system where each layer builds upon the foundations established by previous layers, enabling both understanding of individual components and appreciation of their integration into the complete system.

Manifold implements a Geometric Flow Network (GFN) where sequence modeling is reformulated as particle dynamics on a learned Riemannian manifold. The architecture consists of five major subsystems: the embedding layer transforms discrete tokens into continuous force vectors; the M-layer (manifold layer) implements the core geodesic dynamics with attention-like interactions; the integration layer numerically evolves the state according to physical principles; the readout layer produces predictions from the manifold state; and the active inference layer provides adaptive modulation of dynamics based on uncertainty estimates.

Understanding this architecture requires familiarity with the mathematical foundations presented in the Mathematical Foundations document. This reference assumes that readers understand manifolds, Christoffel symbols, Hamiltonian mechanics, and symplectic integration. Where mathematical details are relevant to implementation, they are provided explicitly.

## System Architecture

### Data Flow Overview

The Manifold data flow transforms discrete token sequences into continuous manifold dynamics and back to discrete predictions:

```
Input Tokens [B, L]
    │
    ▼
Embedding Layer: Tokens → Force Vectors F [B, L, D]
    │
    ▼
M-Layer Stack: (x₀, v₀), F → (x₁, v₁), ..., (xₙ, vₙ) [B, L, 2D]
    │
    ▼
Readout Layer: Final State → Logits [B, L, V]
    │
    ▼
Output Predictions
```

During training, all L tokens are processed in parallel, enabling efficient batch computation. During inference, tokens are processed autoregressively with state persistence, maintaining O(1) memory regardless of sequence length.

### State Representation

The Manifold state consists of position and velocity components that together encode the complete context:

The position component $x \in \mathbb{R}^{d}$ represents the semantic location in the manifold. Points that are close in the manifold geometry correspond to semantically similar states. The position evolves according to the geodesic equation, curving in response to the manifold curvature encoded in Christoffel symbols.

The velocity component $v \in \mathbb{R}^{d}$ represents the momentum of the system, which serves as the memory mechanism. Unlike explicit memory stores in other architectures, momentum encodes context implicitly through the history of interactions. High momentum indicates strong preservation of previous state; low momentum indicates recent context dominates.

The force component $F \in \mathbb{R}^{d}$ is derived from the input token embedding and acts as an external input that pushes the system through the manifold. Different tokens produce different force vectors, causing the state to move in different directions through the semantic space.

### Configuration Hierarchy

Manifold uses a hierarchical configuration system where each level corresponds to a major subsystem:

```python
config = {
    'embedding': {...},       # Embedding configuration
    'readout': {...},         # Readout configuration
    'active_inference': {...}, # Adaptive dynamics
    'fractal': {...},         # Hierarchical structure
    'topology': {...},        # Global structure
    'stability': {...}        # Numerical parameters
}
```

This structure ensures consistency between components while allowing fine-grained control. Each subsystem validates its configuration during initialization and provides meaningful error messages for invalid settings.

## Embedding Layer

### FunctionalEmbedding

The FunctionalEmbedding class implements neural field-based embeddings using SIREN (Sinusoidal Representation Networks):

```python
class FunctionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        coord_dim: int = 16,
        mode: str = 'linear',
        hidden_dim: int = None
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            emb_dim: Output embedding dimension
            coord_dim: Coordinate dimension for neural field
            mode: 'linear' or 'binary' coordinate encoding
            hidden_dim: Hidden dimension for SIREN MLP
        """
```

The forward pass transforms token IDs through coordinate encoding and SIREN evaluation:

```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    # Coordinate encoding: [B, L] → [B, L, coord_dim]
    coords = self.encode_coordinates(input_ids)
    
    # SIREN evaluation: [B, L, coord_dim] → [B, L, emb_dim]
    embeddings = self.siren(coords)
    
    return embeddings
```

The coordinate encoding maps token IDs to points in a coordinate space. Linear mode produces smooth interpolation between token representations:

```python
def encode_coordinates_linear(self, token_ids: torch.Tensor) -> torch.Tensor:
    # Normalize token IDs to [0, 1]
    coords = token_ids.float() / (self.vocab_size - 1)
    # Scale to [-1, 1]
    coords = coords * 2 - 1
    # Expand to coordinate dimension
    return coords.unsqueeze(-1).expand(-1, -1, self.coord_dim)
```

Binary mode produces discrete binary coordinates:

```python
def encode_coordinates_binary(self, token_ids: torch.Tensor) -> torch.Tensor:
    # Convert to binary representation
    binary = token_ids.unsqueeze(-1).bitwise_and(
        2 ** torch.arange(self.coord_dim, device=token_ids.device)
    ) > 0
    # Convert True/False to 1.0/0.0
    return binary.float()
```

The SIREN module uses sinusoidal activations with specific initialization:

```python
class SIREN(nn.Module):
    def __init__(self, coord_dim, hidden_dim, out_dim, num_layers=3):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(coord_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
        # Initialize for first-layer frequency ω₀ = 30
        self.layers[0].weight.data.uniform_(-1/coord_dim, 1/coord_dim)
        self.layers[0].bias.data.zero_()
        
        # Initialize hidden layers for ω₀ = 1
        for layer in self.layers[1:-1]:
            layer.weight.data.uniform_(-np.sqrt(6/hidden_dim)/30, 
                                       np.sqrt(6/hidden_dim)/30)
            layer.bias.data.zero_()
    
    def forward(self, coords):
        x = coords
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))
        return self.layers[-1](x)
```

### ImplicitEmbedding

The ImplicitEmbedding class uses a learnable coordinate table:

```python
class ImplicitEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        coord_dim: int = 16,
        learn_coords: bool = True
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            emb_dim: Output embedding dimension
            coord_dim: Coordinate dimension
            learn_coords: Whether to learn coordinate positions
        """
        super().__init__()
        self.coord_dim = coord_dim
        
        # Learnable coordinate table
        if learn_coords:
            self.coordinates = nn.Parameter(
                torch.randn(vocab_size, coord_dim) * 0.1
            )
        else:
            self.register_buffer('coordinates', 
                                torch.randn(vocab_size, coord_dim) * 0.1)
        
        # Coordinate to embedding projection
        self.proj = nn.Linear(coord_dim, emb_dim)
    
    def forward(self, input_ids):
        coords = self.coordinates[input_ids]
        return self.proj(coords)
```

### Embedding Factory

The embedding system provides a factory function for creating embeddings:

```python
def create_embedding(config: dict) -> nn.Module:
    """Create embedding layer from configuration."""
    embedding_type = config.get('type', 'functional')
    
    if embedding_type == 'functional':
        return FunctionalEmbedding(
            vocab_size=config['vocab_size'],
            emb_dim=config['dim'],
            coord_dim=config.get('coord_dim', 16),
            mode=config.get('mode', 'linear')
        )
    elif embedding_type == 'implicit':
        return ImplicitEmbedding(
            vocab_size=config['vocab_size'],
            emb_dim=config['dim'],
            coord_dim=config.get('coord_dim', 16)
        )
    elif embedding_type == 'standard':
        return nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['dim']
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
```

## M-Layer (Manifold Layer)

### Layer Structure

The M-Layer implements the core geodesic dynamics, replacing the attention mechanism in Transformers:

```python
class ManifoldLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        integrator_type: str = 'leapfrog',
        physics_config: dict = None,
        dropout: float = 0.0
    ):
        """
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            integrator_type: Type of symplectic integrator
            physics_config: Configuration for physics parameters
            dropout: Dropout rate
        """
```

The forward pass implements multi-head geodesic flow:

```python
def forward(
    self,
    x: torch.Tensor,
    v: torch.Tensor,
    F: torch.Tensor,
    context: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Position tensor [B, L, D]
        v: Velocity tensor [B, L, D]
        F: Force tensor [B, L, D]
        context: Inter-layer context [B, L, D] (optional)
    
    Returns:
        x_out: Updated position [B, L, D]
        v_out: Updated velocity [B, L, D]
        trajectory: List of intermediate states
    """
```

### Christoffel Computation

The Christoffel symbols are computed using a low-rank parameterization:

```python
class ChristoffelModule(nn.Module):
    def __init__(self, dim: int, rank: int = 16):
        """
        Args:
            dim: Model dimension
            rank: Low-rank approximation rank
        """
        super().__init__()
        self.dim = dim
        self.rank = rank
        
        # Low-rank parameterization
        self.U = nn.Linear(dim, rank, bias=False)
        self.W = nn.Linear(rank, dim, bias=False)
        
        # Gate network for adaptive gating
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1)
        )
        
        # Position modulation
        self.pos_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Christoffel symbols from velocity and position.
        
        Args:
            v: Velocity tensor [B, L, D]
            x: Position tensor [B, L, D]
        
        Returns:
            Gamma: Christoffel symbols [B, L, D]
        """
        # Low-rank velocity projection
        v_proj = self.U(v)  # [B, L, rank]
        v_squared = v_proj ** 2  # Quadratic interaction
        
        # Saturation for numerical stability
        v_norm = v.norm(dim=-1, keepdim=True)
        saturation = torch.sigmoid(v_norm / 10.0)  # Scale factor
        
        # Base Christoffel symbols
        Gamma_base = self.W(v_squared * saturation)
        
        # Adaptive gating
        gate = torch.sigmoid(self.gate_net(x))  # [B, L, 1]
        Gamma_gated = gate * Gamma_base
        
        # Position modulation
        pos_scale = 1.0 + torch.tanh(self.pos_net(x))  # [B, L, 1]
        Gamma = Gamma_gated * pos_scale
        
        # Clamp for numerical stability
        Gamma = torch.clamp(Gamma, -5.0, 5.0)
        
        return Gamma
```

### Multi-Head Architecture

The multi-head architecture processes the state through independent geodesic flows:

```python
class MultiHeadManifold(nn.Module):
    def __init__(self, dim: int, num_heads: int, **kwargs):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        
        # Head-specific Christoffel modules
        self.christoffel_heads = nn.ModuleList([
            ChristoffelModule(self.head_dim, rank=self.head_dim // 4)
            for _ in range(num_heads)
        ])
        
        # Head-specific gates
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 4),
                nn.SiLU(),
                nn.Linear(self.head_dim // 4, 1)
            )
            for _ in range(num_heads)
        ])
        
        # Head mixing projections
        self.x_mix = nn.Linear(dim, dim)
        self.v_mix = nn.Linear(dim, dim)
        
        # Pre-layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, v: torch.Tensor, F: torch.Tensor):
        B, L, D = x.shape
        
        # Normalize position
        x_norm = self.norm(x)
        
        # Split into heads
        x_heads = x_norm.view(B, L, self.num_heads, self.head_dim)
        v_heads = v.view(B, L, self.num_heads, self.head_dim)
        F_heads = F.view(B, L, self.num_heads, self.head_dim)
        
        x_out_heads = []
        v_out_heads = []
        
        for h in range(self.num_heads):
            x_h = x_heads[:, :, h]
            v_h = v_heads[:, :, h]
            F_h = F_heads[:, :, h]
            
            # Compute Christoffel symbols
            christoffel = self.christoffel_heads[h](v_h, x_h)
            
            # Compute gate
            gate = torch.sigmoid(self.gate_heads[h](x_h))
            
            # Compute friction
            friction_scale = gate * 5.0  # Scales to [0, 5]
            
            # Net force
            net_force = F_h - christoffel - friction_scale * v_h
            
            # Update velocity
            v_h = v_h + 0.5 * net_force
            
            # Update position
            x_h = x_h + 0.4 * v_h  # base_dt = 0.4
            
            # Recompute Christoffel for half-step
            christoffel_new = self.christoffel_heads[h](v_h, x_h)
            friction_new = friction_scale * v_h
            
            # Complete velocity update
            net_force_new = F_h - christoffel_new - friction_new
            v_h = v_h + 0.5 * net_force_new
            
            # Normalize velocity (critical for stability)
            v_h = v_h / (v_h.norm(dim=-1, keepdim=True) + 1e-6)
            
            x_out_heads.append(x_h)
            v_out_heads.append(v_h)
        
        # Concatenate heads
        x_out = torch.cat(x_out_heads, dim=-1)
        v_out = torch.cat(v_out_heads, dim=-1)
        
        # Head mixing
        x_out = self.x_mix(x_out)
        v_out = self.v_mix(v_out)
        
        return x_out, v_out
```

## Symplectic Integrators

### Base Integrator Class

All integrators inherit from a base class that provides common functionality:

```python
class SymplecticIntegrator(nn.Module):
    def __init__(self, dt: float = 0.4):
        """
        Args:
            dt: Integration timestep
        """
        super().__init__()
        self.dt = dt
    
    def step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        F: torch.Tensor,
        christoffel_fn: callable,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one integration step.
        
        Args:
            x: Position tensor
            v: Velocity tensor
            F: Force tensor
            christoffel_fn: Function to compute Christoffel symbols
        
        Returns:
            x_new, v_new: Updated position and velocity
        """
        raise NotImplementedError
    
    def _normalize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize velocity to prevent explosion."""
        norm = v.norm(dim=-1, keepdim=True)
        return v / (norm + 1e-6)
```

### Leapfrog Integrator

The leapfrog (Velocity Verlet) integrator provides second-order accuracy with symplectic structure:

```python
class LeapfrogIntegrator(SymplecticIntegrator):
    """Velocity Verlet (Leapfrog) symplectic integrator."""
    
    def step(self, x, v, F, christoffel_fn, **kwargs):
        # First half-step in velocity
        gamma = christoffel_fn(v, x)
        v_half = v + 0.5 * self.dt * (F - gamma)
        
        # Full step in position
        x_new = x + self.dt * v_half
        
        # Second half-step in velocity
        gamma_new = christoffel_fn(v_half, x_new)
        v_new = v_half + 0.5 * self.dt * (F - gamma_new)
        
        # Normalize velocity (critical for stability)
        v_new = self._normalize_velocity(v_new)
        
        return x_new, v_new
```

### Forest-Ruth Integrator

The Forest-Ruth integrator provides fourth-order accuracy:

```python
class ForestRuthIntegrator(SymplecticIntegrator):
    """Forest-Ruth fourth-order symplectic integrator."""
    
    # Coefficients for Forest-Ruth
    THETA = 0.1786178958448091
    LAMBDA = -0.2123416400622214
    CHI = -0.0662645826693094
    
    def step(self, x, v, F, christoffel_fn, **kwargs):
        dt = self.dt
        
        # Substeps following Forest-Ruth coefficients
        x_new, v_new = self._forest_ruth_step(x, v, F, christoffel_fn, dt)
        
        return x_new, v_new
    
    def _forest_ruth_step(self, x, v, F, christoffel_fn, dt):
        """Single Forest-Ruth step."""
        theta = self.THETA
        lam = self.LAMBDA
        chi = self.CHI
        
        # Phase 1: Position update by theta * dt/2
        x1 = x + theta * dt * v
        
        # Phase 2: Velocity update by (1 - 2*lambda) * dt/2
        v1 = v + (1 - 2 * lam) * dt / 2 * F
        gamma1 = christoffel_fn(v1, x1)
        v1 = v1 - lam * dt * gamma1
        
        # Phase 3: Position update by chi * dt
        x2 = x1 + chi * dt * v1
        
        # Phase 4: Velocity update by lambda * dt
        gamma2 = christoffel_fn(v1, x2)
        v2 = v1 - lam * dt * gamma2
        
        # Phase 5: Position update by (1 - 2*theta) * dt
        x3 = x2 + (1 - 2 * theta - 2 * chi) * dt * v2
        
        # Phase 6: Velocity update by lambda * dt
        gamma3 = christoffel_fn(v2, x3)
        v3 = v2 - lam * dt * gamma3
        
        # Phase 7: Position update by chi * dt
        x4 = x3 + chi * dt * v3
        
        # Phase 8: Velocity update by (1 - 2*lambda) * dt/2
        gamma4 = christoffel_fn(v3, x4)
        v4 = v3 + (1 - 2 * lam) * dt / 2 * F
        v4 = v4 - lam * dt * gamma4
        
        # Phase 9: Position update by theta * dt/2
        x_new = x4 + theta * dt * v4
        
        # Normalize velocity
        v_new = self._normalize_velocity(v4)
        
        return x_new, v_new
```

### Integrator Factory

A factory function creates integrators from configuration:

```python
def create_integrator(config: dict) -> SymplecticIntegrator:
    """Create integrator from configuration."""
    integrator_type = config.get('type', 'leapfrog')
    dt = config.get('base_dt', 0.4)
    
    integrators = {
        'leapfrog': LeapfrogIntegrator,
        'forest_ruth': ForestRuthIntegrator,
        'heun': HeunIntegrator,
        'rk4': RK4Integrator,
    }
    
    if integrator_type not in integrators:
        raise ValueError(f"Unknown integrator type: {integrator_type}")
    
    return integrators[integrator_type](dt=dt)
```

## Readout Layer

### Implicit Readout

The implicit readout uses a neural field to decode manifold state to token predictions:

```python
class ImplicitReadout(nn.Module):
    def __init__(self, dim: int, vocab_size: int, coord_dim: int = 16):
        """
        Args:
            dim: Model dimension
            vocab_size: Number of tokens in vocabulary
            coord_dim: Coordinate dimension for inverse field
        """
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        
        # Project manifold state to coordinates
        self.coord_proj = nn.Linear(dim, coord_dim)
        
        # Inverse SIREN for coordinate to logit mapping
        self.inverse_siren = SIREN(
            coord_dim=coord_dim,
            hidden_dim=vocab_size // 2,
            out_dim=vocab_size,
            num_layers=3
        )
    
    def forward(self, x_final: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_final: Final position state [B, L, D]
        
        Returns:
            logits: Output logits [B, L, V]
        """
        # Project to coordinate space
        coords = self.coord_proj(x_final)
        
        # Normalize coordinates to valid range
        coords = torch.tanh(coords)
        
        # Evaluate inverse field
        logits = self.inverse_siren(coords)
        
        return logits
```

### Explicit Readout

The explicit readout uses a simple linear projection:

```python
class ExplicitReadout(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size)
    
    def forward(self, x_final: torch.Tensor) -> torch.Tensor:
        return self.proj(x_final)
```

## Active Inference Module

### Uncertainty Estimation

Active inference requires estimating uncertainty in the current state:

```python
class UncertaintyEstimator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty from position and velocity.
        
        Args:
            x: Position tensor
            v: Velocity tensor
        
        Returns:
            uncertainty: Uncertainty estimate [B, L, 1]
        """
        state = torch.cat([x, v], dim=-1)
        return self.estimator(state)
```

### Dynamic Time Warping

Adaptive timestep based on uncertainty:

```python
class DynamicTimeWarping(nn.Module):
    def __init__(self, base_dt: float = 0.4, min_dt: float = 0.1, max_dt: float = 0.8):
        super().__init__()
        self.base_dt = base_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
    
    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive timestep based on uncertainty.
        
        Args:
            uncertainty: Uncertainty estimate [B, L, 1]
        
        Returns:
            dt: Adaptive timestep [B, L, 1]
        """
        # High uncertainty → smaller timestep
        dt = self.base_dt / (1.0 + 5.0 * uncertainty)
        dt = torch.clamp(dt, self.min_dt, self.max_dt)
        return dt
```

### Reactive Curvature

Adaptive curvature modulation:

```python
class ReactiveCurvature(nn.Module):
    def __init__(self, dim: int, plasticity: float = 0.2):
        super().__init__()
        self.plasticity = plasticity
        
        # Plasticity scalar network
        self.plasticity_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Tanh()
        )
    
    def forward(
        self,
        Gamma: torch.Tensor,
        uncertainty: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate curvature based on uncertainty.
        
        Args:
            Gamma: Base Christoffel symbols
            uncertainty: Uncertainty estimate
            x: Position state
        
        Returns:
            Gamma_eff: Modulated Christoffel symbols
        """
        plasticity = self.plasticity * (1.0 + self.plasticity_net(x))
        
        # Higher uncertainty → more plasticity
        Gamma_eff = Gamma * (1.0 + plasticity * uncertainty)
        
        return Gamma_eff
```

### Complete Active Inference Module

```python
class ActiveInference(nn.Module):
    def __init__(self, dim: int, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        
        self.uncertainty = UncertaintyEstimator(dim)
        
        if config.get('dynamic_time', {}).get('enabled', False):
            self.dynamic_time = DynamicTimeWarping(
                base_dt=config.get('stability', {}).get('base_dt', 0.4)
            )
        else:
            self.dynamic_time = None
        
        if config.get('reactive_curvature', {}).get('enabled', False):
            self.reactive_curvature = ReactiveCurvature(
                dim,
                plasticity=config['reactive_curvature'].get('plasticity', 0.2)
            )
        else:
            self.reactive_curvature = None
        
        if config.get('singularities', {}).get('enabled', False):
            self.singularities = SingularityModule(
                dim,
                strength=config['singularities'].get('strength', 20.0),
                threshold=config['singularities'].get('threshold', 0.8)
            )
        else:
            self.singularities = None
    
    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        F: torch.Tensor,
        christoffel_fn: callable
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Apply active inference modulation.
        
        Returns:
            x, v: Modulated state
            info: Diagnostic information
        """
        if not self.enabled:
            return x, v, {}
        
        info = {}
        
        # Estimate uncertainty
        uncertainty = self.uncertainty(x, v)
        info['uncertainty'] = uncertainty
        
        # Adaptive timestep
        if self.dynamic_time is not None:
            dt = self.dynamic_time(uncertainty)
            info['dt'] = dt
        else:
            dt = None
        
        # Compute Christoffel symbols
        Gamma = christoffel_fn(v, x)
        
        # Reactive curvature
        if self.reactive_curvature is not None:
            Gamma = self.reactive_curvature(Gamma, uncertainty, x)
        
        # Singularities
        if self.singularities is not None:
            Gamma = self.singularities(Gamma, uncertainty)
        
        return Gamma, dt, info
```

## Main Model Class

### Manifold Model

The complete Manifold model integrates all components:

```python
class Manifold(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 4,
        integrator_type: str = 'leapfrog',
        physics_config: dict = None,
        impulse_scale: float = 80.0,
        holographic: bool = True,
        **kwargs
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            dim: Model dimension
            depth: Number of M-layers
            heads: Number of attention heads
            integrator_type: Type of symplectic integrator
            physics_config: Configuration for physics parameters
            impulse_scale: Scale factor for input impulses
            holographic: Use holographic state representation
        """
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.impulse_scale = impulse_scale
        self.holographic = holographic
        
        # Default configuration
        if physics_config is None:
            physics_config = self._default_config()
        
        self.physics_config = physics_config
        
        # Embedding layer
        self.embedding = self._create_embedding(vocab_size, dim, physics_config)
        
        # M-Layers
        self.layers = nn.ModuleList([
            ManifoldLayer(
                dim=dim,
                num_heads=heads,
                integrator_type=integrator_type,
                physics_config=physics_config
            )
            for _ in range(depth)
        ])
        
        # Readout layer
        self.readout = self._create_readout(dim, vocab_size, physics_config)
        
        # Active inference module
        self.active_inference = ActiveInference(dim, physics_config.get('active_inference', {}))
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'embedding': {
                'type': 'functional',
                'mode': 'linear',
                'coord_dim': 16
            },
            'readout': {
                'type': 'implicit',
                'coord_dim': 16
            },
            'active_inference': {
                'enabled': True,
                'dynamic_time': {'enabled': True},
                'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
                'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
            },
            'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
            'topology': {'type': 'torus'},
            'stability': {'base_dt': 0.4}
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], list]:
        """
        Forward pass through Manifold.
        
        Args:
            input_ids: Token IDs [B, L]
            state: Previous state (x, v) for inference [B, D], [B, D]
        
        Returns:
            logits: Output logits [B, L, V]
            state: Final state (x, v)
            trajectory: List of intermediate states
        """
        B, L = input_ids.shape
        
        # Embed input tokens
        F = self.embedding(input_ids) * self.impulse_scale
        
        # Initialize or use provided state
        if state is None:
            x = torch.zeros(B, L, self.dim, device=input_ids.device)
            v = torch.zeros(B, L, self.dim, device=input_ids.device)
        else:
            x, v = state
            # Expand to sequence dimension
            x = x.unsqueeze(1).expand(-1, L, -1)
            v = v.unsqueeze(1).expand(-1, L, -1)
        
        trajectory = []
        
        # Process through M-layers
        for layer in self.layers:
            # Get Christoffel function for this layer
            christoffel_fn = lambda v, x, layer=layer: layer.christoffel(v, x)
            
            # Apply active inference
            if self.physics_config.get('active_inference', {}).get('enabled', False):
                Gamma, dt, info = self.active_inference(x, v, F, christoffel_fn)
                # Apply active inference modulation
                x, v = layer(x, v, F, christoffel_fn)
            else:
                x, v = layer(x, v, F, christoffel_fn)
            
            trajectory.append((x.clone(), v.clone()))
        
        # Readout
        logits = self.readout(x)
        
        # Return final state for inference (use last position)
        final_state = (x[:, -1], v[:, -1])
        
        return logits, final_state, trajectory
```

## Summary

This architecture reference has described the complete Manifold system, from embedding layers through active inference modules. The modular design allows individual components to be replaced or modified while maintaining the overall geometric framework. The configuration system provides flexibility while ensuring consistency between components.

For additional information on mathematical foundations, consult the Mathematical Foundations document. For practical usage examples, consult the Tutorial document. For API documentation, consult the API Reference.

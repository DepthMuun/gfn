# Class and API Reference

## 1. Top-Level Imports (gfn.api_registry)

In DepthMuun v2, you do not need to traverse deep internal module paths to import classes. You can access them directly via the `gfn` module, powered by the `api_registry`.

```python
import gfn
# E.g., gfn.LeapfrogIntegrator, gfn.RiemannianAdam, gfn.Config
```

However, if you wish to inspect or import from the underlying files, the direct modular paths are documented below.

## Main Class: Model

The `Model` class (formerly `Manifold`) is the main entry point of the system. It encapsulates processing logic over the Riemannian manifold and integrates all components including embeddings, manifold layers, and readout mechanisms.

### Constructor

```python
import gfn
from gfn.core.config.manifold_configuration import ManifoldConfig, PhysicsConfig

# Using configuration object (recommended)
config = ManifoldConfig(
    vocab_size: int,
    dim: int = 256,
    depth: int = 4,
    rank: int = 32,
    heads: int = 4,
    integrator: str = "yoshida",
    use_scan: bool = False,
    adjoint_rtol: float = 1e-4,
    adjoint_atol: float = 1e-5,
    impulse_scale: float = 0.5,
    physics_config: PhysicsConfig = None
)

model = gfn.create(config)
```

Constructor parameters:
- `vocab_size`: Input vocabulary size
- `dim`: Internal embedding dimension (default: 256)
- `depth`: Number of processing layers (default: 4)
- `rank`: Rank for low-rank metric approximation (default: 32)
- `heads`: Number of attention heads (default: 4)
- `integrator`: Integrator type ("heun", "leapfrog", etc) (default: "yoshida")
- `use_scan`: Whether to use scan-based parallel processing (default: False)
- `adjoint_rtol` / `adjoint_atol`: Tolerances for adjoint integrator solvers.
- `impulse_scale`: The initial embedding velocity momentum magnitude.
- `physics_config`: PhysicsConfig object containing physics-related settings

### Physics Configuration

The physics configuration controls the manifold dynamics and learning behavior:

```python
from gfn.configs import PhysicsConfig

physics_config = PhysicsConfig(
    embedding={
        'type': str,          # 'functional', 'implicit', 'standard'
        'mode': str,          # 'linear', 'binary'
        'coord_dim': int,     # Coordinate dimension
        'impulse_scale': float,
        'omega_0': float
    },
    readout={
        'type': str,          # 'implicit', 'explicit', 'binary'
        'coord_dim': int
    },
    active_inference={
        'enabled': bool,
        'dynamic_time': {'enabled': bool},
        'reactive_curvature': {'enabled': bool, 'plasticity': float},
        'singularities': {'enabled': bool, 'strength': float, 'threshold': float}
    },
    hysteresis={
        'enabled': bool
    },
    fractal={
        'enabled': bool,
        'threshold': float,
        'alpha': float
    },
    topology={
        'type': str  # 'torus', 'euclidean'
    },
    stability={
        'base_dt': float
    },
    cuda_fusion={
        'allow_fused_training': bool
    }
)
```

### Main Methods

### Main Methods

`forward(input_ids, return_state=False)` processes a token sequence over the manifold.

Parameters:
- `input_ids`: Tensor of shape (batch, seq_len) with token indices
- `return_state`: Whether to return the final phase-space state.

Returns:
- If `return_state` is False: `logits` only.
- If `return_state` is True: Tuple `(logits, state)` where `state` is the Tuple `(x, v)`.

*Note: In V2, Christoffel accumulation and physical constraints are handled dynamically by the `PhysicsInformedLoss` wrapper over the forward pass, and are not dumped out directly by the base model's `forward` function.*

`generate(prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None)` performs autoregressive generation.

Parameters:
- `prompt_ids`: Tensor of shape (1, prompt_len) with token indices
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Softmax temperature (lower = more deterministic)
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter

Returns:
- List of generated token IDs

## Module Classes

### FunctionalEmbedding

Neural field-based embeddings using SIREN (Sinusoidal Representation Networks).

```python
from gfn.nn.layers.functional_embedding import FunctionalEmbedding

embedding = FunctionalEmbedding(
    vocab_size: int,
    emb_dim: int,
    coord_dim: int = 16,
    mode: str = 'linear',
    impulse_scale: float = 1.0,
    omega_0: float = 30.0
)
```

Parameters:
- `vocab_size`: Size of vocabulary
- `emb_dim`: Output embedding dimension
- `coord_dim`: Coordinate dimension for neural field
- `mode`: 'linear' or 'binary' coordinate encoding
- `impulse_scale`: Scale factor for impulses
- `omega_0`: SIREN first layer frequency

### ImplicitEmbedding

Learnable coordinate table with projection network.

```python
from gfn.nn.layers.implicit_embedding import ImplicitEmbedding

embedding = ImplicitEmbedding(
    vocab_size: int,
    emb_dim: int,
    coord_dim: int = 16
)
```

### ManifoldLayer

Manifold layer implementing core geodesic dynamics.

```python
from gfn.nn.layers.flow.manifold_layer import ManifoldLayer

layer = ManifoldLayer(
    dim: int,
    heads: int = 4,
    rank: int = 32,
    base_dt: float = 0.4,
    integrator_type: str = 'heun',
    physics_config: dict = None,
    layer_idx: int = 0,
    total_depth: int = 1
)
```

Parameters:
- `dim`: Model dimension
- `heads`: Number of attention heads
- `rank`: Low-rank approximation rank
- `base_dt`: Base integration timestep
- `integrator_type`: Type of integrator
- `physics_config`: Physics configuration dictionary
- `layer_idx`: Index of this layer in the stack
- `total_depth`: Total number of layers

### ParallelModel

Parallel scan-based manifold model for improved efficiency.

```python
from gfn.nn.parallel import ParallelModel

layer = ParallelModel(
    dim: int,
    heads: int = 4,
    physics_config: dict = None
)
```

### FractalMLayer

Fractal manifold layer with hierarchical structure learning.

```python
from gfn.nn.layers.flow.fractal_layer import FractalManifoldLayer

layer = FractalManifoldLayer(
    dim: int,
    heads: int = 4,
    rank: int = 32,
    integrator_type: str = 'heun',
    physics_config: dict = None,
    layer_idx: int = 0,
    total_depth: int = 1
)
```

### ImplicitReadout

Neural field-based readout for manifold state to logits.

```python
from gfn.nn.layers.readout.readout import ImplicitReadout

readout = ImplicitReadout(
    dim: int,
    coord_dim: int = 16,
    topology: int = 1  # 1 for torus, 0 for euclidean
)
```

## Integrator Classes

### HeunIntegrator

Second-order integrator with good accuracy/stability balance.

```python
from gfn.nn.physics.integrators.runge_kutta.heun import HeunIntegrator

integrator = HeunIntegrator(dt: float = 0.4)
```

### RK4Integrator

Fourth-order Runge-Kutta integrator.

```python
from gfn.nn.physics.integrators.runge_kutta.rk4 import RK4Integrator

integrator = RK4Integrator(dt: float = 0.4)
```

### LeapfrogIntegrator

Symplectic leapfrog (Velocity Verlet) integrator.

```python
from gfn.nn.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator

integrator = LeapfrogIntegrator(dt: float = 0.4)
```

### ForestRuthIntegrator

Fourth-order symplectic integrator.

```python
from gfn.nn.physics.integrators.symplectic.forest_ruth import ForestRuthIntegrator

integrator = ForestRuthIntegrator(dt: float = 0.4)
```

## Physics Losses (Functions)

In V2, physics-informed regularizations have been refactored from class-based structures to pure functional components orchestrated by `PhysicsInformedLoss`.

### Hamiltonian / Conservation

Penalty for energy non-conservation.

```python
from gfn.loss.physics.hamiltonian import hamiltonian_loss

loss = hamiltonian_loss(velocities, lambda_h=0.0)
```

### Geodesic / Curvature

Controls geometric singularities.

```python
from gfn.loss.physics.geodesic import geodesic_loss

loss = geodesic_loss(christoffels, lambda_g=0.00005)
```

### Curiosity / Entropy

Encourages phase space exploration.

```python
from gfn.loss.physics.curiosity import curiosity_loss

loss = curiosity_loss(velocities, lambda_c=0.0001)
```

### Noether / Symmetry

Preserves spatial invariant states.

```python
from gfn.loss.physics.noether import noether_loss

loss = noether_loss(christoffels, isomeric_groups=groups, lambda_n=0.01)
```

### PhysicsInformed Wrapper

The composite architecture uses the `PhysicsInformedLoss` class to coordinate these functions alongside the semantic task loss.

## Optimizer Classes

### RiemannianAdam

Custom Riemannian Adam optimizer for manifold parameters.

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(
    params,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-7,
    weight_decay: float = 0.01,
    retraction_mode: str = 'project'
)
```

Parameters:
- `params`: Model parameters
- `lr`: Learning rate
- `betas`: Adam beta coefficients
- `eps`: Epsilon for numerical stability
- `weight_decay`: Weight decay coefficient
- `retraction_mode`: Mode for retraction onto manifold

### ManifoldSGD

SGD optimizer with Riemannian momentum.

```python
from gfn.optim import ManifoldSGD

optimizer = ManifoldSGD(
    params,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.01
)
```

## Configuration Classes

### ManifoldConfig

Main configuration class for Manifold model.

```python
from gfn.core.config.manifold_configuration import ManifoldConfig

config = ManifoldConfig(
    vocab_size: int,
    dim: int = 256,
    depth: int = 4,
    rank: int = 32,
    heads: int = 4,
    integrator: str = 'heun',
    use_scan: bool = False,
    physics_config: PhysicsConfig = None
)
```

### PhysicsConfig

Physics-related configuration.

```python
from gfn.core.config.manifold_configuration import PhysicsConfig

physics_config = PhysicsConfig(
    embedding: dict = None,
    readout: dict = None,
    active_inference: dict = None,
    hysteresis: dict = None,
    fractal: dict = None,
    topology: dict = None,
    stability: dict = None,
    cuda_fusion: dict = None
)
```

### TopologyConfig

Topology configuration.

```python
from gfn.configs import TopologyConfig

topology_config = TopologyConfig(
    type: str = 'torus'  # 'torus', 'euclidean'
)
```

### StabilityConfig

Stability and integration configuration.

```python
from gfn.configs import StabilityConfig

stability_config = StabilityConfig(
    base_dt: float = 0.4
)
```

## Interface Contracts

### Integrator Interface

Every integrator must implement the following interface:

```python
class BaseIntegrator(nn.Module):
    def forward(self, x, v, force, christoffel_fn, **kwargs):
        """
        Perform one integration step.
        
        Args:
            x: Position tensor of shape (..., dim)
            v: Velocity tensor of shape (..., dim)
            force: Force tensor of shape (..., dim)
            christoffel_fn: Callable that computes Christoffel symbols
        
        Returns:
            Tuple (x_new, v_new) with updated state
        """
        raise NotImplementedError
```

### Geometry Interface

Every geometry module must implement:

```python
class BaseGeometry(nn.Module):
    def forward(self, v, x):
        """
        Compute Christoffel symbols.
        
        Args:
            v: Velocity tensor
            x: Position tensor
        
        Returns:
            Christoffel symbols tensor
        """
        pass
    
    def metric(self, x):
        """
        Compute metric tensor.
        
        Args:
            x: Position tensor
        
        Returns:
            Metric tensor of shape (..., dim, dim)
        """
        pass
```

### Loss Interface

Every loss must implement:

```python
class BaseLoss(nn.Module):
    def forward(self, predictions, targets, **kwargs):
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Scalar loss
        """
        pass
```

## Usage Examples

### Basic Model Creation

```python
import gfn
from gfn.core.config.manifold_configuration import ManifoldConfig, PhysicsConfig

# Create physics configuration
physics_config = PhysicsConfig(
    embedding={'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
    readout={'type': 'implicit', 'coord_dim': 16},
    active_inference={'enabled': True},
    topology={'type': 'torus'},
    stability={'base_dt': 0.4}
)

# Create model configuration
config = ManifoldConfig(
    vocab_size=1000,
    dim=256,
    depth=4,
    physics_config=physics_config
)

# Create model
model = gfn.create(config)
```

### Training with Custom Loss

```python
import torch
from gfn.loss.orchestration.physics_informed import PhysicsInformedLoss

loss_fn = PhysicsInformedLoss(
    base_loss_fn=torch.nn.CrossEntropyLoss(),
    config=config.physics_config
)

# Training loop
for batch in dataloader:
    inputs, targets = batch
    
    # Enable Christoffel accumulation context
    with loss_fn.context(model):
        outputs = model(inputs)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
```

### Inference Pipeline

```python
# Load trained model
model = gfn.create(config)
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
prompt = torch.tensor([[1, 2, 3, 4, 5]])
generated = model.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)

print(generated)
```

---

**Manifold Framework Reference**

# Class and API Reference

## Main Class: Manifold

The `Manifold` class is the main entry point of the system. It encapsulates processing logic over the Riemannian manifold.

### Constructor

```python
class Manifold(
    vocab_size: int,
    dim: int,
    depth: int,
    rank: int = 64,
    heads: int = 8,
    integrator_type: str = "leapfrog",
    # New parameters (v2.6.5+)
    hyst_alpha: float = 0.9,
    hyst_beta: float = 0.1,
    enable_trace_normalization: bool = True,
    velocity_friction_scale: float = 0.02,
    position_friction_scale: float = 0.0,
    plasticity_coef: float = 0.02,
    singularity_threshold: float = 0.5,
)
```

Constructor parameters:
- `vocab_size`: Input vocabulary size
- `dim`: Internal embedding dimension
- `depth`: Number of processing layers
- `rank`: Rank for low-rank metric approximation
- `heads`: Number of attention heads
- `integrator_type`: Integrator type ("leapfrog", "yoshida", "forest_ruth", "heun")
- `hyst_alpha`: Hysteresis coefficient for the forget gate
- `hyst_beta`: Hysteresis coefficient for the ghost force
- `enable_trace_normalization`: Enables metric trace normalization
- `velocity_friction_scale`: Velocity-dependent friction scale
- `position_friction_scale`: Position-dependent friction scale
- `plasticity_coef`: Plasticity coefficient for reactive curvature
- `singularity_threshold`: Threshold for singularity activation

### Main Methods

`forward(input_ids, attention_mask=None)` processes a token sequence.

Parameters:
- `input_ids`: Tensor of shape (batch, seq_len) with token indices
- `attention_mask`: Optional tensor of shape (batch, seq_len) with masks

Returns:
- logits of shape (batch, seq_len, vocab_size)

`init_state(batch_size, device)` initializes the system state.

Parameters:
- `batch_size`: Number of sequences in the batch
- `device`: Device for the tensors

Returns:
- Initial system state

`step(state, inputs)` evolves the state one step.

Parameters:
- `state`: Current system state
- `inputs`: Inputs for this step

Returns:
- New evolved state

## Class: ChristoffelLowRank

Implements Christoffel symbol computation with a low-rank approximation.

```python
class ChristoffelLowRank(
    dim: int,
    rank: int = 64,
    curvature_clamp: float = 3.0,
    enable_trace_normalization: bool = True,
)
```

### Attributes

`metric_A`: Low-rank factorization matrix A, shape (dim, rank)

`metric_b`: Factorization vector b, shape (dim,)

`sigma`: Factorization regularization

### Methods

`forward(x)` computes Christoffel symbols.

Parameters:
- `x`: Input of shape (..., dim)

Returns:
- Christoffel symbols of shape (..., dim, dim, dim)

`compute_metric(x)` computes the full metric.

Parameters:
- `x`: Input of shape (..., dim)

Returns:
- Metric of shape (..., dim, dim)

## Class: LeapfrogIntegrator

Implements the Leapfrog integrator with friction.

```python
class LeapfrogIntegrator(
    dim: int,
    dt: float = 0.05,
    substeps: int = 3,
    friction_scale: float = 0.02,
    epsilon: float = 1e-7,
    use_cuda: bool = True,
)
```

### Attributes

`dim`: State dimension

`dt`: Integration timestep

`substeps`: Number of substeps per token step

`friction_scale`: Friction coefficient

### Methods

`forward(q, p, force_fn)` evolves the state.

Parameters:
- `q`: Positions of shape (..., dim)
- `p`: Momenta of shape (..., dim)
- `force_fn`: Function that computes force given (q, p)

Returns:
- Tuple (q_new, p_new) of tensors with the same shape

`backward(q, p, dq, dp)` backward gradient.

Parameters:
- `q`, `p`: Final state
- `dq`, `dp`: Loss gradients with respect to q_new, p_new

Returns:
- Loss gradients with respect to q, p, and force

## Class: HamiltonianLoss

Computes the Hamiltonian loss term.

```python
class HamiltonianLoss(
    lambda_h: float = 0.0,
    reduction: str = "mean",
)
```

### Parameters

`lambda_h`: Weight of the Hamiltonian term

`reduction`: Reduction type ("none", "mean", "sum")

### Methods

`forward(state, initial_energy)` computes the loss.

Parameters:
- `state`: Final system state
- `initial_energy`: Initial energy (Hamiltonian)

Returns:
- Hamiltonian loss

## Class: GeodesicLoss

Computes the geodesic loss term.

```python
class GeodesicLoss(
    lambda_g: float = 0.00005,
    reduction: str = "mean",
)
```

### Parameters

`lambda_g`: Weight of the geodesic term

`reduction`: Reduction type

### Methods

`forward(trajectory)` computes the loss.

Parameters:
- `trajectory`: Trajectory of shape (seq_len, ..., dim)

Returns:
- Scalar geodesic loss

## Class: CombinedLoss

Combines multiple loss terms.

```python
class CombinedLoss(
    lambda_h: float = 0.0,
    lambda_g: float = 0.00005,
    lambda_k: float = 0.0001,
    lambda_n: float = 0.0,
)
```

### Terms

- Standard cross-entropy loss
- Hamiltonian loss (energy conservation)
- Geodesic loss (optimal trajectories)
- Kinetic loss (damping)
- Noether loss (symmetries)

## Class: ReactiveCurvature

Implements reactive curvature with plasticity.

```python
class ReactiveCurvature(
    dim: int,
    plasticity: float = 0.02,
    threshold: float = 0.5,
    curvature_lr: float = 0.01,
)
```

### Methods

`forward(x, curvature)` evolves curvature.

Parameters:
- `x`: Current state
- `curvature`: Current curvature

Returns:
- New adjusted curvature

## Class: HysteresisState

Manages hysteresis state for long-term memory.

```python
class HysteresisState(
    forget_gate_init: float = 0.9,
    state_momentum: float = 0.95,
)
```

### Attributes

`forget_gate`: Forget gate (between 0 and 1)

`hysteresis_state`: Accumulated hysteresis state

### Methods

`update(new_inputs)` updates the state.

`get_memory()` returns the memorized state

## Class: GFNFusion

Implements fusion of GF embeddings with manifold dynamics.

```python
class GFNFusion(
    dim: int,
    embedding_scale: float = 1.5,
    impulse_scale: float = 0.5,
)
```

## Class: RAdam

Custom Riemannian Adam optimizer.

```python
class RAdam(
    params,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.99),
    eps: float = 1e-7,
    weight_decay: float = 0.001,
)
```

## Class: ManifoldSGD

SGD optimizer with Riemannian momentum.

```python
class ManifoldSGD(
    params,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.001,
)
```

## Interface Contracts

### Integrator

Every integrator must implement:

```python
class BaseIntegrator:
    def forward(self, q, p, force_fn):
        # q, p: tensors of shape (..., dim)
        # force_fn: callable(q, p) -> force of shape (..., dim)
        # Returns: (q_new, p_new)
        pass
    
    def backward(self, q, p, dq, dp):
        # dq, dp: loss gradients
        # Returns: gradients of (q, p, force_fn)
        pass
```

### Geometry

Every geometry must implement:

```python
class BaseGeometry:
    def forward(self, x):
        # x: tensor of shape (..., dim)
        # Returns: Christoffel symbols of shape (..., dim, dim, dim)
        pass
    
    def metric(self, x):
        # Returns: metric of shape (..., dim, dim)
        pass
```

### Loss

Every loss must implement:

```python
class BaseLoss:
    def forward(self, predictions, targets):
        # predictions, targets: compatible tensors
        # Returns: scalar loss
        pass
```

---

**Manifold Labs (Joaquín Stürtz)**

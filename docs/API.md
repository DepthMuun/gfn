# Manifold API Reference


**Last Updated:** February 2026  
**API Stability:** Beta (breaking changes possible)

This document provides complete Python API documentation for the Manifold Geometric Flow Network implementation. Manifold is a physically-structured State Space Model that reformulates sequence modeling through the lens of Geometric Mechanics, achieving O(1) inference memory complexity and infinite-horizon generalization capabilities through symplectic integration on learnable Riemannian manifolds.

---

## 1. Installation and Setup

### 1.1 System Requirements

The Manifold framework has the following dependencies that must be satisfied before installation:

```bash
# Core requirements
Python 3.10+
PyTorch 2.3+
NumPy
SciPy

# Optional for GPU acceleration
CUDA 11.8+ (for NVIDIA GPUs)
cuDNN 8.0+

# Optional dependencies
Matplotlib (for visualization)
TQDM (for progress bars)
PyYAML (for configuration files)
```

### 1.2 Installation Methods

The framework can be installed through several methods depending on the use case:

**Basic Installation (PyPI):**

```bash
pip install gfn
```

This installation provides the core functionality with pure PyTorch implementations suitable for development, experimentation, and small-scale deployment.

**Development Installation (Source):**

```bash
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold
pip install -e ".[dev]"
```

The development installation links the package to the local source directory, enabling immediate testing of modifications without reinstallation. The `[dev]` extra installs testing and development dependencies.

**Production Installation with CUDA Acceleration:**

```bash
# For CUDA 12.x (recommended)
cd manifold/gfn/cuda
python compile_cuda_12.9.bat

# For CUDA 11.8
cd manifold/gfn/cuda
python compile_cuda_11.8.bat
```

CUDA kernels provide 10-50x speedup for large-scale training and inference. See `gfn/cuda/README.md` for detailed compilation instructions and troubleshooting.

### 1.3 Verification

After installation, verify that the package is correctly installed and functioning:

```python
import torch
from gfn import Manifold

# Check version
import gfn
print(f"Manifold version: {gfn.__version__}")

# Verify CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create a minimal model for verification
model = Manifold(vocab_size=2, dim=64, depth=2, heads=2)
print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
```

---

## 2. Core Model: Manifold

### 2.1 Class Overview

The `Manifold` class is the primary model interface for the Geometric Flow Network architecture. It implements a physically-structured sequence model where tokens are processed as forces acting on a dynamical system traversing a learnable Riemannian manifold.

```python
from gfn.model import Manifold

model = Manifold(
    vocab_size: int = 50257,
    dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    rank: int = 32,
    integrator_type: str = 'leapfrog',
    use_scan: bool = False,
    physics_config: Optional[Dict] = None,
    impulse_scale: float = 1.0,
    holographic: bool = False
)
```

### 2.2 Constructor Parameters

The following table provides detailed descriptions for all constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | 50257 | Size of the token vocabulary. Determines the dimensionality of output logits and the expected range of input token indices. Standard GPT-2 vocabulary uses 50257 tokens. |
| `dim` | `int` | 256 | Latent manifold dimension (hidden size). All internal representations have this dimension. Larger values provide more capacity but increase computational cost. Typical values range from 128 to 2048. |
| `depth` | `int` | 4 | Number of stacked M-Layers. Each layer applies a geodesic update to the state. Deeper models can represent more complex transformations but require more computation. Typical values range from 2 to 24. |
| `heads` | `int` | 4 | Number of parallel geodesic flow heads. The model dimension must be divisible by the number of heads. More heads provide more parallel geodesic flows but increase parameter count. |
| `rank` | `int` | 32 | Low-rank Christoffel approximation rank. Controls the complexity of the curvature representation. Higher ranks provide more expressive geometry but increase memory and computation. |
| `integrator_type` | `str` | `'leapfrog'` | Numerical integration scheme for geodesic dynamics. Options are documented in Section 2.3. |
| `use_scan` | `bool` | `False` | Enable parallel scan for O(log N) training. When True, uses prefix-sum scan for efficient parallel processing of sequences. Experimental feature. |
| `physics_config` | `Optional[Dict]` | `None` | Physics engine configuration dictionary. See Section 3 for complete configuration reference. When None, uses default configuration. |
| `impulse_scale` | `float` | `1.0` | Scaling factor for token embedding forces. Larger values increase the response to input tokens but may cause instability. Typical range is 10-100. |
| `holographic` | `bool` | `False` | Enable holographic readout mode. When True, uses coordinate-based readout that preserves geometric structure through the output layer. |

### 2.3 Integrator Types

The `integrator_type` parameter selects the numerical integration scheme for geodesic dynamics. The choice of integrator affects accuracy, stability, and computational cost:

**`'leapfrog'` (Default):** Symplectic Velocity Verlet integrator. This is the recommended choice for production use due to its excellent balance of speed and stability. The leapfrog integrator has O(dt³) local error with exact volume preservation, making it ideal for long-sequence tasks. The algorithm alternates between half-steps in velocity and full steps in position, providing the symplectic structure that ensures gradient stability.

**`'heun'`:** Second-order Runge-Kutta method. This integrator provides a good balance between accuracy and computational cost. It is suitable for debugging and initial model development when simplicity is valued over maximum performance. The Heun method is not symplectic, so it does not preserve phase-space volume exactly.

**`'rk4'`:** Fourth-order Runge-Kutta method. This integrator provides the highest accuracy among the standard options but is computationally expensive. Note: The RK4 integrator may diverge on tasks with non-smooth dynamics, such as parity with singularities enabled. Use with caution on complex logical operations.

**`'forest_ruth'`:** Fourth-order symplectic integrator. This integrator provides excellent energy conservation for high-precision reasoning tasks. It is slower than leapfrog but offers superior stability for complex logical operations that require precise integration over many timesteps.

**`'yoshida'`:** Fourth-order symplectic integrator. This is an alternative high-order scheme with different coefficients than Forest-Ruth. The choice between them depends on specific task characteristics.

**`'omelyan'`:** Optimized fourth-order symplectic integrator. This integrator provides a good compromise between Forest-Ruth and Yoshida, offering the benefits of fourth-order accuracy with optimized coefficients for performance.

**`'verlet'`:** Basic Velocity Verlet without symplectic optimization. This is a simple and fast integrator that does not include the full symplectic structure. It is primarily useful for ablation studies and baselines.

### 2.4 Forward Pass

#### Standard Forward Pass

The standard forward pass processes a batch of sequences in parallel:

```python
# Input: token IDs [batch_size, seq_len]
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Forward pass
logits, state, christoffels = model(input_ids)

# Returns:
#   logits:      [batch_size, seq_len, vocab_size] - prediction logits
#   state:       Tuple (x_final, v_final) - final position and velocity tensors
#   christoffels: List of curvature tensors per layer (for analysis)
```

**Return Values:**

The forward pass returns a tuple containing three elements. The `logits` tensor has shape `[batch_size, seq_len, vocab_size]` and contains the output predictions for each position in the sequence. The `state` is a tuple of tensors `(x_final, v_final)` representing the final position and velocity of the dynamical system after processing the entire sequence. The `christoffels` is a list of curvature tensors, one per layer, which can be used for analysis and visualization.

#### Autoregressive Forward with State

For autoregressive generation, maintain state across tokens to achieve O(1) memory inference:

```python
# Initialize state
state = None

# Process sequence token by token
for t in range(sequence_length):
    logits, state, christoffels = model(input_ids[:, t:t+1], state=state)
    # state is passed to next step, enabling O(1) memory inference

# Returns:
#   logits:      [batch_size, 1, vocab_size] - single token predictions
#   state:       Tuple (x, v) - updated dynamical state
#   christoffels: Curvature tensor for current step
```

The state persistence mechanism is the key to Manifold's O(1) inference memory. Instead of storing the entire sequence history, the system maintains a constant-size state `(x, v)` that encodes context implicitly through geometric and momentum-based representations.

### 2.5 Generation Method

The `generate` method provides convenient autoregressive text generation:

```python
model.eval()

prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Token IDs
generated = model.generate(
    prompt_ids=prompt,
    max_new_tokens=100,
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    do_sample: bool = True,
    device: str = 'cuda'
)
```

**Generation Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `torch.Tensor` | Required | Input token IDs of shape `[batch_size, prompt_len]`. |
| `max_new_tokens` | `int` | Required | Number of new tokens to generate. |
| `temperature` | `float` | 1.0 | Softmax temperature. Lower values produce more deterministic output. Typical range is 0.1 to 1.5. |
| `top_k` | `Optional[int]` | 40 | Top-k sampling filter. Filters logits to keep only the k most likely tokens at each step. Set to None to disable. |
| `top_p` | `Optional[float]` | 0.9 | Nucleus sampling threshold. Filters logits to keep tokens whose cumulative probability exceeds p. Set to None to disable. |
| `do_sample` | `bool` | `True` | If True, uses sampling for token selection. If False, uses greedy decoding (always select most likely token). |
| `device` | `str` | `'cuda'` | Device for computation. Options are `'cuda'` or `'cpu'`. |

---

## 3. Physics Configuration

### 3.1 Configuration Structure

The `physics_config` parameter is a dictionary that controls the physics engine behavior. This section provides the complete reference for all configuration options:

```python
physics_config = {
    'embedding': {...},           # Embedding layer configuration
    'readout': {...},             # Readout layer configuration
    'active_inference': {...},    # Active inference configuration
    'fractal': {...},             # Fractal manifold configuration
    'topology': {...},            # Manifold topology configuration
    'stability': {...}            # Stability parameters
}
```

### 3.2 Embedding Configuration

The embedding configuration controls how input tokens are transformed into manifold coordinates:

```python
embedding_config = {
    'type': 'functional',     # 'functional', 'implicit', or 'standard'
    'mode': 'linear',         # 'linear' or 'binary' (functional type only)
    'coord_dim': 16           # Coordinate dimension for neural field
}
```

**Functional Embedding (Recommended):** The functional embedding uses a neural field (SIREN MLP) to map token coordinates to embedding vectors. This approach achieves O(1) memory scaling with vocabulary size, meaning the number of parameters remains constant regardless of how many tokens exist in the vocabulary.

**Key Advantages of Functional Embedding:**

- O(1) parameters regardless of vocabulary size, enabling infinite vocabulary scaling
- Smooth interpolation between token representations
- Enables generalization to unseen tokens through coordinate continuity
- The linear mode provides superior out-of-distribution generalization compared to binary mode

**Embedding Modes Comparison:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `'linear'` | Linear coordinate mapping with smooth interpolation between token representations | Recommended - provides superior generalization to unseen tokens and sequences |
| `'binary'` | Discrete binary coordinate representation using bit patterns | Legacy mode with inferior generalization; included for backward compatibility |

### 3.3 Readout Configuration

The readout configuration controls how the final manifold state is transformed to output predictions:

```python
readout_config = {
    'type': 'implicit',       # 'implicit' or 'explicit'
    'coord_dim': 16           # Coordinate dimension for inverse mapping
}
```

**Implicit Readout (Recommended):** The implicit readout uses an inverse neural field to map manifold coordinates directly to token logits. This approach provides holographic alignment where the latent state IS the answer, enabling perfect coordinate-to-token mapping without additional classifier weights.

**Advantages of Implicit Readout:**

- Holographic alignment: the latent state encodes predictions directly
- No additional classifier weights needed beyond the manifold representation
- Perfect coordinate-to-token mapping preserves geometric structure
- Enables principled uncertainty estimation from manifold geometry

### 3.4 Active Inference Configuration

Active inference enables the manifold to adapt its dynamics based on uncertainty estimates:

```python
active_inference_config = {
    'enabled': True,
    'dynamic_time': {
        'enabled': True
    },
    'reactive_curvature': {
        'enabled': True,
        'plasticity': 0.2
    },
    'singularities': {
        'enabled': True,
        'strength': 20.0,
        'threshold': 0.8
    }
}
```

**Reactive Curvature:** The plasticity scalar modulates curvature based on kinetic energy according to the formula:

$$\lambda(K) = \alpha \cdot \tanh(K)$$

$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot (1 + \lambda(K))$$

When the model experiences high kinetic energy (indicating confusion or uncertainty), the curvature increases, creating a more viscous manifold that slows processing and allows for careful integration of information.

**Logical Singularities:** Singularities represent discrete logical decisions as topological attractors. They enable the model to make sharp transitions between states, which is essential for tasks like parity computation where the output can change abruptly:

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

The singularity strength controls the intensity of the attraction, while the threshold determines the curvature level required to trigger singularity formation.

### 3.5 Fractal Configuration

Fractal manifold resolution enables adaptive precision for high-curvature regions:

```python
fractal_config = {
    'enabled': True,
    'threshold': 0.5,
    'alpha': 0.2
}
```

When local curvature $\mathcal{R}$ exceeds the threshold, the manifold recursively opens sub-manifolds with finer temporal resolution ($dt' \ll dt$). This adaptive resolution enables the model to handle regions of high geometric complexity without sacrificing efficiency in smoother regions.

### 3.6 Stability Configuration

```python
stability_config = {
    'base_dt': 0.4,            # Base integration timestep
    'curvature_clamp': 5.0     # Maximum |Γ| for numerical stability
}
```

**Timestep Guidelines:**

| dt Range | Behavior |
|----------|----------|
| 0.1-0.2 | Over-cautious, slow learning with excessive computation |
| 0.3-0.4 | Optimal - balanced stability and speed for most tasks |
| 0.5-0.6 | Borderline unstable - may work with careful initialization |
| >0.7 | Complete divergence - not recommended |

### 3.7 Complete Configuration Example

The following configuration reflects the optimal settings validated in the superiority benchmark:

```python
physics_config = {
    'embedding': {
        'type': 'functional',     # Neural field embedding (O(1) vocabulary scaling)
        'mode': 'linear',          # 'linear' or 'binary' - linear is superior
        'coord_dim': 16            # Coordinate dimension for neural field
    },
    'readout': {
        'type': 'implicit',        # Implicit neural field readout
        'coord_dim': 16            # Coordinate dimension for inverse mapping
    },
    'active_inference': {
        'enabled': True,           # Enable active inference dynamics
        'dynamic_time': {
            'enabled': True        # Adaptive timestep based on uncertainty
        },
        'reactive_curvature': {
            'enabled': True,       # Curvature modulation by kinetic energy
            'plasticity': 0.2      # Plasticity coefficient α for λ(K) = α·tanh(K)
        },
        'singularities': {
            'enabled': True,       # Enable logical singularities
            'strength': 20.0,      # Singularity attraction strength
            'threshold': 0.8       # Activation threshold for singularity detection
        }
    },
    'fractal': {
        'enabled': True,           # Enable fractal manifold resolution
        'threshold': 0.5,          # Curvature threshold for sub-manifold opening
        'alpha': 0.2               # Recursive resolution parameter
    },
    'topology': {
        'type': 'torus'            # Toroidal topology for cyclic logic
    },
    'stability': {
        'base_dt': 0.4,            # Base integration timestep
        'curvature_clamp': 5.0     # Maximum curvature magnitude
    }
}

model = Manifold(
    vocab_size=2,
    dim=128,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=physics_config,
    impulse_scale=80.0,
    holographic=True
).to(device)
```

---

## 4. Geometry Module

### 4.1 LowRankChristoffel

The `LowRankChristoffel` class computes curvature (Christoffel symbols) using a low-rank approximation for memory efficiency:

```python
from gfn.geometry import LowRankChristoffel

christoffel = LowRankChristoffel(
    dim=512,
    rank=32,
    physics_config=None
)

# Compute curvature: gamma = Γ(v, x)
# Input:  v [batch, dim] - velocity
#         x [batch, dim] - position
# Output: gamma [batch, dim] - Christoffel symbols
gamma = christoffel(v, x)
```

**Key Features:**

- **Adaptive Gating:** Learnable gate controls when curvature is applied, enabling the model to selectively use geometric information
- **Dynamic Modulation:** Position-dependent scaling of curvature allows different geometric regions to have different interaction patterns
- **Saturation:** Built-in clamping for numerical stability prevents curvature explosion in edge cases

### 4.2 ToroidalChristoffel

Toroidal-specific Christoffel symbols for cyclic logic tasks:

```python
from gfn.geometry import ToroidalChristoffel

christoffel = ToroidalChristoffel(
    dim=512,
    rank=32,
    major_radius=1.0,
    minor_radius=0.5,
    physics_config=None
)
```

**Advantages for Parity Tasks:**

- Natural encoding of modular arithmetic (mod 2 → mod $2\pi$) leverages the topology of the torus
- Phase space topology matches cyclic logical structure, enabling efficient representation of periodic computations
- Boundary conditions on the torus naturally handle wraparound behavior in cumulative operations

### 4.3 ReactiveChristoffel

Curvature modulated by kinetic energy for active inference:

```python
from gfn.geometry import ReactiveChristoffel

christoffel = ReactiveChristoffel(
    dim=512,
    rank=32,
    plasticity=0.2
)
```

The reactive Christoffel symbols automatically adjust curvature based on the kinetic energy of the system, providing adaptive geometric responses to processing demands.

### 4.4 HyperChristoffel

Hyperbolic Christoffel symbols for hierarchical representations:

```python
from gfn.geometry import HyperChristoffel

christoffel = HyperChristoffel(
    dim=512,
    rank=32,
    curvature_scale=1.0
)
```

Hyperbolic geometry is particularly suitable for representing hierarchical and tree-like structures due to the exponential growth of hyperbolic space.

---

## 5. Symplectic Integrators

### 5.1 Leapfrog Integrator (Default)

The leapfrog integrator implements Velocity Verlet integration with friction:

```python
from gfn.integrators import LeapfrogIntegrator

integrator = LeapfrogIntegrator(
    christoffel=christoffel,
    dt=0.4
)

# One integration step
x_next, v_next = integrator(x, v, force=F, dt_scale=1.0)
```

**Algorithm (Velocity Verlet with friction):**

```python
# Friction term: μ(x, u) · v
# where μ = sigmoid(gate_activ) · 5.0

a_t = F - Γ(v_t, x_t) - friction(v_t, x_t)
v_half = v_t + 0.5 * dt * a_t
x_{t+1} = x_t + dt * v_half
a_{t+1} = F - Γ(v_half, x_{t+1}) - friction(v_half, x_{t+1})
v_{t+1} = v_half + 0.5 * dt * a_{t+1}
v_{t+1} = v_{t+1} / (||v_{t+1}|| + ε)  # Velocity normalization
```

### 5.2 Forest-Ruth Integrator (High Precision)

The Forest-Ruth integrator provides fourth-order symplectic integration with superior energy conservation:

```python
from gfn.integrators import ForestRuthIntegrator

integrator = ForestRuthIntegrator(
    christoffel=christoffel,
    dt=0.4
)
```

This integrator is recommended for high-precision reasoning tasks where accurate integration over many timesteps is critical.

### 5.3 Available Integrators Reference

| Integrator | Order | Symplectic | Use Case |
|------------|-------|------------|----------|
| `LeapfrogIntegrator` | 2nd | Yes | Default - general purpose with excellent stability |
| `ForestRuthIntegrator` | 4th | Yes | High-precision reasoning with superior energy conservation |
| `YoshidaIntegrator` | 4th | Yes | Alternative high-precision symplectic scheme |
| `OmelyanIntegrator` | 4th | Yes | Optimized fourth-order symplectic integrator |
| `VerletIntegrator` | 2nd | Yes | Simple symplectic integrator for baselines |
| `HeunIntegrator` | 2nd | No | Debugging and initial training |
| `RK4Integrator` | 4th | No | High accuracy but may diverge on complex tasks |
| `EulerIntegrator` | 1st | No | Baselines only - not recommended for production |

---

## 6. Layers

### 6.1 MLayer

The MLayer is the core Manifold layer that replaces Transformer attention:

```python
from gfn.layers import MLayer

layer = MLayer(
    dim=512,
    heads=8,
    rank=32,
    integrator_type='leapfrog',
    physics_config=None
)

# Forward pass
x_next, v_next, context, christoffels = layer(x, v, force, context)
```

**Multi-Head Processing:**

The MLayer splits the state into multiple independent geodesic flows. Each head has independent Christoffel symbols and integrator, allowing different heads to learn different geometric structures. The outputs of all heads are mixed via a learned projection to produce the final layer output.

### 6.2 ParallelMLayer

The ParallelMLayer provides a parallel scan variant for O(log N) training:

```python
from gfn.layers import ParallelMLayer

layer = ParallelMLayer(
    dim=512,
    heads=8,
    rank=32
)

# Process entire sequence in parallel
x_out, v_out, ctx, christoffels = layer(
    None, None, force=force_sequence
)
```

### 6.3 GatingLayer

The GatingLayer implements thermodynamic gating for switching between memory and computation:

```python
from gfn.layers import GatingLayer

gating = GatingLayer(
    dim=512,
    gate_type='thermodynamic'
)

# Learnable friction coefficient
mu = gating(x, v, temperature=1.0)
# mu → 0: Superfluid phase (memory preservation)
# mu → 1: Dissipative phase (computation/update)
```

---

## 7. Loss Functions

### 7.1 ToroidalDistanceLoss

Specialized loss for toroidal manifolds with cyclic coordinates:

```python
from gfn.losses import ToroidalDistanceLoss

criterion = ToroidalDistanceLoss()

# Compute loss on circular coordinates
loss = criterion(predictions, targets)
# Handles π/2 offset for binary classification
```

### 7.2 Geodesic Regularization

Encourages particles to follow geodesic paths:

```python
from gfn.losses import geodesic_regularization

loss_geo = geodesic_regularization(
    christoffels,
    lambda_g=0.001
)
```

### 7.3 Hamiltonian Loss

Energy conservation regularization:

```python
from gfn.losses import hamiltonian_loss

loss_ham = hamiltonian_loss(
    v_sequence,
    states=x_sequence,
    metric_fn=metric_fn,
    lambda_h=0.0,
    forces=forces
)
```

---

## 8. Optimization

### 8.1 RiemannianAdam (Required)

Standard Adam optimization causes Euclidean drift on manifolds. The RiemannianAdam optimizer includes retraction operations to maintain geometric validity:

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    retraction='normalize',
    max_norm=10.0
)

# Standard PyTorch usage
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Retraction Types:**

| Type | Description |
|------|-------------|
| `'normalize'` | Clip gradient norm to max_norm - recommended for most applications |
| `'cayley'` | Orthogonal projection using Cayley transform - requires square matrices |
| `'euclidean'` | No retraction - unstable and not recommended |

### 8.2 Recommended Training Configuration

```python
# Optimizer groups with differential learning rates
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    total_steps=max_steps,
    pct_start=0.2
)

# Gradient clipping (critical for stability)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 9. Utilities

### 9.1 Memory Measurement

```python
from tests.benchmarks.bench_utils import PerformanceStats

def forward_fn():
    return model(x)

peak_mb = PerformanceStats.measure_peak_memory(model, forward_fn)
```

### 9.2 Configuration Loading

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = Manifold(**config['model'])
```

### 9.3 Model Checkpointing

```python
# Save
torch.save({
    'model_state': model.state_dict(),
    'config': model.config,
    'physics_config': model.physics_config
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state'])
```

---

## 10. Complete Training Example

```python
import torch
import torch.nn as nn
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimal configuration (from superiority benchmark)
physics_config = {
    'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
    'readout': {'type': 'implicit', 'coord_dim': 16},
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

# Model instantiation
model = Manifold(
    vocab_size=2,
    dim=128,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=physics_config,
    impulse_scale=80.0,
    holographic=True
).to(device)

# Optimizer with differential learning rates
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, total_steps=1000, pct_start=0.2
)

criterion = ToroidalDistanceLoss()

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(inputs, collect_christoffels=False)
    x_pred = output[0]
    
    # Compute loss
    loss_val = criterion(x_pred, targets.float().unsqueeze(-1).expand_as(x_pred))
    
    # Physics losses (if collecting Christoffel data)
    if isinstance(output, tuple) and len(output) >= 6:
        christoffels = output[2]
        if christoffels:
            loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)
            loss_val = loss_val + loss_phy
    
    # Backward pass
    total_loss = loss_val
    if not torch.isnan(total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Logging
    if step % 50 == 0:
        accuracy = compute_accuracy(x_pred, targets)
        print(f"Step {step}: Loss = {loss_val.item():.4f}, Acc = {accuracy:.2%}")
```

---

## 11. Troubleshooting Guide

### 11.1 Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Training loss oscillates chaotically | Using standard Adam optimizer | Use `RiemannianAdam` with appropriate retraction |
| Model diverges after ~100 steps | Missing velocity normalization | Enable velocity normalization in stability config |
| Out of memory | Excessive batch size | Reduce batch size or use gradient accumulation |
| Slow training | Sequential Christoffel computation | Enable CUDA kernels for fused operations |
| Poor generalization | Using binary embedding mode | Use `mode='linear'` for superior generalization |
| Low accuracy | Incorrect readout type | Use `type='implicit'` for best performance |

### 11.2 Gradient Issues

If gradients become NaN during training:

```python
# Enable stricter gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Check for NaN in inputs
assert not torch.isnan(inputs).any()

# Enable numerical stability checks
model.check_numerical_stability()
```

### 11.3 Numerical Instability

For persistent numerical instability:

```python
# Enable stability checks
model.check_numerical_stability()

# Reduce integration timestep
physics_config['stability']['base_dt'] = 0.3

# Increase curvature clamping
physics_config['stability']['curvature_clamp'] = 3.0

# Reduce impulse scale
model.impulse_scale = 50.0
```

---

## 12. API Changelog

### Version 2.6.4 (Current)

- Enhanced CUDA kernel performance for leapfrog integration
- Added support for parallel scan in training mode
- Improved numerical stability for reactive curvature
- Updated documentation with comprehensive examples

### Version 2.6.2

- Fixed CUDA kernel saturation terms
- Added Forest-Ruth integrator support
- Improved active inference documentation
- Updated optimal configuration to use `linear` embedding mode

### Version 2.6.0

- Added Dynamic Forget Gate (Thermodynamic Friction)
- Updated M-Layer with friction term
- Verified 100K token generalization
- Added Parallel Scan to CUDA kernels

### Version 2.5.0

- Initial production release
- Functional embeddings
- Binary readout mode
- Multi-head architecture

---

**Documentation Version:** 2.6.4  
**API Stability:** Beta (breaking changes possible)  
**License:** Apache 2.0

For theoretical foundations, see [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md).  
For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).  
For physics derivations, see [PHYSICS.md](PHYSICS.md).

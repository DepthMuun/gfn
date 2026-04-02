# GSSM Code Analysis

Complete analysis of the GSSM (Geodesic State Space Model) codebase.

---

## 1. Architecture Overview

```
gfn.create('gssm', ...)
    ↓
ModelFactory.build()
    ├── EmbeddingBuilder → FunctionalEmbedding
    ├── LayerBuilder → ManifoldLayer × depth
    │       ├── Integrator (Leapfrog/Yoshida/etc)
    │       ├── PhysicsEngine → Geometry
    │       ├── Mixer (FlowMixer)
    │       └── Plugins (dynamic_time, fractal)
    └── ReadoutBuilder → CategoricalReadout
```

---

## 2. Models Module (`gfn/realizations/gssm/models/`)

### 2.1 BaseModel (`base.py`)

**Purpose**: Core evolution engine - orchestrates the forward pass through all layers.

**Key Methods**:
- `forward(input_ids, state, force_manual)` → `(logits, (x, v), info)`
- `_evolve_sequence(x, v, forces, mask)` → Internal loop through layers

**Data Flow**:
```
1. Resolve Forces: input_ids → embedding → forces
2. Initialize State: x0, v0 (or from state parameter)
3. For each timestep in sequence:
    a. For each layer in depth:
        - layer(x, v, force) → x', v'
    b. Collect logits from readout
4. Return: logits, (x_final, v_final), state_info
```

**Key Attributes**:
- `self.layers`: ModuleList of ManifoldLayer
- `self.embedding`: FunctionalEmbedding
- `self.x0, self.v0`: Initial position/velocity parameters
- `self.hooks`: HookManager for lifecycle events

### 2.2 ManifoldModel (`manifold.py`)

**Purpose**: Concrete implementation, just wraps BaseModel.

```python
@register_model('manifold')
class ManifoldModel(BaseModel):
    # Inherits all from BaseModel
    pass
```

### 2.3 ManifoldLayer (`manifold_layer.py`)

**Purpose**: Single layer of manifold evolution.

**Components**:
- `integrator`: Numerical solver (Leapfrog, Yoshida, etc.)
- `mixer`: FlowMixer for head interaction
- `dynamics_x, dynamics_v`: State update routing
- `norm_x, norm_v`: Manifold normalization
- `plugins`: ModuleDict of enabled plugins

**Forward Pass**:
```
1. Reshape: [B, S, H, D] → [B*S, H, D]
2. Pre-integrate hooks (dynamic_time adjusts dt)
3. Integrator.step(x, v, force, dt) → x_stepped, v_stepped
4. Mixer(x_stepped, v_stepped) → mixed state
5. Dynamics(mixed state) → x_next, v_next
6. Topology resolution (wrap to manifold bounds)
7. Reshape back: [B*S, H, D] → [B, S, H, D]
```

**Plugins**:
- `dynamic_time`: Adaptive timestep per head
- `fractal`: Sub-manifold refinement for high curvature

### 2.4 Factory (`factory.py`)

**Purpose**: Constructs complete model from config.

**Build Process**:
```
1. EmbeddingBuilder → FunctionalEmbedding
2. LayerBuilder → ManifoldLayer × depth
   - Creates Geometry from config
   - Creates PhysicsEngine with Geometry
   - Creates Integrator with PhysicsEngine
   - Creates Mixer
3. ReadoutBuilder → CategoricalReadout
4. Assemble: ManifoldModel(layers, embedding, x0, v0)
```

---

## 3. Physics Module (`gfn/realizations/gssm/physics/`)

### 3.1 PhysicsEngine (`engine.py`)

**Purpose**: Computes net acceleration from all forces.

**Formula**:
```
dv/dt = -Γ(x,v) + F_ext + F_friction + F_ghost + F_stochastic + F_curiosity
```

**Components**:
- `geometry`: Computes Christoffel symbols Γ(x,v)
- `singularity_gate`: Handles geometric singularities
- `hysteresis`: Ghost force for memory
- `stochasticity_module`: Brownian/OU noise
- `curiosity_module`: Exploration drive

**Key Method**: `compute_acceleration(x, v, force, dt)`

```
1. Call geometry(x, v, force) → Γ (Christoffel)
2. Get friction coefficient μ from geometry or config
3. friction_term = μ * v
4. net_accel = -Γ - friction_term + force
5. Add ghost force (if hysteresis enabled)
6. Add stochastic force (if enabled)
7. Add curiosity force (if enabled)
8. Apply singularity damping (if enabled)
9. Return net_accel
```

### 3.2 Integrators (`integrators/`)

**Purpose**: Solves the ODE to get next (x, v) from current state.

#### Symplectic (Energy-preserving)
| Integrator | Order | File |
|------------|-------|------|
| `LeapfrogIntegrator` | 2nd | `symplectic/leapfrog.py` |
| `VerletIntegrator` | 2nd | `symplectic/verlet.py` |
| `YoshidaIntegrator` | 4th | `symplectic/yoshida.py` |
| `ForestRuthIntegrator` | 4th | `symplectic/forest_ruth.py` |
| `OmelyanIntegrator` | 2nd | `symplectic/omelyan.py` |

#### Runge-Kutta (Accuracy-focused)
| Integrator | Order | File |
|------------|-------|------|
| `RK4Integrator` | 4th | `runge_kutta/rk4.py` |
| `HeunIntegrator` | 2nd | `runge_kutta/heun.py` |

#### Adaptive
| Integrator | Description |
|------------|-------------|
| `AdaptiveIntegrator` | Auto-adjusts dt based on error |

**Leapfrog Algorithm** (default):
```
# Half step velocity
v_half = v + 0.5 * dt * a(x, v, force)

# Full step position
x_new = x + dt * v_half

# Compute new acceleration
a_new = compute_acceleration(x_new, v_half, force)

# Full step velocity
v_new = v_half + 0.5 * dt * a_new
```

### 3.3 Dynamics (`dynamics/`)

**Purpose**: Routes the mixed state to next state.

| Type | Description |
|------|-------------|
| `direct` | x_next = x_stepped |
| `residual` | x_next = x + (x_stepped - x) |
| `gated` | x_next = gate * x_stepped + (1-gate) * x |
| `mix` | Combines multiple dynamics |

---

## 4. Geometry Module (`gfn/realizations/gssm/geometry/`)

### 4.1 Base Classes

**Geometry Protocol** (`interfaces/geometry.py`):
```python
class Geometry(Protocol):
    def __call__(self, v, x, force=None) -> torch.Tensor | Tuple[torch.Tensor, float]:
        """Compute Christoffel symbols Γ(x,v)"""
        
    def metric(self, x) -> torch.Tensor:
        """Compute metric tensor g_ij"""
        
    def project(self, x) -> torch.Tensor:
        """Project to manifold"""
        
    def dist(self, x1, x2) -> torch.Tensor:
        """Geodesic distance"""
```

### 4.2 Geometry Implementations

| Geometry | File | Description |
|----------|------|-------------|
| `ToroidalRiemannianGeometry` | `torus.py` | Toroidal manifold S¹×S¹... |
| `EuclideanGeometry` | `euclidean.py` | Flat space |
| `LowRankRiemannianGeometry` | `low_rank.py` | Low-rank approximation |
| `ReactiveRiemannianGeometry` | `reactive.py` | Adaptive curvature |
| `AdaptiveGeometry` | `adaptive.py` | Self-adjusting |
| `HyperbolicGeometry` | `hyperbolic.py` | Poincaré ball |
| `HolographicGeometry` | `holographic.py` | Interference patterns |
| `HierarchicalGeometry` | `hierarchical.py` | Multi-scale |
| `SphericalGeometry` | `spherical.py` | Sphere S^n |

### 4.3 Torus Geometry (`torus.py`)

**Purpose**: Default geometry for GSSM.

**Key Method**: `connection(v, w, x)` → Christoffel symbols

```python
def connection(self, v, w, x):
    # For each (θ, φ) pair:
    # Γ_θ = (R+r*cos(θ)) * sin(θ) / r * v_φ * w_φ
    # Γ_φ = -r*sin(θ) / (R+r*cos(θ)) * (v_φ*w_θ + v_θ*w_φ)
    
    # Returns: acceleration due to curvature
    return gamma  # shape: [B, H, D]
```

**Also returns**: friction coefficient μ based on curvature.

### 4.4 Low-Rank Geometry (`low_rank.py`)

**Purpose**: Efficient approximation for large dimensions.

**Key Idea**: Decompose Christoffel as:
```
Γ ≈ Σ_r W_rk * (U_ir * U_jr)
```
Reduces O(D³) to O(Rank² × D).

---

## 5. Configuration (`gfn/realizations/gssm/config/`)

### 5.1 Schema (`schema.py`)

**Main Config Classes**:

```python
@dataclass
class TopologyConfig:
    type: str = 'torus'      # Topology type
    R: float = 2.0           # Major radius
    r: float = 1.0           # Minor radius
    learnable_r: bool = True # Make r learnable

@dataclass
class StabilityConfig:
    base_dt: float = 0.1     # Time step
    adaptive: bool = True    # Adaptive dt
    friction: float = 0.01   # Base friction
    integrator_type: str = 'leapfrog'
    enable_trace_normalization: bool = True
    velocity_saturation: float = 0.0

@dataclass
class PhysicsConfig:
    topology: TopologyConfig
    stability: StabilityConfig
    dynamics: DynamicsConfig
    active_inference: ActiveInferenceConfig
    hysteresis: HysteresisConfig
    fractal: FractalConfig
    singularities: SingularityConfig
    embedding: EmbeddingConfig

@dataclass
class ManifoldConfig:
    vocab_size: int
    dim: int = 64
    heads: int = 4
    depth: int = 4
    rank: int = 16
    initial_spread: float = 0.1
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
```

### 5.2 Defaults (`defaults.py`)

Centralized default values:
- `MODEL_DEFAULTS`: dim, heads, depth, rank, vocab_size, initial_spread
- `PHYSICS_DEFAULTS`: topology_type, base_dt, friction, etc.
- `TRAINING_DEFAULTS`: lr, optimizer, weight_decay, etc.

### 5.3 Constants (`constants.py`)

```python
DEFAULT_DT = 0.1
DEFAULT_FRICTION = 0.01
DEFAULT_PLASTICITY = 0.05
MAX_VELOCITY = 10.0
CURVATURE_CLAMP = 1e5
SINGULARITY_THRESHOLD = 1e-4
BLACK_HOLE_STRENGTH = 0.1
EPSILON_STANDARD = 1e-8
TOPOLOGY_TORUS = 'torus'
TOPOLOGY_EUCLIDEAN = 'euclidean'
```

---

## 6. Embeddings (`models/components/embedding.py`)

### FunctionalEmbedding

**Purpose**: Maps token IDs to manifold impulses (forces).

```python
class FunctionalEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, coord_dim=16, mode='linear', 
                 impulse_scale=1.0, omega_0=30.0):
        # Creates learnable coordinates for each token
        self.coords = nn.Embedding(vocab_size, coord_dim)
        # Projects to full dimension
        self.proj = nn.Linear(coord_dim, dim)
        
    def forward(self, token_ids):
        # coords: [B, S] → [B, S, coord_dim]
        # proj: [B, S, coord_dim] → [B, S, dim]
        return self.proj(self.coords(token_ids))
```

**Modes**:
- `linear`: Direct projection
- `bilinear`: Uses coordinates as basis

---

## 7. Readout (`models/components/readout.py`)

### CategoricalReadout

**Purpose**: Projects manifold state to vocabulary logits.

```python
class CategoricalReadout(nn.Module):
    def __init__(self, dim_total, vocab_size, topology='torus'):
        # If torus: use sin/cos encoding
        # If euclidean: direct projection
        self.proj = nn.Linear(dim_total, vocab_size)
        
    def forward(self, x_final):
        # x_final: [B, S, H, D]
        # If torus: x → [sin(x), cos(x)]
        # Project to vocab
        return self.proj(encoded)
```

---

## 8. Losses (`gfn/realizations/gssm/losses/`)

### Loss Types

| Loss | File | Purpose |
|------|------|---------|
| `ManifoldGenerativeLoss` | `generative.py` | Standard NLL for token prediction |
| `ToroidalDistanceLoss` | `toroidal.py` | Angular distance for periodic data |
| `PhysicsInformedLoss` | `physics.py` | NLL + physics terms |

### PhysicsInformedLoss Components

```
L_total = λ_NLL * L_NLL + λ_geo * L_geodesic + λ_ham * L_hamiltonian + λ_kin * L_kinetic
```

- **L_geodesic**: Penalizes high curvature paths
- **L_hamiltonian**: Penalizes energy drift
- **L_kinetic**: Prevents velocity explosion

---

## 9. Hooks System (`models/hooks.py`)

**Purpose**: Plugin system for injecting logic without modifying core.

**Available Hooks**:
- `on_batch_start`: Before processing batch
- `on_batch_end`: After batch processing
- `on_timestep_start`: Before each sequence position
- `on_timestep_end`: After each sequence position
- `on_layer_start`: Before each layer
- `on_layer_end`: After each layer
- `state_init`: Custom state initialization
- `wrap_evolution`: Wrap entire evolution function

---

## 10. Complete Forward Pass

```
Input: token_ids [B, S]

1. Embedding(token_ids) → forces [B, S, D]

2. Initialize: x = x0.expand(B, ...), v = v0.expand(B, ...)

3. For each timestep t in S:
   For each layer l in depth:
      a. PhysicsEngine.compute_acceleration(x, v, force)
         → acceleration = -Christoffel - friction*v + force
      
      b. Integrator.step(x, v, acceleration, dt)
         → x', v'
      
      c. Mixer(x', v') → mixed state
      
      d. Dynamics(mixed state) → x_next, v_next
      
      e. Project to manifold (wrap to [-π, π] for torus)
   
   f. Readout(x_final) → logits [B, S, V]

4. Return: logits, (x_final, v_final), info dict
```

---

## 11. Key Parameters Summary

| Parameter | Where | Effect |
|-----------|-------|--------|
| `base_dt` | StabilityConfig | Integration timestep - smaller = more stable |
| `friction` | StabilityConfig | Velocity damping - higher = more damping |
| `integrator_type` | StabilityConfig | Numerical method - leapfrog = most stable |
| `topology.type` | TopologyConfig | Manifold - torus = bounded/stable |
| `initial_spread` | ManifoldConfig | Initial state variance - affects gradient flow |
| `depth` | ManifoldConfig | Number of evolution steps |
| `dim` | ManifoldConfig | Model dimension |
| `heads` | ManifoldConfig | Number of heads |

---

*Analysis Date: 2026-04-02*
*Version: GSSM v2.7.2*

# MANIFOLD Technical Handbook


**Status:** Unified Reference
**Last Updated:** January 27, 2026

This handbook provides the mathematical foundations, training protocols, and technical specifications of the MANIFOLD architecture, an implementation of Geodesic Flow Networks (GFN) for sequence modeling with O(1) memory complexity.



## I. Fundamental Equations

The MANIFOLD architecture is governed by the principles of **Symplectic Sequence Modeling**, which reformulates neural computation as the movement of a particle over a learnable Riemannian manifold.

### 1.1 The Geodesic Equation

State evolution is modeled as a particle following the shortest path (geodesic) in a latent Riemann space:

$$\frac{d^2x^k}{d\tau^2} + \Gamma^k_{ij} \frac{dx^i}{d\tau} \frac{dx^j}{d\tau} + \mathcal{F}^k_{\text{friction}} = F^k_{\text{ext}}$$

Where notation follows the conventions of mathematical physics:

| Symbol | Meaning | Neural Analogue |
|--------|---------|-----------------|
| $x$ | Semantic position | Latent state |
| $v = \frac{dx}{d\tau}$ | Semantic velocity | Information momentum |
| $\Gamma^k_{ij}$ | Christoffel symbols | Manifold curvature |
| $F_{\text{ext}}$ | External force | Token embeddings $E(\text{token}_t)$ |

### 1.2 Low-Rank Christoffel Parameterization

To ensure O(d) parameter complexity, we approximate Christoffel symbols using a low-rank symmetric decomposition:

$$\Gamma(v, x) \approx W \cdot \left[ (U^T v)^2 \odot \sigma(V^T x) \right]$$

This formulation enables O(d²) interaction resolution (Multi-Head Attention-like mixing) while remaining linear in memory.

| Component | Dimension | Function |
|------------|-----------|---------|
| $U \in \mathbb{R}^{d \times r}$ | Basis matrix | Velocity projection |
| $W \in \mathbb{R}^{d \times r}$ | Weight matrix | Curvature composition |
| $\sigma(\cdot)$ | Soft saturation | Numerical stability |
| $r$ | Rank (16-64) | Compression |

### 1.3 Symplectic Integration (Leapfrog)

To preserve information over infinite horizons, we use the second-order Velocity Verlet scheme:

1. **Velocity half-step**:
   $$v_{t+\frac{1}{2}} = v_t + \frac{1}{2}\Delta t \cdot \left(A(x_t, v_t) - \sigma(W_f x_t)v_t\right)$$

2. **Position full-step**:
   $$x_{t+1} = x_t + \Delta t \cdot v_{t+\frac{1}{2}}$$

3. **Final velocity half-step**:
   $$v_{t+1} = v_{t+\frac{1}{2}} + \frac{1}{2}\Delta t \cdot \left(A(x_{t+1}, v_{t+\frac{1}{2}}) - \sigma(W_f x_{t+1})v_{t+\frac{1}{2}}\right)$$

This phase-space volume preservation ensures that gradients neither vanish nor explode, solving the fundamental gradient problem in traditional RNNs.



## II. Loss Engine (Force Fields)

MANIFOLD uses a composite loss function to balance task performance with geometric stability.

### 2.1 Hamiltonian Loss (`hamiltonian_loss`)

**Purpose**: Energy conservation.

**Formula**: $L_H = \lambda_h \sum |E_t - E_{t-1}|$ where $E = \|v\|^2$.

**Usage**: Prevents violent transitions in latent energy. If the model "startles" between tokens, this loss penalizes the discontinuity.

| Weight | Behavior |
|------|----------------|
| High | Rigid, linear reasoning |
| Low | Fluid, highly curved reasoning |

### 2.2 Geodesic Regularization (`geodesic_regularization`)

**Purpose**: Curvature control.

**Formula**: $L_G = \lambda_g \| \Gamma(v, v) \|^2$.

**Usage**: Prevents "Semantic Black Holes" (singularities). Keeps the manifold "locally flat," ensuring that small input changes do not lead to catastrophic state changes.

### 2.3 Curiosity Loss (`curiosity_loss`)

**Purpose**: Entropy maximization / Exploration.

**Formula**: $L_C = -\lambda_c \sum \log(\text{std}(v))$.

**Usage**: Prevents "Cognitive Collapse" where all neurons synchronize to the same value. Forces the manifold to use its full dimensional capacity.

> [!WARNING]
> High curiosity weights can cause divergence in rigid logic tasks (e.g., Parity).

### 2.4 Noether Loss (`noether_loss`)

**Purpose**: Semantic symmetry.

**Usage**: Ensures that different "Heads" on the manifold learn consistent physical laws. Enforce SO(N) rotational invariance in latent space.



## III. Riemannian Optimization

Standard optimizers (Adam, SGD) assume a flat Euclidean space. MANIFOLD requires **Riemannian Optimization** to respect weight constraints.

### 3.1 The Euclidean Drift Problem

In curved spaces, a standard update $W \leftarrow W - \eta \nabla$ moves weights "out of" the desired manifold, leading to "Gradient Drift" and loss oscillations.

### 3.2 RiemannianAdam

**Implementation**: `gfn.optim.RiemannianAdam`.

**Key mechanism: Retraction.**

After the Adam update, we project the weights back onto the manifold boundary:

$$W_{\text{new}} = \text{Retract}(W_{\text{old}} - \eta \cdot \hat{g}) = \frac{W_{\text{old}} - \eta \cdot \hat{g}}{\max(1, \|W\| / \text{max\_norm})}$$

### 3.3 Recommended Configuration

| Parameter | Value | Reason |
|-----------|-------|-------|
| `lr` | 1e-4 (stability) to 3e-4 (speed) | Precision-speed balance |
| `max_norm` | 10.0 | Stops weight explosion |
| `retraction` | `'normalize'` | Stable projection |
| `weight_decay` | 1e-4 | Regularization |



## IV. Optimal Physics Configuration

The following configuration has been validated in the superiority benchmark (`vis_gfn_superiority.py`) and represents the recommended configuration:

```python
physics_config = {
    'embedding': {
        'type': 'functional',     # Neural field embedding
        'mode': 'linear',          # Superior to 'binary' for generalization
        'coord_dim': 16
    },
    'readout': {
        'type': 'implicit',        # Holographic - latent state is answer
        'coord_dim': 16
    },
    'active_inference': {
        'enabled': True,
        'dynamic_time': {
            'enabled': True        # Adaptive timestep
        },
        'reactive_curvature': {
            'enabled': True,
            'plasticity': 0.2      # α for λ(K) = α·tanh(K)
        },
        'singularities': {
            'enabled': True,
            'strength': 20.0,      # Logical attraction strength
            'threshold': 0.8       # Activation
        }
    },
    'fractal': {
        'enabled': True,
        'threshold': 0.5,
        'alpha': 0.2
    },
    'topology': {
        'type': 'torus'            # Cyclic topology
    },
    'stability': {
        'base_dt': 0.4,            # Integration timestep
        'curvature_clamp': 5.0     # Max curvature
    }
}
```

### 4.1 Reactive Curvature

Reactive curvature modulates geometry based on model uncertainty:

$$K = \frac{1}{2}\|v\|^2 \quad \text{(Kinetic energy)}$$
$$\lambda(K) = \alpha \cdot \tanh(K)$$
$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot (1 + \lambda(K))$$

When the model is "confused" (high velocity), space becomes more curved, acting as automatic braking.

### 4.2 Logical Singularities

Singularities represent discrete decisions as topological attractors:

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

This allows a continuous system to represent discrete logical operations.



## V. Stability Protocols

### 5.1 Gradient Clipping

Use strictly $0.05$ to $1.0$. Manifold dynamics are sensitive to high-frequency noise.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
```

### 5.2 Velocity Normalization

The v2.6.0+ engine automatically normalizes $v$ to unit norm post-integration to prevent numerical drift:

```python
v_next = v_next / (||v_next|| + ε)
```

### 5.3 Warm-up

Always use learning rate warm-up (e.g., `OneCycleLR`) to allow curvature kernels to stabilize before high-speed training.

### 5.4 Integrator Parameters

| Integrator | Order | Energy Error | Use Case |
|------------|-------|------------------|-------------|
| Forest-Ruth | 4th | 0.000048% | **Maximum precision** |
| Leapfrog | 2nd | Low | **Production** |
| Heun | 2nd | Medium | Debugging |



## VI. Specifications Summary

### 6.1 Computational Complexity

| Operation | Training | Inference |
|-----------|---------------|------------|
| Embedding | O(L·d²) | O(d²) |
| Christoffel | O(L·d²·r) | O(d²·r) |
| Integration | O(L·d) | O(d) |
| Readout | O(L·d·V) | O(d·V) |
| **Total** | **O(L·d²·r)** | **O(d²·r)** |

### 6.2 Memory Scaling

| Configuration | VRAM (approx) |
|---------------|--------------|
| Small (d=128, depth=4) | 512 MB |
| Medium (d=256, depth=6) | 1.2 GB |
| Large (d=512, depth=12) | 3.8 GB |
| XL (d=1024, depth=24) | 8.5 GB |

### 6.3 Architecture Comparison

| Architecture | Memory | Gradients | Inductive Bias |
|--------------|---------|------------|----------------|
| Transformer | O(N²) | Good | Permutation |
| LSTM/GRU | O(1) | Poor | Sequential |
| Mamba | O(1) | Medium | Compression |
| **Manifold** | **O(1)** | **Excellent** | **Geometry** |



## VII. Experimental Validation

### 7.1 Parity Task

| Length | Accuracy | VRAM | Ratio |
|----------|----------|------|-------|
| 20 (train) | 100% | 28.3 MB | 1× |
| 1,000 | 100% | 30.5 MB | 50× |
| 100,000 | 100% | 30.6 MB | 5,000× |

**Key result**: Perfect generalization to sequences 5,000× longer than training.

### 7.2 Transformer Comparison

| Metric | Manifold | Transformer |
|---------|----------|-------------|
| Accuracy (L=1000) | 100% | 52.5% |
| VRAM (L=2000) | 33.3 MB | 325.1 MB |
| Convergence | ~500 steps | ~4000 steps |



**Document Version**: 2.6.2  
**Last Update**: January 27, 2026

For implementation details, see [API.md](API.md) or [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).  
For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For training, see [TRAINING.md](TRAINING.md).  
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).  
For theoretical foundations, see [PHYSICS.md](PHYSICS.md).

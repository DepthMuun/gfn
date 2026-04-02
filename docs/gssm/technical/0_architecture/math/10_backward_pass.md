# Backward Pass - Gradient Flow

**Source Files Analyzed:**
- PyTorch autograd system
- `models/base.py` (gradient flow through evolution loop)
- `physics/integrators/base.py` (differentiable operations)

---

## 1. Overview

GSSM relies entirely on PyTorch's automatic differentiation (autograd). No manual gradients are computed - all operations in the forward pass are differentiable.

---

## 2. Gradient Entry Point

```python
# Training loop
logits, (x_final, v_final), info = model(input_ids)
loss = cross_entropy(logits, targets)
loss.backward()  # <-- Gradient computation starts here
```

**Flow**:
```
loss ← logits ← readout ← x_final ← v_final ← ... ← x0, v0, embedding
```

---

## 3. Differentiable Operations in Forward Pass

### 3.1 Embedding Layer

```python
all_forces = self.embedding(input_ids)
# nn.Embedding is differentiable
# Gradient flows to: embedding.weight
```

### 3.2 State Initialization

```python
x = self.x0.expand(batch_size, ...) + torch.randn_like(x) * self.initial_spread
# .expand() is differentiable (views)
# torch.randn_like() is non-differentiable (noise)
# Gradient flows to: x0 parameter
```

### 3.3 Evolution Loop

**In `base.py:_evolve_sequence()`**:

```python
for i in range(l_seq_len):
    force = fs[:, i] * ms[:, i]  # Element-wise mul: differentiable
    
    for layer in self.layers:
        res = layer(local_x, local_v, force)
        local_x, local_v = res[0], res[1]
```

Each layer call is differentiable - gradients flow through:
- Layer parameters
- State transitions
- Force applications

---

## 4. ManifoldLayer Gradient Flow

**Location**: `models/manifold_layer.py:119`

### Step-by-Step Differentiability

```python
# 1. Reshape (view) - no gradient needed
x_3d = x.reshape(B * S, self.heads, self.head_dim)

# 2. Pre-integrate plugins (if differentiable)
for plugin in self.plugins.values():
    x_3d, v_3d, dt_eff = plugin.pre_integrate(...)

# 3. Integrator step
res = self.integrator.step(x_3d, v_3d, force=f_3d, dt=dt_eff)
x_stepped, v_stepped = res["x"], res["v"]

# 4. Post-integrate plugins
for plugin in self.plugins.values():
    x_stepped, v_stepped = plugin.post_integrate(...)

# 5. Mixer
x_mix, v_mix = self.mixer(x_stepped, v_stepped)
# FlowMixer: nn.Linear layers are differentiable

# 6. Dynamics routing
x_next = self.dynamics_x(x_ref_h, x_mix, context_x=x_ref_h)
# Dynamics: linear operations + activations, all differentiable

# 7. Topology wrap (differentiable)
x_next = self.integrator._resolve_topology(x_next)
# Uses: torch.atan2(torch.sin(x), torch.cos(x))
# Both sin and atan2 are differentiable
```

---

## 5. Integrator Differentiability

### 5.1 Leapfrog (lines 90-115)

```python
# All operations are differentiable:
mu1 = self._resolve_friction_mu(...)  # Parameter access
a1 = self._get_acceleration(...)        # Physics engine forward
a1_nf = a1 + mu1 * curr_v             # + and * are differentiable
v_half = (curr_v + 0.5 * eff_dt * a1_nf) / (1.0 + 0.5 * eff_dt * mu1 + EPS)
# +, *, / are all differentiable

# Position update
curr_x = self._resolve_topology(curr_x + eff_dt * v_half)
# Topology uses sin/cos/atan2 - all differentiable

# Velocity update
curr_v = (curr_v + eff_dt * a_avg) / (1.0 + eff_dt * mu_avg + EPS)
# All differentiable operations
```

### 5.2 Yoshida (lines 105-127)

```python
curr_x = self._resolve_topology(curr_x + self.c1 * eff_dt * curr_v)
a1 = self._get_acceleration(...)
curr_v = self._clamp_velocity(curr_v + self.d1 * eff_dt * a1)
# All differentiable: topology, +, *, clamp
```

### 5.3 RK4 (lines 40-69)

```python
k2_v_val = curr_v + (h / 2.0) * k1_a
k2_x_val = self._resolve_topology(curr_x + (h / 2.0) * k1_v)
k2_a = self._get_acceleration(k2_x_val, self._clamp_velocity(k2_v_val), ...)
# All differentiable operations

curr_x = self._resolve_topology(
    curr_x + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
)
# Weighted sum is differentiable
```

---

## 6. Physics Engine Differentiability

**Location**: `physics/engine.py:104`

```python
def compute_acceleration(self, x, v, force, dt):
    # 1. Geometry call
    geo_out = self.geometry(x, v, force=force)
    # Geometry: Christoffel computation is differentiable
    
    if isinstance(geo_out, tuple):
        christoffel, mu_geo = geo_out
    else:
        christoffel = geo_out
        mu_geo = 0.0
    
    # 2. Friction (differentiable)
    mu_total = self.get_friction_coefficient(x, v, mu_geo=mu_geo)
    friction_term = mu_total * v
    
    # 3. Net acceleration (differentiable)
    net_accel = -christoffel - friction_term
    
    # 4. Force addition (differentiable)
    if force is not None:
        net_accel = net_accel + force
    
    # 5. Auxiliary forces (if enabled)
    if self.hysteresis is not None:
        ghost_force = self.hysteresis(x, v)
        # Hysteresis: state update uses differentiable operations
        net_accel = net_accel + ghost_force
    
    if self.stochasticity_module is not None:
        stoch_force = self.stochasticity_module(x, v, dt)
        # Stochastic: generates noise (non-differentiable)
        # BUT: if sigma is learnable, gradient flows there
        net_accel = net_accel + stoch_force
    
    return net_accel
```

**Key Point**: Stochastic forces add noise (non-differentiable sampling), but if `sigma` is a parameter, gradients flow to it.

---

## 7. Geometry Differentiability

**Location**: `geometry/torus.py` (Christoffel computation)

```python
def connection(self, v, w, x):
    # Differentiable operations:
    # - torch.sin(x)
    # - torch.cos(x)
    # - *, +, / operations
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Gamma computation uses only differentiable ops
    gamma = (R + r * cos_theta) * sin_theta / r * v_phi * w_phi
    # All: *, +, /, sin, cos are differentiable
    
    return gamma
```

---

## 8. Gradient Flow Summary

### Trainable Parameters

| Component | Parameters | Gradient Source |
|-----------|-----------|-----------------|
| Embedding | `embedding.weight` | Token forces |
| Initial State | `x0`, `v0` | State evolution |
| Mixer | `FlowMixer` linear weights | Head mixing |
| Dynamics | `residual_scale`, gate weights | State updates |
| Geometry | `R`, `r` (if learnable) | Christoffel symbols |
| Hysteresis | `weight`, `bias` | Ghost forces |
| Stochasticity | `sigma`, `theta` | Noise magnitude |
| Curiosity | `strength` | Repulsion force |
| Readout | `proj.weight`, `proj.bias` | Logits |

### Non-Differentiable Operations

```python
# Random sampling (gradients don't flow through randomness)
torch.randn_like(v)

# Boolean masks (used for selection, not gradients)
mask = (dist_to_sing < threshold)

# Control flow (if statements block gradients)
if torch.isnan(x).any():  # Gradient doesn't flow through this check
    break
```

---

## 9. Checkpointing for Memory

For long sequences, gradient checkpointing can be used:

```python
from torch.utils.checkpoint import checkpoint

# Wrap layer forward
x = checkpoint(layer.forward, x, v, F)
```

**Effect**:
- Recomputes forward pass during backward
- Saves memory (trades compute for memory)
- Gradients are still correct

---

## 10. Jacobian and Spectral Radius

The Jacobian of the transformation relates to gradient flow:

```python
# Jacobian: how output changes w.r.t. input
J = torch.autograd.functional.jacobian(func, x)

# Spectral radius: largest eigenvalue magnitude
spectral_radius = torch.linalg.norm(J, 2)
```

**Large spectral radius** (> 1): Gradients may explode
**Small spectral radius** (< 1): Gradients may vanish

---

## 11. Complete Gradient Flow

```
Loss.backward()
    ↓
CrossEntropy gradients ← Readout projection weights
    ↓
x_final, v_final ← Layer depth gradients
    ↓
For each layer (reversed):
    ↓
    Dynamics routing gradients
        ↓
    Mixer gradients
        ↓
    Integrator step gradients
        ↓
        Physics engine gradients
            ↓
            Geometry Christoffel gradients
            Friction coefficient gradients
        ↓
        State update gradients
    ↓
    Plugin gradients (if any)
    ↓
x0, v0 gradients ← Initial state parameters
Embedding gradients ← Token embedding weights
```

---

## 12. Common Gradient Issues

### Vanishing Gradients

**Symptom**: `min_gradient_norm ~ 1e-08`

**Cause**:
- Long sequences (depth × seq_len operations)
- Small `initial_spread` (near-zero initialization)
- High friction (damps gradients)

**Solution**:
```python
initial_spread = 0.1  # Increase from 1e-3
friction = 0.01  # Don't increase too much
use_residual = True  # Better gradient flow
```

### Exploding Gradients

**Symptom**: `max_gradient_norm > 1000`

**Cause**:
- Large `dt` without proper scaling
- Velocity saturation disabled
- Long unrolled sequences

**Solution**:
```python
dt = 0.05  # Decrease timestep
velocity_saturation = 10.0  # Enable clamping
grad_clip = 1.0  # Gradient clipping
```

---

*File: technical/0_architecture/math/10_backward_pass.md*
*Last Updated: 2026-04-02*

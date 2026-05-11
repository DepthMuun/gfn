# Physics Engine - Mathematical Foundation

## Overview

The Physics Engine computes the net acceleration that drives the evolution of the manifold state. It combines geometric, external, and auxiliary forces.

---

## 1. Core Equation

The fundamental equation governing the dynamics:

$$\frac{dv}{dt} = a_{net}$$

Where the net acceleration is:

$$a_{net} = -\Gamma(x, v) + F_{ext} + F_{friction} + F_{ghost} + F_{stochastic} + F_{curiosity}$$

---

## 2. Christoffel Symbols (Geometric Force)

### Definition

The Christoffel symbols $\Gamma^\sigma_{\mu\nu}$ represent the geometric force arising from manifold curvature:

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right)$$

### In Code

```python
# From geometry(x, v, force)
christoffel = geometry(x, v)  # Returns Γ(x,v)
```

### Geometric Force

The geometric acceleration is:

$$F_{geometric} = -\Gamma(x, v)$$

This force pushes the state along geodesics (shortest paths) on the manifold.

---

## 3. Friction Force

### Definition

Friction provides velocity damping to prevent runaway acceleration:

$$F_{friction} = -\mu \cdot v$$

Where $\mu$ is the friction coefficient.

### Friction Calculation

The total friction coefficient combines geometry-provided and config values:

```python
mu_total = get_friction_coefficient(x, v, mu_geo)
friction_term = mu_total * v
```

### Sources of Friction

1. **Geometry-provided** (from geometry return tuple):
   - $\mu_{geo} = f(\text{curvature}, \text{velocity})$

2. **Config fallback** (from `stability.friction`):
   - Default: $\mu = 0.01$

3. **Velocity-dependent** (from `stability.velocity_friction_scale`):
   - $\mu_{total} = \mu_{base} + \mu_{velocity} \cdot \|v\|$

---

## 4. External Force

### Definition

The external force comes from token embeddings:

$$F_{ext} = \text{Embedding}(token\_ids)$$

### In Code

```python
force = embedding(input_ids)  # [B, S, D]
net_accel = net_accel + force
```

---

## 5. Hysteresis Ghost Force

### Purpose

Provides memory of previous states through a "ghost" force that persists across timesteps.

### Mathematical Form

$$F_{ghost} = W \cdot \tanh(b + h_{prev})$$

Where:
- $h_{prev}$ is the hysteresis state from previous timestep
- $W$ is a learnable weight matrix
- $b$ is a bias term

### Update Rule

$$h_{new} = (1 - \alpha) \cdot h_{prev} + \alpha \cdot v$$

Where $\alpha = decay$ is the hysteresis decay rate.

### In Code

```python
if self.hysteresis is not None:
    ghost_force = self.hysteresis(x, v, topo_id)
    net_accel = net_accel + ghost_force
```

---

## 6. Stochastic Forces

### Brownian Motion

For exploration/noise injection:

$$F_{stochastic} = \sigma \cdot \mathcal{N}(0, 1)$$

Where $\sigma$ is the noise magnitude.

### Ornstein-Uhlenbeck (OU)

For correlated noise:

$$dX = -\theta(X - \mu)dt + \sigma dW$$

In discrete form:

$$F_{ou} = -\theta \cdot (v - \mu) + \sigma \cdot \sqrt{dt} \cdot \mathcal{N}(0, 1)$$

### In Code

```python
if self.stochasticity_module is not None:
    stoch_force = self.stochasticity_module(x, v, dt)
    net_accel = net_accel + stoch_force
```

---

## 7. Curiosity Force

### Purpose

Encourages exploration by adding a force that increases future uncertainty.

### Mathematical Form

$$F_{curiosity} = \lambda \cdot \nabla_v H$$

Where:
- $H$ is an entropy-like measure of state diversity
- $\lambda$ is the curiosity strength

### Implementation

```python
if self.curiosity_module is not None:
    curiosity_force = self.curiosity_module(x, v)
    net_accel = net_accel + curiosity_force
```

---

## 8. Singularity Damping

### Purpose

Prevents numerical explosion when approaching singularities (where metric tensor becomes singular).

### Mathematical Form

When $\|v\| > threshold$:

$$F_{singularity} = -S \cdot \frac{v}{\|v\|} \cdot \tanh\left(\frac{\|v\| - \epsilon}{\epsilon}\right)$$

Where:
- $S$ is the singularity strength
- $\epsilon$ is a small threshold

---

## 9. Velocity Saturation

### Purpose

Clamps maximum velocity to prevent instability.

### Mathematical Form

$$v_{sat} = v_{max} \cdot \tanh\left(\frac{v}{v_{max}}\right)$$

Or if disabled ($v_{max} = 0$):

$$v_{sat} = v$$

---

## 10. Complete Acceleration Computation

```python
def compute_acceleration(x, v, force, dt):
    # 1. Get Christoffel from geometry
    geo_out = geometry(x, v, force)
    christoffel, mu_geo = geo_out  # or just christoffel
    
    # 2. Compute friction
    mu_total = get_friction_coefficient(x, v, mu_geo)
    friction_term = mu_total * v
    
    # 3. Net acceleration
    net_accel = -christoffel - friction_term
    
    # 4. Add external force
    if force is not None:
        net_accel = net_accel + force
    
    # 5. Add hysteresis ghost force
    if hysteresis is not None:
        net_accel = net_accel + hysteresis(x, v)
    
    # 6. Add stochastic force
    if stochasticity is not None:
        net_accel = net_accel + stochasticity(x, v, dt)
    
    # 7. Add curiosity force
    if curiosity is not None:
        net_accel = net_accel + curiosity(x, v)
    
    return net_accel
```

---

## Parameter Summary

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| `friction` | $\mu$ | 0.01 | Velocity damping |
| `velocity_friction_scale` | $\mu_v$ | 0.0 | Velocity-dependent friction |
| `velocity_saturation` | $v_{max}$ | 0.0 | Max velocity (0=off) |
| `hyst_decay` | $\alpha$ | 0.1 | Hysteresis memory decay |
| `stochasticity.sigma` | $\sigma$ | 0.01 | Noise magnitude |
| `stochasticity.theta` | $\theta$ | 0.15 | OU mean reversion |
| `curiosity.strength` | $\lambda$ | 0.1 | Exploration drive |
| `singularity.strength` | $S$ | 0.1 | Singularity damping |

---

*File: technical/0_architecture/math/01_physics_engine.md*
*Last Updated: 2026-04-02*

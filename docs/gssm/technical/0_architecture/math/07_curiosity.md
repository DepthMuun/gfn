# Curiosity - Mathematical Foundation

## Overview

Curiosity provides an intrinsic motivation force that encourages exploration by pushing the system away from high-density states (batch center), promoting diverse state coverage.

---

## 1. Geometric Curiosity Force

### Purpose

- Escape local density clusters
- Promote state space exploration
- Prevent mode collapse

### Core Idea

Repel from the batch "center of mass":

$$F_{curiosity} \propto \frac{(x - \bar{x})}{\|x - \bar{x}\|^2}$$

Where $\bar{x}$ is the batch center.

---

## 2. Toroidal Geometry

### Batch Center

For toroidal manifolds, we use circular mean:

$$\bar{x}_{torus} = \arctan_2\left(\frac{1}{N}\sum_i \sin(x_i), \frac{1}{N}\sum_i \cos(x_i)\right)$$

Where:
- $\arctan_2(y, x)$ is the two-argument arctangent
- $N$ is batch size

### Escape Direction

$$d = x - \bar{x}_{torus}$$

$$d_{geo} = \arctan_2(\sin(d), \cos(d))$$

The $\arctan_2(\sin, \cos)$ wraps the angular difference correctly.

### Implementation

```python
sin_x = torch.sin(x)
cos_x = torch.cos(x)
batch_center = torch.atan2(sin_x.mean(dim=0), cos_x.mean(dim=0))
direction = x - batch_center
direction = torch.atan2(torch.sin(direction), torch.cos(direction))
```

---

## 3. Euclidean Geometry

### Batch Center

Simple arithmetic mean:

$$\bar{x}_{euclidean} = \frac{1}{N}\sum_i x_i$$

### Escape Direction

$$d = x - \bar{x}_{euclidean}$$

### Implementation

```python
batch_center = x.mean(dim=0)
direction = x - batch_center
```

---

## 4. Repulsion Magnitude

### Formula

From `GeometricCuriosityForce.forward()`:

$$\|F_{curiosity}\| = \frac{\lambda}{\|d\|^2 + \epsilon}$$

Where:
- $\lambda$ = `strength` (default: 0.1)
- $\|d\|^2$ = squared distance to center
- $\epsilon$ = 1e-6 (numerical stability)

### Complete Force

$$F_{curiosity} = \frac{d}{\|d\|} \cdot \frac{\lambda}{\|d\|^2 + \epsilon}$$

Or written differently:

$$F_{curiosity} = \lambda \cdot \frac{d}{\|d\|^3 + \epsilon'}$$

Where $\epsilon' = \epsilon \cdot \|d\|$.

### In Code

```python
dist_sq = (direction ** 2).sum(dim=-1, keepdim=True) + 1e-6
repulsion_mag = self.strength / dist_sq
force = (direction / (dist_sq ** 0.5 + 1e-8)) * repulsion_mag
```

---

## 5. Force Clamping

### Maximum Force

To prevent instability:

$$F_{clamped} = \text{clamp}(F_{curiosity}, -5.0, 5.0)$$

This limits the maximum curiosity force to prevent runaway dynamics.

---

## 6. Configuration

```python
physics = {
    'active_inference': {
        'curiosity': {
            'enabled': True,
            'strength': 0.1,    # Repulsion magnitude
            'decay': 0.99       # Not used in current implementation
        }
    }
}
```

### Parameter Effects

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| `strength` | $\lambda$ | 0.1 | Repulsion force scale |
| `decay` | - | 0.99 | Reserved for future use |

---

## 7. Physical Interpretation

### Analogies

| Concept | Physics | GSSM |
|---------|---------|------|
| Like charges | Coulomb repulsion | $F \propto 1/r^2$ |
| Diffusion | Brownian motion | Spread from center |
| Entropy | $S = k \ln \Omega$ | Increase state diversity |

### Behavior

- Near center ($r \to 0$): Strong repulsion
- Far from center ($r \to \infty$): Weak repulsion
- Isotropic: Same magnitude in all directions

---

## 8. Comparison with Stochasticity

| Property | Curiosity | Stochasticity |
|----------|-----------|---------------|
| Direction | Deterministic | Random |
| Source | Batch statistics | Random sampling |
| Purpose | Exploration | Noise injection |
| Gradient | Preserves structure | Disrupts structure |
| Physics | Coulomb-like | Thermal-like |

---

## 9. Integration with Physics Engine

```python
# In PhysicsEngine.compute_acceleration()

if self.curiosity_module is not None:
    curiosity_force = self.curiosity_module(x, v)
    net_accel = net_accel + curiosity_force
```

Added to net acceleration just like stochastic forces.

---

## 10. Use Cases

1. **Mode collapse prevention**: Spread representations
2. **Exploration**: Discover new regions of state space
3. **Diversity**: Increase batch diversity
4. **Clustering**: Counter-act excessive clustering

---

## 11. Limitations

- Batch-dependent: Requires batch size > 1
- Computation: Needs mean computation each step
- Stability: Strong repulsion near center (clamped)
- Topology-aware: Different behavior on torus vs euclidean

---

*File: technical/0_architecture/math/07_curiosity.md*
*Last Updated: 2026-04-02*

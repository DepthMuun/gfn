# Singularities - Mathematical Foundation

## Overview

Singularity handling prevents numerical explosions when the metric tensor approaches singular values (determinant → 0), which occurs in regions of extreme curvature.

---

## 1. Singularity Detection

### Metric Tensor Analysis

A singularity is detected when the metric tensor becomes degenerate:

$$\det(g_{ij}) \approx 0 \quad \text{or} \quad \lambda_{min}(g_{ij}) \approx 0$$

Where:
- $\det(g_{ij})$ is the determinant of the metric
- $\lambda_{min}$ is the smallest eigenvalue

### Detection Formula

From `SingularityDetector`:

$$s_{measure} = \min(|\det(g)|, |\lambda_{min}|)$$

$$is\_singular = \mathbb{1}(s_{measure} < \epsilon_{threshold})$$

Where $\mathbb{1}$ is the indicator function and $\epsilon_{threshold}$ = `singularity.threshold` (default: 1e-4).

---

## 2. Singularity Gating

### Purpose

Smoothly damp velocity and force as the metric approaches singularity.

### Gate Function

From `SingularityGate.forward()`:

$$g = \sigma(s \cdot (d - \tau))$$

Where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $s$ = `slope` (strength × 20.0)
- $d$ = distance to singularity = $|metric\_component|$
- $\tau$ = `threshold`

### Behavior

- When $d \gg \tau$: gate → 1 (no damping)
- When $d \approx \tau$: gate → 0.5 (partial damping)
- When $d \ll \tau$: gate → 0 (full damping)

---

## 3. Velocity Damping

### Formula

$$v_{damped} = v \cdot g$$

Where $g$ is the gate value from above.

### Implementation

```python
def damp_velocity(self, v, metric_component):
    gate = self.forward(None, None, metric_component)
    return v * gate
```

---

## 4. Force Damping

### Formula

$$F_{damped} = F \cdot g$$

Same gate applied to external forces.

### Implementation

```python
def damp_force(self, force, metric_component):
    gate = self.forward(None, None, metric_component)
    return force * gate
```

---

## 5. Sigmoid Gate Properties

### Smooth Transition

The sigmoid provides a smooth, differentiable transition:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Derivative:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### Parameter Effects

| Parameter | Effect |
|-----------|--------|
| slope ↑ | Sharper transition |
| slope ↓ | Smoother transition |
| threshold ↑ | Earlier damping |
| threshold ↓ | Later damping |

---

## 6. Configuration

```python
physics = {
    'singularities': {
        'enabled': True,
        'epsilon': 1e-8,        # Small value for numerical stability
        'strength': 0.1,        # Slope = strength × 20.0
        'threshold': 0.0001     # Distance threshold
    }
}
```

### Parameter Formulas

| Config | Code Variable | Default | Formula |
|--------|--------------|---------|---------|
| `epsilon` | `EPS` | 1e-8 | $\epsilon_{numerical}$ |
| `strength` | `slope` | 0.1 | $s = strength \times 20$ |
| `threshold` | `threshold` | 1e-4 | $\tau$ |

---

## 7. Complete Flow

```python
# 1. Detect singularity
singularity_measure = min(|det(g)|, |λ_min|)

# 2. Compute gate
gate = sigmoid(slope * (singularity_measure - threshold))

# 3. Apply damping
v_damped = v * gate
F_damped = F * gate

# 4. Return to physics engine
net_accel = net_accel + F_damped  # etc.
```

---

## 8. Physical Interpretation

- **Singularity**: Point where metric tensor loses invertibility
- **Metric determinant**: Volume element of tangent space
- **Near singularities**: Christoffel symbols diverge
- **Gate function**: Smoothly prevents the system from entering unstable regions

---

## 9. Comparison with Physics

| Concept | Physics | GSSM |
|---------|---------|------|
| Black hole singularity | $r \to 0$ | $\det(g) \to 0$ |
| Event horizon | $r = 2M$ | $d = threshold$ |
| Escape velocity | $v_{esc}$ | Gate → 0 |

---

*File: technical/0_architecture/math/05_singularities.md*
*Last Updated: 2026-04-02*

# Geometry - Mathematical Foundation

## Overview

Geometry modules compute the Christoffel symbols and metric tensor that define the manifold's curvature and distance measures.

---

## 1. Metric Tensor

### Definition

The metric tensor $g_{ij}$ defines the inner product on the tangent space:

$$ds^2 = g_{ij} dx^i dx^j$$

### Properties

- Symmetric: $g_{ij} = g_{ji}$
- Positive definite (for Riemannian manifolds)
- Used for: distance, angles, volume measures

### Inverse Metric

$$g^{ij} = (g_{ij})^{-1}$$

Used to raise/lower indices:

$$v^i = g^{ij} v_j$$

---

## 2. Christoffel Symbols

### Definition

The Christoffel symbols $\Gamma^k_{ij}$ are the Levi-Civita connection coefficients:

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right)$$

### Physical Interpretation

- Not a tensor (doesn't transform covariantly)
- Represents the "straightest lines" (geodesics) on curved manifolds
- $\Gamma^k_{ij} = 0$ in flat (Euclidean) space

### Geometric Force

The acceleration due to curvature:

$$a^k = -\Gamma^k_{ij} v^i v^j$$

This is the geometric component of the physics engine:

$$F_{geometric} = -\Gamma(x, v)$$

---

## 3. Toroidal Geometry

### Manifold

The torus $T^n = S^1 \times S^1 \times ... \times S^1$ is the product of $n$ circles.

### Metric Tensor

For a 2D torus with major radius $R$ and minor radius $r$:

$$g = \begin{pmatrix} (R + r\cos\theta)^2 & 0 \\ 0 & r^2 \end{pmatrix}$$

### Christoffel Symbols (2D Torus)

Non-zero components:

$$\Gamma^\theta_{\theta\phi} = \Gamma^\theta_{\phi\theta} = -\frac{r\sin\theta}{R + r\cos\theta}$$

$$\Gamma^\phi_{\theta\theta} = \frac{(R + r\cos\theta)\sin\theta}{r}$$

All others are zero.

### In Code

```python
# From torus.py connection() method
# For each head dimension pair (θ, φ):
gamma_theta = (R + r * cos(theta)) * sin(theta) / r * v_phi * w_phi
gamma_phi = -r * sin(theta) / (R + r * cos(theta)) * (v_phi * w_theta + v_theta * w_phi)
```

### Properties

- **Bounded**: Position wraps at $2\pi$
- **Periodic**: $\theta + 2\pi \equiv \theta$
- **Curved**: Non-zero Christoffel symbols
- **Recommended**: Default for GSSM training

---

## 4. Euclidean Geometry

### Manifold

Flat space $\mathbb{R}^n$.

### Metric Tensor

$$g_{ij} = \delta_{ij}$$

### Christoffel Symbols

All zero:

$$\Gamma^k_{ij} = 0 \quad \forall i,j,k$$

### In Code

```python
# From euclidean.py
return torch.zeros_like(v)  # No geometric force
```

### Properties

- Unbounded (can explode)
- No curvature
- Simple but unstable

---

## 5. Low-Rank Riemannian Geometry

### Purpose

Efficient approximation for high-dimensional manifolds.

### Key Idea

Instead of computing full $D \times D$ Christoffel matrix, decompose as low-rank:

$$\Gamma \approx \sum_{r=1}^{R} U_r \cdot W_r$$

Where $R \ll D$.

### Approximation

```python
# From low_rank.py
# gamma ≈ sum_r (U_r @ W_r.T) * (v^T @ V_r)
gamma = sum_r (U_r @ W_r.T) * inner_product
```

### Complexity

| Method | Complexity |
|--------|------------|
| Full | $O(D^3)$ |
| Low-Rank | $O(R^2 \cdot D)$ |

### Properties

- Memory efficient
- Faster computation
- Approximate (not exact)

---

## 6. Reactive Geometry

### Purpose

Adaptive geometry that adjusts curvature based on state.

### Key Idea

The Christoffel symbols are modulated by a learnable "reactivity" factor:

$$\Gamma_{reactive} = \gamma_{base} \cdot (1 + \alpha \cdot \tanh(s))$$

Where:
- $\gamma_{base}$ is the static Christoffel
- $\alpha$ is a plasticity parameter
- $s$ is a state-dependent signal

### Friction Gating

Also returns a friction coefficient based on curvature:

```python
# From reactive.py
mu_geo = base_friction + plasticity * curvature_magnitude
return christoffel, mu_geo
```

### Properties

- Adaptive to input
- Can learn task-specific geometry
- More complex training

---

## 7. Hyperbolic Geometry (Poincaré Ball)

### Manifold

Poincaré ball model of hyperbolic space.

### Metric Tensor

For point $x$ in unit ball:

$$g_{ij} = \frac{4}{(1 - \|x\|^2)^2} \delta_{ij}$$

### Christoffel Symbols

$$\Gamma^k_{ij} = \frac{1}{1 - \|x\|^2} (x_i \delta_{kj} + x_j \delta_{ki} - x_k \delta_{ij})$$

### Distance

$$d(u, v) = \text{arcosh}\left(1 + \frac{2\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

### Properties

- Negative curvature
- Good for tree-like structures
- Bounded (unit ball)

---

## 8. Spherical Geometry

### Manifold

Sphere $S^n$ of radius $r$.

### Metric Tensor

$$g_{ij} = r^2 \delta_{ij}$$

(in local coordinates)

### Christoffel Symbols

For sphere embedded in $\mathbb{R}^{n+1}$:

$$\Gamma^k_{ij} = -\frac{1}{r^2} (x_i \delta_{kj} + x_j \delta_{ki} - x_k \delta_{ij})$$

### Properties

- Constant positive curvature
- Bounded surface
- Good for rotational data

---

## 9. Holographic Geometry

### Purpose

Uses holographic (interference) patterns for representation.

### Key Idea

Represent state as complex-valued amplitude:

$$\psi(x) = A(x) e^{i\phi(x)}$$

Where:
- $A(x)$ is amplitude (magnitude)
- $\phi(x)$ is phase

### Christoffel from Phase

The geometric force comes from phase gradients:

$$\Gamma \propto \nabla \phi$$

### Properties

- Dense representations
- Quantum-inspired
- Experimental

---

## 10. Geometry Interface

All geometries implement:

```python
class Geometry(Protocol):
    def __call__(self, v, x, force=None):
        """
        Compute Christoffel symbols.
        Returns: torch.Tensor or (torch.Tensor, float)
        """
        
    def metric(self, x):
        """Compute metric tensor g_ij"""
        
    def project(self, x):
        """Project to manifold surface"""
        
    def dist(self, x1, x2):
        """Geodesic distance between points"""
```

---

## 11. Geometry Comparison

| Geometry | Curvature | Bounded | Complexity | Use Case |
|----------|-----------|---------|------------|----------|
| torus | Variable | ✅ | Medium | Default, general |
| euclidean | Zero | ❌ | Minimal | Simple tasks |
| low_rank | Variable | - | Low | High dimensions |
| reactive | Adaptive | - | Medium | Adaptive tasks |
| hyperbolic | Negative | ✅ | Medium | Trees, graphs |
| spherical | Positive | ✅ | Medium | Rotations |
| holographic | Variable | - | High | Experimental |

---

*File: technical/0_architecture/math/03_geometry.md*
*Last Updated: 2026-04-02*

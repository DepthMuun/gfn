# GFN Geometry System

The `gfn/realizations/gssm/geometry` module provides the mathematical foundation for GFN. Each geometry class implements a specific topology and metric space, registered via `GEOMETRY_REGISTRY`.

## 1. Base Strategy
All geometries inherit from `gfn.realizations.gssm.geometry.base.BaseGeometry` and implement the `Geometry` protocol from `gfn.realizations.gssm.interfaces.geometry`.

### Key Methods:
- `metric_tensor(x)`: Returns the metric tensor $g_{ij}$ at position $x$.
- `christoffel_symbols(x)`: Calculates the Christoffel symbols $\Gamma^k_{ij}$ (geodesic curvature).
- `forward(x, v, force)`: Computes acceleration, returns `(christoffel, friction)` tuple.
- `project(x)`: Projects a point back to the manifold.
- `dist(x1, x2)`: Measures the shortest distance (geodesic) between points.

---

## 2. Core Geometries (9 Total)

### Euclidean Geometry (`euclidean.py`)
Standard flat space.
- **Metric**: Identity matrix $I$.
- **Christoffel**: Always zero (straight lines).
- **Use Case**: Regression and baseline comparisons.

### Torus Geometry (`torus.py`)
Maps dimensions into pairs of $(\theta, \phi)$ on nested tori.
- **Metric**: $ds^2 = r^2 d\theta^2 + (R + r \cos \theta)^2 d\phi^2$.
- **Topology**: Periodic in $[-\pi, \pi]$ for all dimensions.
- **Use Case**: Language modeling and cyclic logic (XOR).

### Low-Rank Riemannian (`low_rank.py`)
Efficiently approximates high-dimensional curved spaces using low-rank decomposition.
- **Decomposition**: $\Gamma^k_{ij} \approx \Sigma_r W_{rk} \cdot (U_{ir} \cdot U_{jr})$
- **Optimization**: Reduces $O(D^3)$ to $O(Rank^2 \cdot D)$ via Woodbury Identity.
- **Use Case**: Large models where full metric calculation is prohibitive.

### Reactive Geometry (`reactive.py`)
Geometries that adjust their curvature dynamically based on input flow. Implements geometric plasticity with learnable parameters.

### Adaptive Geometry (`adaptive.py`)
Self-adjusting geometry that adapts to data distribution during training.

### Hyperbolic Geometry (`hyperbolic.py`)
Implements hyperbolic (Poincaré ball) space with negative curvature.
- **Use Case**: Hierarchical data and tree-like structures.

### Holographic Geometry (`holographic.py`)
Representations where geometry itself stores information through interference patterns. Implements associative memory.

### Hierarchical Geometry (`hierarchical.py`)
Multi-scale geometry for nested structural representations.

### Spherical Geometry (`spherical.py`)
Positive curvature geometry on $S^n$ sphere.
- **Use Case**: Directional data and normalized representations.

---

## 3. Geometry Factory
The `GeometryFactory` uses `GEOMETRY_REGISTRY` to instantiate classes based on config:

```python
from gfn.realizations.gssm.geometry.factory import GeometryFactory

geometry = GeometryFactory.create(config.physics)
# Returns appropriate geometry based on topology.type and riemannian_type
```

### Registration Pattern:
```python
from gfn.realizations.gssm.registry import register_geometry

@register_geometry('my_geometry')
class MyGeometry(BaseGeometry):
    ...
```

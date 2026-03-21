# Manifold Geometry System

The `gfn/geometry` module provides the mathematical foundation for GFN. Each geometry class implements a specific topology and metric space.

## 1. Base Strategy
All geometries inherit from `gfn.geometry.base.BaseGeometry` and must implement the metric tensor and connection logic.

### Key Methods:
- `metric(x)`: Returns the metric tensor $g_{\mu\nu}$ at position $x$.
- `christoffel(x, v)`: Calculates the Christoffel symbols $\Gamma^\sigma_{\mu\nu}$ (geodesic curvature).
- `dist(x1, x2)`: Measures the shortest distance (geodesic) between points.

---

## 2. Core Geometries

### Torus Geometry (`torus.py`)
Maps $D$ dimensions into pairs of $(\theta, \phi)$ on nested tori.
- **Metric**: $ds^2 = r^2 d\theta^2 + (R + r \cos \theta)^2 d\phi^2$.
- **Topology**: Periodic in $[-\pi, \pi]$ for all dimensions.
- **Use Case**: Language modeling and cyclic logic (XOR).

### Euclidean Geometry (`euclidean.py`)
Standard flat space.
- **Metric**: Identity matrix $I$.
- **Christoffel**: Always zero (straight lines).
- **Use Case**: Regression and baseline comparisons.

### Low-Rank Riemannian (`low_rank.py`)
Efficiently approximates high-dimensional curved spaces using a low-rank decomposition of the metric $g = I + UU^T$.
- **Optimization**: Reduces $O(D^3)$ inversion to $O(Rank^2 \cdot D)$ via Woodbury Identity.
- **Use Case**: Large models where full metric calculation is prohibitive.

---

## 3. Advanced Features

### Reactive Geometry (`reactive.py`)
Geometries that adjust their curvature $(\Gamma)$ dynamically based on the input flow. Implements geometric plasticity.

### Holographic Geometry (`holographic.md`)
Representations where the geometry itself stores the information (associative memory) through interference patterns.

---

## 4. Geometry Factory
The `GeometryFactory` uses the `GEOMETRY_REGISTRY` to instantiate classes based on the `topology_type` config.

```python
geometry = GeometryFactory.create(config.physics)
# Returns TorusGeometry if config.topology.type == 'torus'
```

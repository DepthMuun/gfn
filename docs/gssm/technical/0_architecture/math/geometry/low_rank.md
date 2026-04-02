# Low-Rank Riemannian Geometry

## What is Low-Rank Geometry?

Low-rank geometry is an efficient approximation of full Riemannian geometry. Instead of computing the complete Christoffel symbols (which is expensive in high dimensions), it uses a low-rank decomposition.

Think of it as: "Approximating complex curvature with a simpler, cheaper model."

---

## The Problem with High Dimensions

### Full Christoffel Computation

In $D$ dimensions, the Christoffel symbols are:
$$\Gamma^k_{ij} \quad \text{for } i, j, k \in \{1, ..., D\}$$

Total: $O(D^3)$ components.

**Computation cost**:
- Computing: $O(D^3)$ operations
- Storing: $O(D^3)$ memory
- For $D = 512$: 134 million values!

### The Bottleneck

High-dimensional manifolds become computationally prohibitive.

---

## Low-Rank Solution

### Approximation

Instead of full Christoffel, use rank-$R$ approximation:

$$\Gamma \approx \sum_{r=1}^R U_r \cdot W_r^T$$

Where:
- $U_r, W_r \in \mathbb{R}^D$ are learnable vectors
- $R \ll D$ (rank much smaller than dimension)
- Typically $R = 16$ or $32$

### Complexity Reduction

| Aspect | Full | Low-Rank |
|--------|------|----------|
| Parameters | $O(D^3)$ | $O(R \cdot D)$ |
| Computation | $O(D^3)$ | $O(R^2 \cdot D)$ |
| For $D=512, R=16$ | 134M | 8K |

**Savings**: ~16,000× fewer parameters!

---

## How It Works

### Christoffel Approximation

The geometric force becomes:
$$F_{geo} = -\Gamma(v, v) \approx -\sum_{r=1}^R (U_r \cdot W_r^T) \cdot (v \odot v)$$

Where $\odot$ is element-wise multiplication.

### Decomposition Structure

**Matrix form**:
$$\Gamma \approx U \cdot \Lambda \cdot W^T$$

Where:
- $U, W \in \mathbb{R}^{D \times R}$ (basis matrices)
- $\Lambda \in \mathbb{R}^{R \times R}$ (diagonal scaling)

### Physical Interpretation

The low-rank structure assumes:
- Curvature has "principal directions" (like PCA)
- Most variation happens in a few directions
- High-dimensional manifold is "almost flat" in most directions

---

## When to Use Low-Rank

### Use When

- **High dimensions**: $D > 128$
- **Speed critical**: Need fast computation
- **Memory constrained**: Limited GPU memory
- **Approximation acceptable**: Can trade accuracy for speed

### Don't Use When

- **Low dimensions**: $D < 64$ (overhead not worth it)
- **Accuracy critical**: Need exact curvature
- **Complex geometry**: Manifold has intricate structure

---

## Comparison with Full Geometry

| Property | Full | Low-Rank |
|----------|------|----------|
| Accuracy | Exact | Approximate |
| Speed | Slow | Fast |
| Memory | High | Low |
| Expressiveness | Complete | Limited |
| Training stability | May vary | Usually better |

---

## Rank Selection

### Guidelines

| Dimension $D$ | Suggested Rank $R$ | Compression |
|---------------|-------------------|-------------|
| 64 | 16 | 4× |
| 128 | 16 | 8× |
| 256 | 16-32 | 16-32× |
| 512 | 32-64 | 64-128× |

### Trade-off

- **Higher rank**: Better accuracy, slower
- **Lower rank**: Faster, more approximate

---

## Mathematical Formulation

### Full Rank
$$\Gamma_{ijk} = \frac{1}{2}\left(\frac{\partial g_{jk}}{\partial x^i} + \frac{\partial g_{ik}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^k}\right)$$

### Low-Rank Approximation
$$\tilde{\Gamma}_{ij}^k = \sum_{r=1}^R U_i^r \cdot W_j^r \cdot \lambda_r$$

Learn $U, W, \lambda$ instead of computing derivatives.

---

## Integration with GSSM

### Configuration

```python
physics = {
    'topology': {
        'type': 'low_rank',
        'rank': 16  # Low-rank dimension
    }
}
```

### Compatibility

Low-rank works with:
- All integrators (Leapfrog, Yoshida, etc.)
- All topologies (wrapped in low-rank form)
- All physics components (friction, hysteresis, etc.)

### CUDA Optimization

Low-rank has optimized CUDA kernels:
- Fused operations
- Memory-efficient
- Parallel across heads

---

*File: technical/0_architecture/math/geometry/low_rank.md*
*Last Updated: 2026-04-02*

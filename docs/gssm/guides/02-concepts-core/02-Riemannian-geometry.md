# Riemannian Geometry

## Manifolds and Metrics

A Riemannian manifold is a mathematical space that locally behaves like Euclidean space but globally can have curvature. In Manifold, the latent space is modeled as a manifold where the metric g(q) defines the local geometric structure.

The metric is a function that assigns an inner product to each point q in the tangent space. Formally, g: T_q M â†’ â„, where T_q M is the tangent space at q. In practical terms, the metric determines how we measure distances and angles at each point on the manifold.

The choice of metric is crucial because it defines what "near" and "far" mean in representation space. A well-designed metric can capture relevant data structure, while an inadequate metric can obscure important relationships.

In DepthMuun, the geometry is not explicitly derived from an invertible metric matrix. Instead, the "metric" behavior is empirically parameterized via static learned projection matrices ($U$ and $W$). This acts as a pseudo-connection that guides dynamic adaptation to the data structure without the $O(d^3)$ overhead of true metric inversion.

## The Connection Tensor

Rather than computing a symmetric $d \times d$ metric tensor, DepthMuun learns the **Christoffel Connection** directly. The connection determines how the latent velocity state $v$ interacts with itself to produce geometric drag:

- If the generalized connection $\Gamma(x, v)$ is close to zero, the manifold is locally flat and allows high-velocity straight-line paths (exploration).
- If $\Gamma(x, v)$ is large, the manifold exhibits high "curvature," inducing significant geometric resistance and bending the trajectory.

We compute $\Gamma$ via a static parameterization ($U$ and $W$ matrices) combined with a scalar state-dependent Friction Gate. This directly defines the system dynamics without requiring intermediate neural network matrix generation.

## Curvature Bounding

True curvature on a manifold is formally measured by the Riemann tensor $R^k_{ijk}$, which requires fourth-order index computation capturing parallel transport holonomy.

In DepthMuun V2, we **do not** compute the Riemann tensor computationally, as it is completely intractable for sequences of dimension $d=256$ or higher ($O(d^4)$ cost). 

Instead of bounding formal scalar curvature $K$, we directly bound the raw connection vector outputs ($\Gamma$) computationally.

In DepthMuun, we limit effective trajectory bending via an asymptotic projection:

```python
Gamma = CURVATURE_CLAMP * torch.tanh(Gamma / CURVATURE_CLAMP)
```

This prevents the pseudo-connection from becoming numerically unstable or blowing up gradients during long sequence integration, acting as an empirical safeguard against curvature singularities.

## Low-Rank Formulation

Computing formal Christoffel symbols directly from an explicit metric requires $O(d^3)$ operations due to metric inversion and derivative computation. For high-dimensional ML embeddings, this is prohibitive.

DepthMuun utilizes a strictly $O(1)$ memory architecture relying entirely on a **Low-Rank Formula** for the connection:

$$\Gamma(v, x) \approx W \cdot \left[ (U^T v)^2 \odot \sigma(\|U^T v\|_2) \right]$$

Here $U$ and $W$ are $d \times r$ explicit matrices, where $r \ll d$. This approximation reduces the continuous time-step cost to $O(d^2 \cdot r)$. 

The system relies *exclusively* on this empirical low-rank formulation. There is no fallback to "full exact metric" execution, as doing so would violate the strict $O(1)$ hardware design assumptions underlying the Leapfrog symplectic kernel.

## Core Topologies

DepthMuun implements several spatial topologies determining boundary interactions:

- **Toroidal Topology (`type='torus'`)**: The space is bounded with periodic boundaries (e.g., $[-\pi, \pi]$). Particles that exit one side of the representation space re-enter from the opposite side. Toroidal topologies are highly effective at enforcing bounded gradient energy and capturing cyclic sequence semantics naturally (like language context attention limits).
- **Euclidean Topology**: Standard unbounded vector space. Can grow infinitely without hard boundaries, but is more susceptible to exploding trajectories if the conformal friction gate fails to learn appropriate damping.

---

**DepthMuuns (JoaquÃ­n StÃ¼rtz)**

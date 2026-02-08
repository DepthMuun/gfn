# Riemannian Geometry

## Manifolds and Metrics

A Riemannian manifold is a mathematical space that locally behaves like Euclidean space but globally can have curvature. In Manifold, the latent space is modeled as a manifold where the metric g(q) defines the local geometric structure.

The metric is a function that assigns an inner product to each point q in the tangent space. Formally, g: T_q M → ℝ, where T_q M is the tangent space at q. In practical terms, the metric determines how we measure distances and angles at each point on the manifold.

The choice of metric is crucial because it defines what "near" and "far" mean in representation space. A well-designed metric can capture relevant data structure, while an inadequate metric can obscure important relationships.

In Manifold, the metric is not fixed but depends on the state of the system. This means the geometry evolves during processing, enabling dynamic adaptation to data structure.

## Metric Matrix

The metric matrix g(q) is a positive-definite symmetric d×d matrix where d is the embedding dimension. Its elements g_ij(q) define the inner product between basis vectors i and j at point q.

The properties of g(q) determine system behavior:

- If g(q) is close to the identity, the manifold is locally similar to Euclidean space.
- If g(q) has very different eigenvalues, the manifold is anisotropic.
- If g(q) changes rapidly, the manifold has significant curvature.

We compute g(q) via a neural network that takes the current state as input. This network produces a symmetric matrix used to define the system dynamics.

## Curvature

The curvature of a manifold measures how much it deviates from being flat. In high-dimensional manifolds, curvature is a tensor (the Riemann tensor) with multiple components.

The Riemann tensor R^k_ijk has 4 indices and captures the manifold's holonomy: how much a vector rotates when it is parallel-transported along a closed path.

For practical purposes, we use a scalar curvature measure based on the tensor norm:

K = ||R|| / (d * (d-1))

K values close to zero indicate locally flat geometry. Large values indicate significant curvature that affects geodesic trajectories.

In Manifold, we limit effective curvature via CURVATURE_CLAMP. This prevents the metric from becoming singular or numerically unstable.

## Energy-Momentum Tensor

The Einstein tensor emerges naturally when considering metric dynamics. Its divergence encodes how matter (in our case, the embedding state) affects geometry.

G^μν + Λg^μν = T^μν

Where G is the Einstein tensor, Λ is the cosmological constant, and T is the system's energy-momentum tensor. In Manifold, we use a simplified version where T captures the state's "density" in different regions of the manifold.

This connection to general relativity is more than an analogy: it provides a framework for understanding how data structure can "curve" representation space in ways that facilitate processing.

## Low-Rank Factorization

Computing Christoffel symbols directly requires O(d^3) operations due to metric inversion and derivative computation. For high-dimensional embeddings, this is prohibitive.

Low-rank factorization approximates the metric as g(q) = AA^T + σ^2 I where A is d×r with r << d. This approximation reduces the cost to O(d^2 r).

The factorization also provides implicit regularization: metric components with magnitude smaller than σ are ignored. This prevents numerical instability and can improve generalization.

The trade-off is accuracy versus speed. For critical applications, we use the full metric. For scalability, we use the low-rank approximation with small r.

## Implemented Geometries

Manifold implements several geometries with different properties.

Hyperbolic geometry has constant negative curvature. It is appropriate for data with hierarchical structure, such as syntax trees or taxonomies. Hyperbolic space "grows exponentially," enabling efficient representation of hierarchical structures.

Toroidal geometry has curvature that changes sign in different directions. It is appropriate for data with cyclic structure, such as time series or circular coordinates.

Adaptive geometry adjusts its curvature based on the data. It starts with gentle curvature and increases in regions of high variability. It is the most general option but requires more data to converge.

---

**Manifold Labs (Joaquín Stürtz)**

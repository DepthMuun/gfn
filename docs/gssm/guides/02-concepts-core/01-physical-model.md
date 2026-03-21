# Physical Model: Hamiltonian Mechanics

## Theoretical Foundations

The Manifold system models information processing as a Hamiltonian dynamical system. This perspective allows us to apply analytical mechanics tools to neural network design.

In Hamiltonian mechanics, a system is described by generalized coordinates q and conjugate momenta p. Time evolution is determined by Hamilton's equations, which derive from a scalar function H(q,p) called the Hamiltonian. This function represents the total energy of the system.

Hamilton's equations are:

dq/dt = âˆ‚H/âˆ‚p
dp/dt = -âˆ‚H/âˆ‚q

This formulation has attractive properties for model design. Pure Hamiltonian systems preserve volume in phase space (Liouville's theorem); however, DepthMuun operates as a **Conformal Symplectic** system by introducing state-dependent friction. This intentional geometric drag allows phase-space volume to contract precisely where the model needs to converge onto targets or "forget" stale contexts, avoiding the severe stability-plasticity oscillations of pure Hamiltonian models.

In the Manifold context, $q$ coordinates represent the embedding state on the manifold, while $p$ momenta represent processing dynamics (velocity). The Hamiltonian combines potential energy terms (input-dependent forces) and kinetic energy terms.

## Metric and Geometry

The underlying Riemannian manifold is defined by a metric g(q). This metric determines distances and angles in representation space, and fundamentally affects how information flows through the system.

In theoretical differential geometry, kinetic energy takes the form:

$T(p,q) = \frac{1}{2} p^T g(q)^{-1} p$

However, for computational efficiency and O(1) memory scaling in DepthMuun V2, we **do not** maintain an explicit inverse metric tensor $g(q)^{-1}$. Instead, the system treats the base metric as an identity matrix ($g = I$) and computes the connection (Christoffel symbols) directly through a low-rank pseudo-metric approximation.

The potential energy $V(q)$ encodes input information and problem constraints (derived via functional embeddings). The balance between $T$ and $V$, moderated tightly by the conformal friction gate, determines system behavior: energy injection yields exploration, while friction-induced dissipation yields convergence.

## Christoffel Symbols

Christoffel symbols Î“ are the Levi-Civita connections on the manifold. They determine how a vector is parallel transported as we move on the manifold, and they appear naturally in the geodesic equations.

```markdown
\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij} \right)
```

Where $g^{kl}$ are the elements of the inverse metric matrix. Computing $\Gamma$ strictly requires derivatives of the metric, which makes the computation $O(d^3)$ expensive.

In DepthMuun V2, we completely bypass this expensive full derivative calculation. Instead, the model uses an empirical **low-rank approximation**, which factorizes the expected connection tensor directly to reduce the cost from $O(d^3)$ to $O(d^2 \cdot r)$, where $r$ is the factorization rank. This guarantees symmetrical connections computationally optimized for ML gradients, without formally requiring a true invertible metric tensor.

## Physics-Informed Deterministic Modeling

While the namespace includes historical references to Geodesic Flow Networks (GFNs), the DepthMuun V2 architecture is fundamentally a **deterministic differential equation solver**, rather than a probabilistic flow planner.

The system does not evolve a stochastic policy $\pi(x)$ to maximize entropy $H(\pi)$ via variational free energy. Instead, it evolves deterministic state trajectories via rigid ordinary differential equations (Riemannian Geodesics). 

The learning occurs through a **Physics-Informed Loss** mechanism (`PhysicsInformedLoss`). This objective explicitly minimizes deviations from geometric ideals (like Velocity Parity and Adjoint trajectory consistency) while regressing directly onto the target sequences.

## Interpretation as a Neural Network

From a neural network perspective, the Hamiltonian system can be seen as a structured attention mechanism. The q coordinates are the queries, the p momenta are the keys, and the metric defines the attention.

The advantage of this formulation over standard attention is its dynamic structure: the system evolves over time, allowing iterative refinement of representations. The disadvantage is the additional computational cost of the integrator.

We choose symplectic integrators over generic integrators because they better preserve the Hamiltonian structure. Leapfrog, for example, makes O(h^3) error in energy but O(h^2) error in phase-space structure, which yields more stable long-term trajectories.

---

**DepthMuuns (Joaquin Sturtz)**

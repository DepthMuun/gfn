# Geodesic Flow

## Geodesic Definition

A geodesic is the shortest path between two points on a manifold. In physical terms, it is the trajectory a particle free of external forces would follow. Mathematically, it satisfies the geodesic equations:

d²q^k/dt² + Γ^k_ij * dq^i/dt * dq^j/dt = 0

The first term is acceleration in curvilinear coordinates. The second term involves Christoffel symbols and captures the effect of curvature on motion.

In Manifold, the geodesic flow evolves the system state by following these equations. The system naturally "falls" along the manifold toward lower-energy configurations, with the metric determining the shape of the energy landscape.

## Numerical Integration

The geodesic equations are second-order differential equations. To solve them numerically, we convert them to a first-order system by introducing momenta:

dq/dt = p
dp/dt = -Γ(q)(p,p)

Now we have 2d first-order equations that we can integrate with standard methods.

The Leapfrog integrator (also known as Verlet in mechanics) is our primary choice. Its scheme is:

1. Kick: p(t + h/2) = p(t) + (h/2) * dp/dt(q(t))
2. Drift: q(t + h) = q(t) + h * g(q(t))^(-1) * p(t + h/2)
3. Kick: p(t + h) = p(t + h/2) + (h/2) * dp/dt(q(t + h))

This scheme preserves volume in phase space and has O(h^3) energy error, much better than generic integrators for Hamiltonian systems.

## Optimality Properties

Geodesic flow has important theoretical properties that inform the model design.

The first property is optimality: given two points on the manifold, the geodesic is the shortest path. In the model context, this means the system finds efficient representations that minimize "distance" in latent space.

The second property is local uniqueness: near any point, there exists exactly one short geodesic between sufficiently close points. This guarantees the system converges to stable representations rather than oscillating indefinitely.

The third property is reversibility: reversing time transforms one geodesic into another. This symmetry is important for training because it enables a well-behaved backward pass.

## Geodesic Regularization

We add a loss term that forces model trajectories to behave like geodesics. This improves representation quality and prevents pathological behavior.

The geodesic loss term is:

L_geo = ||d²q/dt² + Γ(q)(dq/dt, dq/dt)||²

This term penalizes deviations from geodesic dynamics. Small values indicate the model is "falling naturally" along the manifold rather than moving arbitrarily.

The weight of this regularization is controlled by LAMBDA_G_DEFAULT. High values enforce strict geometry but can limit expressive capacity. Low values allow more freedom but the underlying geometry may not emerge.

## Integration Horizon

The number of integration steps (LEAPFROG_SUBSTEPS) determines how long the system evolves per processed token.

Few iterations (1-2): the system barely moves. The dynamics are similar to standard attention with little refinement.

Moderate iterations (3-5): balance between refinement and computational cost. This is the standard configuration.

Many iterations (>5): the system has time to fully converge, but the cost grows linearly. Useful for analysis but impractical for training.

The optimal choice depends on the task. Tasks that require multi-step reasoning benefit from more iterations. Simple tasks can work with fewer.

## Singularities and Black Holes

The metric can become singular at some points, causing division by zero in the integrator. We prevent this through clamping and normalization.

The "black hole" mechanism saturates the metric in regions of high curvature, preventing the system from escaping or becoming unstable. BLACK_HOLE_STRENGTH controls the intensity of this saturation.

If you observe NaN loss or erratic behavior, the system may have encountered a singularity. Check curvature values and consider increasing SINGULARITY_THRESHOLD.

---

**Manifold Labs (Joaquín Stürtz)**

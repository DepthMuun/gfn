# Geodesic Flow

## Geodesic Definition

A geodesic is the shortest path between two points on a manifold. In physical terms, it is the trajectory a particle free of external forces would follow. Mathematically, it satisfies the geodesic equations:

dÂ²q^k/dtÂ² + Î“^k_ij * dq^i/dt * dq^j/dt = 0

The first term is acceleration in curvilinear coordinates. The second term involves Christoffel symbols and captures the effect of curvature on motion.

In Manifold, the geodesic flow evolves the system state by following these equations. The system naturally "falls" along the manifold toward lower-energy configurations, with the metric determining the shape of the energy landscape.

## Numerical Integration

The geodesic equations are second-order differential equations. To solve them numerically, we convert them to a first-order system by introducing momenta:

dq/dt = p
dp/dt = -Î“(q)(p,p)

Now we have 2d first-order equations that we can integrate with standard methods.

The **Conformal Symplectic Leapfrog** integrator (a modified Verlet scheme handling friction) is our primary choice. Its structure is:

1. **Kick**: $p_{t+h/2} = \frac{p_t + \frac{h}{2} \cdot \left( \frac{dp}{dt}(q_t) + F_{\text{ghost}} \right)}{1 + \frac{h}{2} \mu_t}$
2. **Drift**: $q_{t+h} = \text{Bnd}(q_t + h \cdot p_{t+h/2})$
3. **Kick**: $p_{t+h} = \frac{p_{t+h/2} + \frac{h}{2} \cdot \left( \frac{dp}{dt}(q_{t+h}) + F_{\text{ghost}} \right)}{1 + \frac{h}{2} \mu_{t+h}}$

Unlike generic integrators or pure conservative Leapfrog, this specific scheme rigorously enforces $O(1)$ memory mapping (by omitting expensive inverse metric multiplications on $p$) and explicitly handles topological boundaries $\text{Bnd}(\cdot)$. Most importantly, the $1 + \mu$ denominator correctly contracts phase-space volume seamlessly corresponding to the active friction gate.

## Optimality Properties

Geodesic flow has important theoretical properties that inform the model design.

The first property is optimality: given two points on the manifold, the geodesic is the shortest path. In the model context, this means the system finds efficient representations that minimize "distance" in latent space.

The second property is local uniqueness: near any point, there exists exactly one short geodesic between sufficiently close points. This guarantees the system converges to stable representations rather than oscillating indefinitely.

The third property is reversibility: reversing time transforms one geodesic into another. This symmetry is important for training because it enables a well-behaved backward pass.

## Geodesic Regularization

We add a loss term that forces model trajectories to behave like geodesics. This improves representation quality and prevents pathological behavior.

The geodesic loss term is:

L_geo = ||dÂ²q/dtÂ² + Î“(q)(dq/dt, dq/dt)||Â²

This term penalizes deviations from geodesic dynamics. Small values indicate the model is "falling naturally" along the manifold rather than moving arbitrarily.

The weight of this regularization is controlled by `LAMBDA_G` (default: **$0.00005$**). High values enforce strict geometry but can severely crush the semantic gradients. This scalar was empirically tuned down to very small margins in DepthMuun V2 specifically to perfectly maintain curvature without overpowering the primary sequence-modeling task vectors.

## Integration Horizon

The number of integration steps (`LEAPFROG_SUBSTEPS`) determines how long the system evolves per processed token layer.

- **Fast mapping (1-2)**: The system barely moves. The dynamics are similar to standard attention with little refinement.
- **V2 Standard (3)**: In DepthMuun V2, the base standard is exactly **3** substeps. We optimized this down from historic defaults (5) because empirical testing proved that 3 iterations provide ideal refinement paths while ensuring a significantly cleaner backward pass tracking.
- **Deep Iterations (>3)**: The system has time to fully converge, but the memory tracking cost during `.backward()` grows linearly. Useful for stress-testing representations but impractical for large-scale training.

The optimal choice depends on the task. Tasks that require multi-step reasoning benefit from more iterations. Simple tasks can work with fewer.

## Singularities and Black Holes

The metric can become singular at some points, causing division by zero in the integrator. We prevent this through clamping and normalization.

The "black hole" mechanism dynamically scales geometry saturation in regions of critical curvature, preventing the system from escaping or crashing the numeric float bounds. `BLACK_HOLE_STRENGTH` controls the intensity of this saturation (tuned to **1.5**).

If you observe NaN loss or erratic gradient behavior, the system may have encountered a singularity jump before the protective boundary triggered. The V2 `SINGULARITY_THRESHOLD` has been lowered from historical defaults down to **0.5**, purposefully triggering these protective asymptotes much earlier during volatile initial training phases.

---

**DepthMuuns (Joaquin Sturtz)**

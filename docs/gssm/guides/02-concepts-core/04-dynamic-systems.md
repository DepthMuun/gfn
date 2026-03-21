# Dynamical Systems and Numerical Stability

## Symplectic Integrators

Symplectic integrators are numerical methods designed specifically for Hamiltonian systems. Unlike generic integrators, they preserve the symplectic structure of the flow, which results in better long-term preservation of invariants.

A system is symplectic if it preserves the symplectic form Ï‰ = dq âˆ§ dp. Integrators that preserve this structure have energy error that oscillates rather than growing monotonically, which is crucial for stable training.

The integrators implemented in Manifold include:

Leapfrog (StÃ¶rmerâ€“Verlet): Second-order integrator with excellent energy preservation. It is the most used due to its balance between accuracy and efficiency. Energy error scales as O(h^2) per step, but trajectory error can accumulate.

Yoshida: Fourth-order integrator that uses optimized coefficients to minimize error. More accurate than Leapfrog but with more operations per step. Useful when high precision is required.

Forest-Ruth: Another fourth-order integrator, an alternative to Yoshida with different stability characteristics. Less common but available for experimentation.

Omelyan: Second-order symplectic integrator with optimized coefficients to reduce error. Useful for large timesteps.

## Numerical Stability

Integrator stability depends on several factors that must be balanced.

The timestep (DEFAULT_DT) is the most critical parameter. Large timesteps reduce computational cost but increase discretization error. If the timestep is too large, the integrator can become unstable and the loss can diverge.

Typical DT values range from 0.02 to 0.1. The optimal value depends on the system's timescale. If you observe loss oscillations, reduce the timestep.

Friction in the system acts as an implicit convergence mechanism. In DepthMuun V2, friction is **not a static global scalar**, but an **Active Inference Gate** (`FrictionGate` and `RiemannianGating`). 

The friction coefficient $\mu$ is computed dynamically based on the current position, the velocity magnitude, and the active forces. This intelligent gating allows the system to lower friction when exploring new representations and dramatically increase friction when it mathematically "needs to forget" or lock onto a final sequence target.

Too much base friction (`DEFAULT_FRICTION`) causes premature convergence to local minima. Too little friction causes runaway trajectories and NaN failures since the network cannot brake its momentum.

## Python-CUDA Parity

The implementation has two versions: pure Python and custom CUDA kernels. For consistent results, both must produce the same values.

Parity is achieved through:

1. Identical constants: EPSILON_STANDARD, FRICTION_SCALE, and other values must be the same in Python and CUDA.
2. Operation order: Operations must run in the same order.
3. Data types: float32 must be consistent across both backends.

Deviations can occur due to:
- Differences in reduction operation order between ATen and custom CUDA architectures
- Deep floating-point precision differences on GPU vs CPU during large accumulation arrays
- Errors in constant propagation

Verify parity by using the orchestrator with the `tests/test_cuda_python_consistency.py` module running through PyTest after any change to the integrator code or C++ `gfn_cuda` underlying files.

## Equations of Motion

The Hamiltonian equations of motion with friction are:

dq/dt = âˆ‚H/âˆ‚p
dp/dt = -âˆ‚H/âˆ‚q - Î¼ * p

The -Î¼*p term is friction, where Î¼ is the friction coefficient. This friction dissipates energy and helps the system converge.

Friction can be constant or velocity-dependent. VELOCITY_FRICTION_SCALE introduces velocity dependence: high-velocity regions experience more friction.

```markdown
v_{t+1/2} = \frac{v_t + \frac{h}{2} \cdot \left( F - \Gamma(x_t, v_t) \right)}{1 + \frac{h}{2} \mu}
```

This implicit formulation (dividing by $1 + \frac{h}{2}\mu$) is strictly required to implement stable **Conformal Symplectic** dynamics. A naive explicit formulation ($v + h \cdot (F - \Gamma - \mu v)$) invariably leads to severe numerical instabilities and fails to preserve the necessary phase-space geometry properties during deep trajectory rolls.

## Energy Conservation

In the absence of friction and external forces, the Hamiltonian H(q,p) should be conserved. In practice, discretization error causes energy drift.

Monitor energy conservation by running diagnostics/conservation_audit.py. This script measures Hamiltonian variation along trajectories and reports whether it is within acceptable limits.

If energy conservation is poor, consider:
- Reducing DEFAULT_DT
- Reducing FRICTION_SCALE
- Increasing LEAPFROG_SUBSTEPS
- Checking for spurious external forces

## Stability Conditions

For the system to be numerically stable, several conditions must be satisfied.

The metric matrix must be positive definite. This is enforced through clamping eigenvalues.

Denominators in the integrator must not be close to zero. `EPSILON_STANDARD` prevents division by zero.

Velocities must remain within reasonable limits over time. `VELOCITY_SATURATION` (default: **$100.0$**) compresses extreme velocities dynamically.

If any condition is violated, the system can produce NaNs or erratic behavior. The safety checks in utils/safety.py detect these conditions.

---

**DepthMuuns (JoaquÃ­n StÃ¼rtz)**

# Dynamical Systems and Numerical Stability

## Symplectic Integrators

Symplectic integrators are numerical methods designed specifically for Hamiltonian systems. Unlike generic integrators, they preserve the symplectic structure of the flow, which results in better long-term preservation of invariants.

A system is symplectic if it preserves the symplectic form ω = dq ∧ dp. Integrators that preserve this structure have energy error that oscillates rather than growing monotonically, which is crucial for stable training.

The integrators implemented in Manifold include:

Leapfrog (Störmer–Verlet): Second-order integrator with excellent energy preservation. It is the most used due to its balance between accuracy and efficiency. Energy error scales as O(h^2) per step, but trajectory error can accumulate.

Yoshida: Fourth-order integrator that uses optimized coefficients to minimize error. More accurate than Leapfrog but with more operations per step. Useful when high precision is required.

Forest-Ruth: Another fourth-order integrator, an alternative to Yoshida with different stability characteristics. Less common but available for experimentation.

Omelyan: Second-order symplectic integrator with optimized coefficients to reduce error. Useful for large timesteps.

## Numerical Stability

Integrator stability depends on several factors that must be balanced.

The timestep (DEFAULT_DT) is the most critical parameter. Large timesteps reduce computational cost but increase discretization error. If the timestep is too large, the integrator can become unstable and the loss can diverge.

Typical DT values range from 0.02 to 0.1. The optimal value depends on the system's timescale. If you observe loss oscillations, reduce the timestep.

Friction in the system (DEFAULT_FRICTION, FRICTION_SCALE) acts as an implicit regularizer. It damps oscillations and helps convergence, but it can prevent the system from fully exploring the manifold.

Too much friction causes premature convergence to local minima. Too little friction causes persistent oscillations and unstable training.

## Python-CUDA Parity

The implementation has two versions: pure Python and custom CUDA kernels. For consistent results, both must produce the same values.

Parity is achieved through:

1. Identical constants: EPSILON_STANDARD, FRICTION_SCALE, and other values must be the same in Python and CUDA.
2. Operation order: Operations must run in the same order.
3. Data types: float32 must be consistent across both backends.

Deviations can occur due to:
- Differences in reduction operation order
- Floating-point precision differences on GPU vs CPU
- Errors in constant propagation

Verify parity by running tests/test_cuda_python_consistency.py after any change to the integrator code.

## Equations of Motion

The Hamiltonian equations of motion with friction are:

dq/dt = ∂H/∂p
dp/dt = -∂H/∂q - μ * p

The -μ*p term is friction, where μ is the friction coefficient. This friction dissipates energy and helps the system converge.

Friction can be constant or velocity-dependent. VELOCITY_FRICTION_SCALE introduces velocity dependence: high-velocity regions experience more friction.

The integrator implementation with friction is:

v_new = (v + h * (force - friction)) / (1 + h * mu)

This implicit form is more stable than the explicit form v + h*(force - friction - mu*v) because it prevents division by small numbers.

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

Denominators in the integrator must not be close to zero. EPSILON_STANDARD prevents division by zero.

Velocities must remain within reasonable limits. VELOCITY_SATURATION compresses extreme velocities.

If any condition is violated, the system can produce NaNs or erratic behavior. The safety checks in utils/safety.py detect these conditions.

---

**Manifold Labs (Joaquín Stürtz)**

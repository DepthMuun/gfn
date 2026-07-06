# Integrators

This document describes the **current integrator runtime** used by GSSM.

The authoritative files are:

- `gfn/realizations/gssm/physics/integrators/base.py`
- `gfn/realizations/gssm/physics/integrators/factory.py`
- `gfn/realizations/gssm/physics/integrators/symplectic/`
- `gfn/realizations/gssm/physics/integrators/runge_kutta/`
- `gfn/realizations/gssm/physics/integrators/adaptive.py`

## Factory Behavior

The integrator factory currently reads:

- `config.stability.integrator_type`

and defaults to:

- `leapfrog`

If the requested key is unknown, it falls back to `leapfrog`.

This is the current effective runtime default, so the docs should not claim `yoshida` or another solver as the default.

## Base Integrator Contract

All current integrators inherit from `BaseIntegrator`.

That base class provides three important behaviors:

- velocity saturation or clamping,
- torus-aware position wrapping,
- access to physics-engine acceleration and friction helpers.

### Velocity saturation

In the current runtime this belongs to the **integrator layer**, not the engine.

If:

- `stability.velocity_saturation > 0`

then the base integrator uses differentiable tanh saturation:

```text
v_sat = tanh(v / sat) * sat
```

Otherwise it falls back to hard clamping.

### Topology resolution

For torus:

```text
x -> atan2(sin(x), cos(x))
```

For Euclidean:

- identity

So topology wrapping is currently standardized at the integrator helper level as well.

## Available Integrators In The Current Factory

The current factory explicitly imports and registers:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

## `leapfrog`

`LeapfrogIntegrator` is the current default and the most important training path.

Its runtime behavior is slightly more sophisticated than the textbook one because it also resolves friction explicitly during the split update.

The slow Python fallback effectively does:

1. resolve friction at the current state,
2. compute acceleration,
3. perform a half-step velocity update,
4. clamp velocity,
5. drift position and wrap topology,
6. recompute friction and acceleration,
7. average the two acceleration estimates,
8. finish the velocity update,
9. clamp again.

So the current leapfrog path is not a pure frictionless textbook Störmer-Verlet implementation; it is a symplectic-style solver adapted to the engine's explicit damping path.

### CUDA fast path

For specific low-rank CUDA-compatible cases, leapfrog can switch to a fused kernel:

- low-rank geometry,
- CUDA available,
- external force present,
- tensors on CUDA.

Otherwise it falls back to the Python implementation.

## `adaptive`

The current adaptive integrator is **not** an embedded error-estimator RK solver.

Instead, it:

1. computes local acceleration norm,
2. sets

```text
dt_eff = base_dt / (1 + alpha * ||accel||)
```

3. clamps `dt_eff` to `[dt_min, base_dt]`,
4. uses the mean effective timestep across the batch,
5. delegates the actual step to a base solver.

The underlying base solver is configured by:

- `stability.base_solver`

and defaults to:

- `verlet`

So this adaptive path is best understood as a **dt modulation wrapper** around another solver, not as a fully separate high-order adaptive solver family.

## Other Integrators

The factory registers additional symplectic and Runge-Kutta integrators:

- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`

These remain valid runtime options, but the most important docs for day-to-day behavior should anchor around:

- `leapfrog` as the current default,
- `adaptive` as the timestep wrapper path.

## Relationship To Dynamic Time Plugin

The `DynamicTimePlugin` and the `adaptive` integrator are different mechanisms.

- `adaptive` changes `dt` from acceleration magnitude at solver level,
- `DynamicTimePlugin` changes `dt` through learned per-head gating before `integrator.step(...)`.

They can interact, so docs should not merge them into one concept.

## Practical Guidance

Use `leapfrog` when:

- you want the standard runtime path,
- you want the most battle-tested default,
- you care about stable geometry-aware training.

Use `adaptive` when:

- you explicitly want timestep shrinkage in high-acceleration regions,
- you are comfortable with a base-solver wrapper rather than a pure standalone solver.

Use the higher-order symplectic solvers when:

- you want more expensive long-horizon trajectories,
- you are intentionally trading cost for trajectory quality.

Use RK-style solvers when:

- you are experimenting with non-symplectic alternatives,
- you do not need the default symplectic bias of the main runtime path.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- `yoshida` is the default integrator,
- adaptive integration uses the classical two-half-step error-estimation scheme shown in many textbooks,
- velocity saturation is a physics-engine responsibility.

Those claims do not match the current implementation.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/base.py`
- `gfn/realizations/gssm/physics/integrators/factory.py`
- `gfn/realizations/gssm/physics/integrators/symplectic/leapfrog.py`
- `gfn/realizations/gssm/physics/integrators/adaptive.py`
- `docs/gssm/guides/03-reference/03-integrators.md`

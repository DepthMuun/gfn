# Velocity Verlet Integrator

This document describes the **current `VerletIntegrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/symplectic/verlet.py`

## What It Is In The Current Runtime

`VerletIntegrator` is a second-order symplectic solver that updates:

- position with explicit `x + v dt + 0.5 a dt^2`,
- then velocity using the average of two acceleration evaluations.

It is closely related to leapfrog, but the current implementation is not identical to the repo's friction-aware leapfrog path.

## Current Step Pattern

The current code performs:

1. compute initial acceleration `a0`,
2. update position with the quadratic term,
3. wrap topology,
4. compute an average velocity `v_avg`,
5. compute new acceleration `a1` at the updated position,
6. update velocity with `0.5 * (a0 + a1) * dt`,
7. clamp velocity.

In code form, the core update is:

```text
a0 = accel(x, v)
x' = resolve_topology(x + v*dt + 0.5*a0*dt^2)
v_avg = v + 0.5*a0*dt
a1 = accel(x', v_avg)
v' = clamp_velocity(v + 0.5*(a0 + a1)*dt)
```

## Topology Handling

Like the other integrators, Verlet uses the shared base helper:

- torus -> wrapped angular coordinates,
- Euclidean -> identity.

So topology handling is part of the actual runtime step, not something external to the integrator.

## Velocity Handling

Velocity is passed through `_clamp_velocity(...)` at the final update.

That means Verlet inherits the same runtime behavior as the shared base class:

- differentiable tanh saturation when `velocity_saturation > 0`,
- otherwise hard clamping.

## Relationship To Leapfrog

The docs should be careful here.

It is reasonable to call Verlet and leapfrog closely related second-order symplectic methods, but in the current runtime they are **not** implemented the same way:

- `LeapfrogIntegrator` has an explicit friction-aware split update and averaging path,
- `VerletIntegrator` uses the more direct `x + v dt + 0.5 a dt^2` form.

So "mathematically related" is accurate.
Line-by-line interchangeability in the current code is not.

## Practical Interpretation

Use Verlet when:

- you want a symplectic second-order solver,
- you want the more explicit position update form,
- you want a simpler alternative to the default leapfrog path.

Compared with leapfrog, the current Verlet implementation is:

- simpler,
- less specialized,
- not the primary documented default.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- the current runtime Verlet is literally the same implementation as leapfrog,
- Verlet has the same explicit friction-correction path as leapfrog,
- topology wrapping is absent from the current step.

Those claims do not match the code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/symplectic/verlet.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/integrators/leapfrog.md`

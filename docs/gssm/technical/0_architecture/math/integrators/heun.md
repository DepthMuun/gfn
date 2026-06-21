# Heun Integrator

This document describes the **current `HeunIntegrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/runge_kutta/heun.py`

## What It Is In The Current Runtime

`HeunIntegrator` is the second-order explicit trapezoidal or predictor-corrector solver in the current factory.

It is not symplectic, but it is fully integrated with the same runtime helpers used by the other solvers:

- topology resolution,
- velocity clamping,
- physics-engine acceleration evaluation.

## Current Step Pattern

The implementation performs:

1. compute `k1` acceleration at the current state,
2. build an Euler-style predicted state,
3. resolve topology for the predicted position,
4. clamp the predicted velocity,
5. compute `k2` acceleration at the predicted state,
6. apply the trapezoidal corrector,
7. resolve topology again for the corrected position,
8. clamp velocity again for the corrected velocity.

In code form, the current path is:

```text
k1_a = accel(x, v)
k1_v = v

x_pred = resolve_topology(x + dt * k1_v)
v_pred = clamp_velocity(v + dt * k1_a)

k2_a = accel(x_pred, v_pred)
k2_v = v_pred

x' = resolve_topology(x + 0.5 * dt * (k1_v + k2_v))
v' = clamp_velocity(v + 0.5 * dt * (k1_a + k2_a))
```

## Runtime Interpretation

So the most faithful description is:

- second-order predictor-corrector,
- non-symplectic,
- topology-aware,
- velocity-saturation-aware through the shared base helper.

## Relationship To Euler

Heun is the runtime's explicit improvement over a pure Euler-style update:

- it predicts once,
- then corrects with an averaged slope.

That is still true conceptually, but the current implementation also includes:

- torus wrapping at prediction and correction,
- shared velocity safety behavior.

## Relationship To Leapfrog

Compared to leapfrog:

- Heun is non-symplectic,
- leapfrog is the default training path,
- both are second-order,
- Heun is structurally simpler but lacks the symplectic bias of the main solver family.

So the docs should present Heun as a legitimate alternative, but not as the recommended default for the main runtime.

## When To Use It

Use Heun when:

- you want a simple non-symplectic second-order solver,
- you are comparing RK-style and symplectic behavior,
- quick experiments matter more than using the default training path.

It is less attractive when:

- you want the standard documented path,
- long-horizon symplectic behavior matters,
- leapfrog already satisfies the need.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- Heun ignores topology in the current runtime,
- Heun does not apply shared velocity control,
- the current implementation is just the abstract trapezoidal rule with no runtime safety helpers.

Those claims do not match the code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/runge_kutta/heun.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/02_integrators.md`

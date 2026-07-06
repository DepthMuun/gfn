# RK4 (Runge-Kutta 4th Order)

This document describes the **current `RK4Integrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/runge_kutta/rk4.py`

## What It Is In The Current Runtime

`RK4Integrator` is the classic non-symplectic fourth-order Runge-Kutta solver adapted to the shared GSSM runtime helpers.

That means the current implementation includes:

- topology resolution at intermediate position proposals,
- velocity clamping on intermediate velocity proposals,
- final topology resolution and final velocity clamping.

So it is not just a bare mathematical tableau detached from the rest of the runtime.

## Current Step Pattern

For each step, the code computes:

- `k1` from the current state,
- `k2` from midpoint state built from `k1`,
- `k3` from midpoint state built from `k2`,
- `k4` from endpoint state built from `k3`.

The actual runtime path is:

```text
k1_v = v
k1_a = accel(x, v)

k2_v_val = v + h/2 * k1_a
k2_x_val = resolve_topology(x + h/2 * k1_v)
k2_a = accel(k2_x_val, clamp_velocity(k2_v_val))

k3_v_val = v + h/2 * k2_a
k3_x_val = resolve_topology(x + h/2 * k2_v)
k3_a = accel(k3_x_val, clamp_velocity(k3_v_val))

k4_v_val = v + h * k3_a
k4_x_val = resolve_topology(x + h * k3_v)
k4_a = accel(k4_x_val, clamp_velocity(k4_v_val))
```

Then it applies the standard RK4 weighted update and again:

- resolves topology for `x`,
- clamps velocity for `v`.

## Important Runtime Detail

The code stores:

- `k2_v = k2_v_val`
- `k3_v = k3_v_val`
- `k4_v = k4_v_val`

for the final weighted position update, while the acceleration calls use the clamped versions.

So the runtime behavior is slightly more nuanced than a pure unclamped textbook RK4 description.

## Relationship To Symplectic Solvers

RK4 is the main non-symplectic fourth-order option in the current factory.

Compared to the symplectic family:

- it favors local accuracy,
- it does not preserve the symplectic structure,
- it still participates in the same topology- and velocity-safety scaffolding.

So the docs should present it as:

- non-symplectic but runtime-integrated,
- not as a completely separate world from the other solvers.

## When To Use It

Use RK4 when:

- you explicitly want a non-symplectic alternative,
- short-horizon local accuracy matters,
- you are comparing solver families.

It is less attractive when:

- you want the main default path,
- long-term symplectic behavior matters,
- you are already satisfied with higher-order symplectic solvers.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- RK4 ignores topology in the current runtime,
- RK4 never applies shared velocity control,
- the current implementation is just the raw textbook formula with no safety helpers.

Those claims do not match the code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/runge_kutta/rk4.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/02_integrators.md`

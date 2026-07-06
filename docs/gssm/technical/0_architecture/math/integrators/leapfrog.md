# Leapfrog Integrator

This document describes the **current `LeapfrogIntegrator` implementation** used by GSSM.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/symplectic/leapfrog.py`

## What It Is In The Current Runtime

`LeapfrogIntegrator` is the current default integrator selected by the factory.

It is still a symplectic kick-drift-kick style solver, but the present runtime implementation is not just the bare textbook formula. It includes:

- explicit friction handling,
- velocity saturation through `BaseIntegrator`,
- torus-aware topology wrapping,
- an optional fused CUDA fast path for low-rank geometries.

## Current Slow-Path Algorithm

In the Python fallback path, one step effectively does:

1. resolve friction coefficient at the current state,
2. compute acceleration,
3. build a non-friction acceleration estimate,
4. perform a half-kick velocity update with damping in the denominator,
5. clamp velocity,
6. drift position and wrap topology,
7. resolve friction again at the updated position,
8. recompute acceleration using `v_half`,
9. average the two acceleration estimates,
10. finish the velocity update with averaged friction,
11. clamp velocity again.

That is more faithful to the current code than the frictionless textbook version.

## Friction-Aware Update

The current update uses:

```text
a1_nf = a1 + mu1 * v
v_half = (v + 0.5 * dt * a1_nf) / (1 + 0.5 * dt * mu1)
```

and later:

```text
a2_nf = a2 + mu2 * v_half
a_avg = (a1_nf + a2_nf) / 2
mu_avg = (mu1 + mu2) / 2
v_next = (v + dt * a_avg) / (1 + dt * mu_avg)
```

So the leapfrog solver is currently adapted to the engine's centralized friction design rather than pretending the system is frictionless.

## Topology Handling

After the drift update, the integrator resolves topology through the shared helper:

- torus -> `atan2(sin(x), cos(x))`
- Euclidean -> identity

This means position wrapping is part of the actual leapfrog runtime path.

## Velocity Saturation

Velocity clamping or saturation also comes from the shared base helper.

If:

- `stability.velocity_saturation > 0`

then the clamp is differentiable:

```text
tanh(v / sat) * sat
```

Otherwise the fallback is hard clamping.

So leapfrog inherits this runtime safety mechanism automatically.

## CUDA Fast Path

There is an optimized fused CUDA path when:

- CUDA extensions are available,
- geometry is `LowRankRiemannianGeometry` or `PaperLowRankRiemannianGeometry`,
- external force is present,
- tensors are on CUDA.

That fused path bundles:

- low-rank geometry tensors,
- friction parameters,
- velocity scaling,
- velocity saturation,
- singularity parameters,
- and gate parameters.

This is important because the current performance characteristics of leapfrog depend strongly on whether that path is available.

## Practical Properties

In the current runtime, leapfrog is still the best default description for:

- stable geometry-aware training,
- standard GSSM sequence evolution,
- the most battle-tested solver path in the codebase.

The most accurate high-level summary is:

- symplectic-style,
- friction-aware,
- topology-aware,
- optionally CUDA-fused.

## Relationship To Verlet

The repo also contains `VerletIntegrator`, but the current implementations are not literally the same code path.

`VerletIntegrator` uses:

- explicit position update with `x + v dt + 0.5 a dt^2`
- then recomputes acceleration and updates velocity

`LeapfrogIntegrator` uses:

- the half-kick / drift / corrected-kick pattern,
- plus explicit friction averaging.

So they are related symplectic methods, but this doc should not oversimplify them as interchangeable line-for-line implementations.

## When To Use It

Use leapfrog when:

- you want the current default,
- you care about stable training,
- you want the path most aligned with the current docs and factory behavior.

Consider other solvers when:

- you need a higher-order symplectic method such as Yoshida,
- you explicitly want the `adaptive` wrapper path,
- you are testing non-symplectic alternatives.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- the current runtime leapfrog is just the exact textbook frictionless formula,
- it has no topology-aware wrapping,
- it has no CUDA specialization,
- it is identical in implementation to the repo's Verlet integrator.

Those claims do not match the current code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/symplectic/leapfrog.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `gfn/realizations/gssm/physics/engine.py`
- `docs/gssm/technical/0_architecture/math/02_integrators.md`

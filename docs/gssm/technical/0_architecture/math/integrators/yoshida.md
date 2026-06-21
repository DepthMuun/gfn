# Yoshida Integrator

This document describes the **current `YoshidaIntegrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/symplectic/yoshida.py`

## What It Is In The Current Runtime

`YoshidaIntegrator` is a fourth-order symplectic solver with:

- explicit Yoshida coefficients,
- shared topology wrapping,
- shared velocity clamping,
- an optional fused CUDA fast path for low-rank geometries.

So the current implementation is not just a pure educational composition formula; it also contains runtime specialization and fallback behavior.

## Coefficients Used By The Code

The runtime sets:

```text
w1 =  1.3512071919596576
w0 = -1.7024143839193153

c1 = c4 = w1 / 2
c2 = c3 = (w0 + w1) / 2
d1 = d3 = w1
d2 = w0
```

These match the standard Yoshida fourth-order composition.

## Current Slow-Path Step Pattern

In the Python fallback path, each full step does:

1. drift with `c1`, compute acceleration, kick with `d1`,
2. drift with `c2`, compute acceleration, kick with `d2`,
3. drift with `c3`, compute acceleration, kick with `d3`,
4. final drift with `c4`,
5. apply topology resolution after each drift and velocity clamping after each kick.

So the current runtime is:

- symplectic,
- topology-aware,
- velocity-saturation-aware through the base class.

## CUDA Fast Path

The current code has a fused fast path when:

- CUDA extensions are available,
- `yoshida_fused` exists,
- geometry is low-rank or paper-low-rank,
- external force is present,
- `x` is on CUDA.

That path passes through:

- low-rank tensors `U` and `W`,
- clamp values,
- friction configuration,
- velocity friction scaling,
- velocity saturation,
- friction-gate parameters,
- singularity settings,
- trace-normalization flag,
- paper-vs-base low-rank flag.

This matters because the real runtime performance and numerical behavior can differ substantially between the fused path and the Python fallback.

## Fallback Behavior

If the fused path is not available, the class emits a one-time warning and falls back to the explicit Python loop.

So the docs should not describe the CUDA fast path as unconditional.

## Relationship To Leapfrog

Compared to leapfrog, Yoshida in the current runtime offers:

- higher formal order,
- more sub-steps,
- higher cost,
- similar shared safety helpers for topology and velocity control.

But it does **not** reuse the leapfrog-specific explicit friction-averaging scheme.

Instead, it repeatedly calls the shared acceleration helper at each sub-step.

## When To Use It

Use Yoshida when:

- you want a higher-order symplectic solver,
- you accept more cost than leapfrog,
- long-horizon trajectory quality matters more than default simplicity.

It is less attractive when:

- you want the main documented default,
- training cost is the top priority,
- you rely on the default leapfrog path already being sufficient.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- Yoshida is the default integrator,
- the current implementation is only a pure textbook composition with no runtime specialization,
- it has no fast path,
- it uses the same friction-correction path as leapfrog.

Those claims do not match the current code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/symplectic/yoshida.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/integrators/leapfrog.md`

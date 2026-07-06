# Curiosity

This document describes the **current `GeometricCuriosityForce` runtime**.

The authoritative code is:

- `gfn/realizations/gssm/physics/components/curiosity.py`
- `gfn/realizations/gssm/physics/engine.py`

## What It Does

The current curiosity module adds a deterministic repulsive force away from the batch center.

Its current role is:

- encourage exploration,
- push states away from dense aggregates,
- reduce collapse toward a shared latent center.

It is not a full density-estimation system; the current implementation explicitly uses the batch center as a simple attractor proxy.

## Core Runtime Path

The current forward path is:

1. compute a batch center,
2. compute the direction away from that center,
3. scale force inversely with squared distance,
4. clamp the resulting force to keep it bounded.

## Toroidal Behavior

When `topology == torus`, the module uses:

- circular mean through `atan2(mean(sin), mean(cos))`,
- wrapped angular differences through `atan2(sin(d), cos(d))`.

So the torus path is topology-aware and uses circular rather than Euclidean averaging.

## Euclidean Behavior

When topology is not toroidal, the module uses:

- ordinary batch mean,
- raw Euclidean direction away from that mean.

## Magnitude Rule

The current implementation computes:

```text
dist_sq = sum(direction^2) + 1e-6
repulsion_mag = strength / dist_sq
force = direction / sqrt(dist_sq) * repulsion_mag
```

Then it clamps force to:

```text
[-5.0, 5.0]
```

So the live runtime behavior is a bounded inverse-square-style repulsion.

## Engine Integration

The physics engine adds curiosity when:

- the curiosity config is enabled,
- and `curiosity_module` has been instantiated.

Then:

```text
net_accel += curiosity_module(x, v, **kwargs)
```

Important current detail:

- `v` is part of the signature for API consistency,
- but the current curiosity computation is driven by `x` and topology, not velocity dynamics directly.

## Configuration Reality

The schema currently exposes:

- `enabled`
- `strength`
- `decay`

Important current caveat:

- `strength` is actively used,
- `decay` is stored on the module but is not used in the current `forward()` implementation.

So the docs should not imply that curiosity currently decays over time just because the config field exists.

## Practical Interpretation

The current curiosity module is best understood as:

- deterministic exploration pressure,
- topology-aware center repulsion,
- lighter-weight and more structured than random stochasticity,
- but still relatively simple compared with a true learned exploration model.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- curiosity is driven by a full density model in the current runtime,
- `decay` actively changes the force over time in the current implementation,
- curiosity fundamentally depends on velocity in the present formula.

Those claims do not match the code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/components/curiosity.py`
- `gfn/realizations/gssm/physics/engine.py`
- `docs/gssm/technical/0_architecture/math/06_stochasticity.md`

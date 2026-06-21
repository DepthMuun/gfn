# Dynamics

This document describes the **current dynamics-routing runtime**.

The authoritative code is:

- `gfn/realizations/gssm/physics/dynamics/__init__.py`
- `gfn/realizations/gssm/physics/dynamics/base.py`
- `gfn/realizations/gssm/physics/dynamics/*.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

## What Dynamics Means In The Current Runtime

Dynamics modules decide how the layer moves from:

- `current_state`

to:

- `absolute_proposal`

after the integrator and mixer have produced a candidate update.

So dynamics is the routing layer between:

- integrated mixed proposal,
- and final next state.

## Current Registry

The current runtime registry supports:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

Important correction:

- older docs that only listed `direct`, `residual`, and `gated` are incomplete.

## Shared Base Contract

All current dynamics modules inherit from `BaseDynamics`.

That base class provides:

- topology-aware `_apply_norm(...)`

with behavior:

- torus -> wrap through `atan2(sin(x), cos(x))`
- otherwise -> apply the injected normalization layer

Important current detail:

- `context_x` can be passed into the normalization layer,
- which matters for metric-aware normalization behavior.

## `direct`

`DirectDynamics` simply returns:

```text
norm(absolute_proposal)
```

This is the simplest routing mode and remains the effective default behavior.

## `residual`

`ResidualDynamics` computes a residual between proposal and current state, normalizes that residual, then applies:

```text
current_state + sigmoid(residual_scale) * residual_normalized
```

For torus:

- the residual is a wrapped angular difference.

Important current detail:

- `residual_scale` is a learnable parameter initialized from `residual_scale=0.1`.

## `gated`

`GatedDynamics` builds a learned gate from:

```text
[current_state, absolute_proposal]
```

then mixes:

```text
g * proposal + (1 - g) * current
```

and normalizes the result.

This is the explicit state-dependent routing path.

## `mix`

`MixDynamics` is a real runtime mode and should be documented explicitly.

It uses:

- a learnable `log_alpha`,
- a learnable `change_scale`.

For Euclidean:

- interpolate linearly between current and proposal.

For torus:

- interpolate in circular form through `sin/cos` averaging,
- then apply a wrapped difference and scaled update.

So `mix` is not just a synonym for gated or residual dynamics; it is its own interpolation-based routing mechanism.

## `stochastic`

`StochasticDynamics` adds learnable noise on top of either:

- the pure proposal,
- or a residual-style base depending on `mode`.

It uses:

- a learnable `sigma`,
- `softplus(sigma) + 1e-6`,
- random Gaussian noise,
- then topology-aware normalization.

So this is a true dynamics-level stochastic routing mode, distinct from the physics-engine stochasticity module.

## Position Vs Velocity Routing

In `ManifoldLayer`, the runtime creates:

- `dynamics_x` with the configured topology,
- `dynamics_v` with Euclidean topology.

That means:

- position routing is manifold-aware,
- velocity routing is always treated in tangent-space Euclidean form.

This distinction is central to the current implementation.

## Configuration Reality

The effective dynamics type can come from:

- `config.dynamics.type`
- or fallback `dynamics_type`

with `ManifoldLayer` resolving the final choice.

So the docs should describe dynamics choice as a resolved runtime setting, not just one isolated config field.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- only three dynamics modes exist,
- `mix` is just an alias for `gated`,
- velocity dynamics use toroidal routing in the current layer path.

Those claims do not match the runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/dynamics/__init__.py`
- `gfn/realizations/gssm/physics/dynamics/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

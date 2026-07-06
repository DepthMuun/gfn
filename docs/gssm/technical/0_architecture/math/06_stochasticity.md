# Stochasticity

This document describes the **current stochastic-force runtime**.

The authoritative code is:

- `gfn/realizations/gssm/physics/components/stochasticity.py`
- `gfn/realizations/gssm/physics/engine.py`

## What Exists In The Current Runtime

The engine can attach one stochastic module through:

- `BrownianForce`
- `OUDynamicsForce`

based on:

- `active_inference.stochasticity.enabled`
- `active_inference.stochasticity.type`

The engine only adds stochastic force when:

- a stochastic module exists,
- and `dt` is passed to `compute_acceleration(...)`.

That `dt` requirement is important and should be explicit in the docs.

## `BrownianForce`

`BrownianForce` returns:

```text
noise = randn_like(v) * sigma * dt^(-1/2)
```

with `dt` clamped to a safe minimum.

Important current runtime details:

- invalid or non-positive `dt` yields zero force,
- `x` is accepted for API compatibility but not used in the actual computation.

So the current Brownian path is:

- isotropic,
- instantaneous,
- white-noise-like,
- scaled so the integrated effect behaves like `sqrt(dt)`.

## `OUDynamicsForce`

`OUDynamicsForce` keeps internal state in:

- `_prev_noise`

and updates it with:

```text
next_noise = prev_noise
           + theta * (mu - prev_noise) * dt
           + sigma * dt^(-1/2) * randn
```

Important current detail:

- despite the OU interpretation, the code still uses the same `dt^(-1/2)` scaling convention as the Brownian implementation because the output is treated as a force-like term added to acceleration.

It also exposes:

- `reset()`

to clear the stored OU state.

## Engine Integration

In the current engine path:

```text
if stochasticity_module is not None and dt is not None:
    net_accel += stochasticity_module(x, v, dt)
```

So stochasticity is:

- additive,
- optional,
- timestep-dependent,
- and absent if the caller does not supply `dt`.

## Configuration Reality

The schema currently exposes:

- `enabled`
- `type`
- `sigma`
- `theta`
- `mu`

These are all meaningfully used by the current stochasticity path.

`theta` and `mu` matter only for:

- `type == "ou"`

## Practical Interpretation

Use Brownian stochasticity when:

- you want simple isotropic exploration noise.

Use OU stochasticity when:

- you want correlated noise with internal temporal state.

Important current caveat:

- both are still force-like perturbations added inside the physics engine,
- not standalone diffusion solvers outside the main acceleration path.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- stochasticity is active even when `dt` is omitted,
- the engine always uses stochasticity whenever the config subtree exists,
- OU noise is stateless in the current implementation.

Those claims do not match the runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/components/stochasticity.py`
- `gfn/realizations/gssm/physics/engine.py`
- `docs/gssm/technical/0_architecture/math/01_physics_engine.md`

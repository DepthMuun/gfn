# GSSM Physics Engine

This document describes the current `ManifoldPhysicsEngine` path used by GSSM.

For integrator details, see:

- `technical/0_architecture/math/01_physics_engine.md`
- `technical/0_architecture/math/02_integrators.md`
- `technical/0_architecture/math/08_dynamics.md`

## Core Role

`ManifoldPhysicsEngine` is the runtime component that turns geometry and optional physical modules into an acceleration update consumed by the integrator.

In the current implementation, the engine is the central authority on:

- reading geometry output
- combining friction
- adding optional secondary forces
- applying singularity damping at the end

## Constructor Inputs

The engine is initialized with:

- a geometry module
- an optional `PhysicsConfig`
- an optional `dim`
- `heads`

During initialization, it also derives:

- whether the topology is toroidal
- fallback friction
- velocity-friction scale
- velocity saturation reference
- optional singularity gate
- optional hysteresis module
- optional stochasticity module
- optional curiosity module

## Acceleration Path

The current `compute_acceleration(...)` flow is:

1. call `geometry(x, v, force=force, **kwargs)`
2. unpack either `gamma` or `(gamma, mu_geo)`
3. compute total friction with `get_friction_coefficient(...)`
4. build the base acceleration
5. add external force if present
6. add hysteresis ghost force if enabled
7. add stochasticity if enabled and `dt` is available
8. add curiosity if enabled
9. apply singularity damping if enabled and a metric component is provided

At a high level:

```text
net_accel = -gamma - mu_total * v + force + optional_modules
```

## Friction Handling

The engine is intentionally the single place where friction is combined.

Current rule:

```text
mu_total = friction_fallback + mu_geo
```

and then, if `velocity_friction_scale > 0`, the engine scales `mu_total` with the velocity norm.

This is a correction of older documentation that implied friction always came from a universal active-inference gate. In the current runtime, the base config friction still matters directly.

## Optional Modules

### Hysteresis

Enabled through `config.hysteresis.enabled`.

When active, the engine creates a `HysteresisModule` and adds its ghost force to the acceleration.

### Stochasticity

Configured under `config.active_inference.stochasticity`.

The engine supports:

- Brownian noise
- OU dynamics noise

Important runtime caveat:

- stochasticity is only applied when enabled and when the engine receives `dt`

### Curiosity

Configured under `config.active_inference.curiosity`.

The engine instantiates `GeometricCuriosityForce` when enabled and adds its output to the acceleration path.

### Singularities

Configured under `config.singularities`.

If enabled, the engine creates a `SingularityGate` and applies damping after all other force contributions have been accumulated.

## What The Engine Does Not Do

Several older docs blurred the boundaries between the engine and nearby systems.

In the current runtime:

- integrators handle timestep stepping and topology wrapping
- velocity saturation is an integrator-side behavior, not the engine's main update rule
- model-level hooks and readout logic live outside the physics engine
- metric-aware normalization is handled by separate normalization components, not by a generic trace-normalization step directly inside `compute_acceleration(...)`

## Boundary Helpers

The engine does expose:

- `validate_state(x, v)`
- `apply_singularity_damping(v, metric_component)`
- `get_ghost_force(x, v)`
- `get_friction_coefficient(x, v, ...)`
- `reset_hysteresis()`
- `apply_boundary(x)`

`apply_boundary(x)` currently wraps toroidal position with:

```python
torch.atan2(torch.sin(x), torch.cos(x))
```

## Practical Caveats

- Do not document the engine as if it directly performs the whole forward pass.
- Do not attribute integrator responsibilities to the engine.
- Do not assume stochasticity or curiosity are part of the default path.
- Do not assume geometry returns only curvature without friction.

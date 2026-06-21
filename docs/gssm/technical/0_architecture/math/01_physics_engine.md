# Physics Engine

This document describes the **current `ManifoldPhysicsEngine` runtime**, not a generic idealized physics stack.

The authoritative implementation lives in:

- `gfn/realizations/gssm/physics/engine.py`
- `gfn/realizations/gssm/physics/components/`

## What The Engine Actually Computes

The engine computes a net acceleration tensor from:

- geometry curvature output,
- friction,
- optional external force,
- optional hysteresis ghost force,
- optional stochastic force,
- optional curiosity force,
- optional singularity damping.

In the current code, the core pattern is:

```text
net_accel = -christoffel - friction_term
net_accel += force
net_accel += ghost_force
net_accel += stochastic_force
net_accel += curiosity_force
net_accel = singularity_gate.damp_force(net_accel, metric_component)   # only if provided
```

## Geometry Contract

The engine expects the geometry to return either:

- `gamma`
- or `(gamma, mu_geo)`

where:

- `gamma` is the curvature contribution,
- `mu_geo` is an optional geometry-provided friction coefficient or friction gate.

This is important because the engine is now the **single authority** on how friction is actually applied.

## Friction Path

The current friction path is:

```text
mu_total = friction_fallback + mu_geo
friction_term = mu_total * v
```

with optional velocity scaling:

```text
mu_total = mu_total * (1 + velocity_friction_scale * ||v|| / sqrt(D))
```

So in the present runtime:

- geometry may provide a friction signal,
- config still provides the fallback base friction,
- the engine sums them,
- and only then applies damping.

This means the docs should not describe friction as purely geometry-owned or purely config-owned.

## External Force

The engine treats `force` as an already prepared external signal.

In normal sequence models that force often comes from the embedding path, but the engine itself does not know where it came from. It only receives a tensor and adds it to the acceleration.

So the most faithful statement is:

- the engine consumes external force,
- force generation belongs elsewhere in the model stack.

## Optional Modules

### Hysteresis

If enabled, the engine instantiates `HysteresisModule` and adds:

```python
ghost_force = self.hysteresis(x, v, topo_id=self.topo_id)
```

This is the engine's memory-like residual force path.

### Stochasticity

The current runtime supports optional active-inference stochasticity through:

- `BrownianForce`
- `OUDynamicsForce`

Important current caveat:

- stochastic force is only added when both the module is enabled and `dt` is provided to `compute_acceleration`.

### Curiosity

If enabled, the engine adds `GeometricCuriosityForce`.

This is a modular exploration term, not a hardwired part of every acceleration computation.

### Singularity Damping

If singularities are enabled, the engine creates a `SingularityGate`.

Important current caveat:

- singularity damping only runs when `metric_component` is explicitly passed into `compute_acceleration`.

So singularity protection exists in the engine, but it is not guaranteed to activate in every forward path automatically.

## What The Engine Does Not Do

The current engine does **not** directly apply:

- timestep integration,
- topology wrapping of the integration step,
- velocity saturation.

Those responsibilities belong elsewhere:

- integration and velocity saturation are handled by integrators,
- coordinate wrapping is typically handled by geometry projection or integrator topology resolution.

This is especially important because `engine.py` stores `velocity_saturation` on the module, but `compute_acceleration()` does not use it directly.

## Helper Methods

The engine also exposes:

- `get_friction_coefficient(...)`
- `get_ghost_force(...)`
- `apply_singularity_damping(...)`
- `apply_boundary(...)`
- `reset_hysteresis()`

Of these, the most important runtime helper is `get_friction_coefficient(...)`, because integrators can call it directly when they need explicit friction handling during split updates.

## Practical Interpretation

The current physics engine is best understood as:

- an acceleration orchestrator,
- centered on geometry plus centralized friction,
- with optional modular add-ons for memory, noise, exploration, and singularity damping.

It is not the whole solver. It produces acceleration-like quantities that the integrator then uses.

## Parameter Notes

The most relevant engine-side controls are:

| Parameter | Current Role |
|-----------|--------------|
| `stability.friction` | fallback base damping |
| `stability.velocity_friction_scale` | multiplies damping as velocity grows |
| `hysteresis.*` | enables ghost-force memory |
| `active_inference.stochasticity.*` | enables Brownian or OU noise |
| `active_inference.curiosity.*` | enables curiosity force |
| `singularities.*` | enables singularity gate |

Important caveat:

- `velocity_saturation` is **not** an engine-side clamp in the current runtime, even though it appears in nearby physics config.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/engine.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `gfn/realizations/gssm/physics/components/friction.py`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

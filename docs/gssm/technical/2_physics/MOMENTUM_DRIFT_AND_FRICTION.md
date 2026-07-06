# Momentum Drift And Friction

This note explains a real behavioral tradeoff in GSSM, but it is now phrased in terms that match the current runtime.

It should be read together with:

- `technical/runtime/01-hyperparameters.md`
- `technical/2_physics/engine.md`
- `technical/0_architecture/math/integrators/`

## The Core Issue

GSSM keeps a velocity state `v`. That is one of the reasons it can preserve motion and trajectory information, but it also means the model can continue moving after the external force becomes small or zero.

This is the practical meaning of momentum drift:

- the input force changes
- but the latent trajectory keeps moving because `v` still carries energy

For tasks that want smooth physical trajectories, that can be desirable.
For tasks that want near-discrete state flips, it can be harmful.

## Where Friction Enters

In the current runtime, friction is combined inside `ManifoldPhysicsEngine`.

At a high level:

```text
mu_total = config.stability.friction + mu_geo
```

and then optional velocity scaling is applied when `velocity_friction_scale > 0`.

So the main knobs that affect drift are:

- `stability.friction`
- `stability.velocity_friction_scale`
- `stability.base_dt`
- `stability.integrator_type`
- the geometry itself, if it returns `mu`

## Why Old Rules Were Too Strong

Older notes often described one fixed "overdamped recipe" such as:

- very high friction
- large velocity saturation
- one preferred integrator

That is too rigid for the current codebase.

The actual effect of friction depends on:

- timestep size
- solver choice
- topology
- force scale from the embedding
- whether velocity saturation is enabled
- whether the geometry returns extra friction

There is no universal rule like "friction must always be between `2.0` and `5.0` for discrete tasks."

## Practical Heuristic

If a task behaves too much like a coasting dynamical system when you really want abrupt state changes, test changes in this order:

1. increase `friction` moderately
2. reduce `base_dt`
3. keep `integrator_type="leapfrog"` until the task is stable
4. only then consider velocity-dependent friction or saturation

This is safer than jumping immediately to an extreme overdamped regime.

## About Velocity Saturation

Current runtime caveat:

- `velocity_saturation` exists in the schema
- its default is `0.0`, which means disabled
- saturation is handled in integrator helpers, not by the physics engine directly

So if you rely on saturation to suppress drift, that is an explicit design choice, not part of the default path.

## About Discrete-Like Tasks

For parity-like, counting-like, or exact symbolic transitions, the main question is not only friction.

You also need to verify:

- whether the embedding produces the intended force pattern
- whether the readout matches the target space
- whether the loss is appropriate for the manifold and target representation
- whether the task actually benefits from persistent velocity

If the task fundamentally wants "token causes immediate state flip and then stop," a high-momentum setup can be the wrong inductive bias regardless of solver order.

## Recommended Interpretation

Use momentum drift as a diagnosis term, not as a fixed recipe:

- if the model keeps moving after the force should effectively stop, friction may be too low for the task
- if the model becomes inert and cannot explore, friction may be too high

The correct setting is the one that matches the task's desired persistence, not the one that follows an old benchmark snapshot.

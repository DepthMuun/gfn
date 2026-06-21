# Geodesic Flow In Practice

This guide explains what "geodesic flow" means in the current GSSM runtime.

For exact implementation details, use:

- `technical/2_physics/engine.md`
- `technical/0_architecture/math/02_integrators.md`
- `technical/0_architecture/math/training/losses.md`

## The Basic Idea

A geodesic is the natural path induced by the geometry of a space.

As intuition, if there were no external force and no damping, a state moving on the manifold would continue along the path preferred by the geometry itself.

In GSSM, this idea is useful, but the real runtime usually includes more than pure geodesic motion:

- external force from the embedding
- friction
- optional hysteresis
- optional stochasticity
- optional curiosity
- optional singularity damping

So the actual model should be understood as geometry-aware forced motion, not as a pure free-particle geodesic solver on every step.

## Runtime Update Picture

At a high level, each timestep follows:

```text
embedding -> external force
geometry -> curvature term
physics engine -> acceleration
integrator -> proposal
dynamics mode -> next state
```

The geodesic idea mainly lives in the geometry + acceleration part of that chain.

## What The Integrator Actually Does

The current integrator family includes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`

The effective default is `leapfrog`.

All integrators inherit shared runtime behaviors from `BaseIntegrator`, including:

- topology-aware position wrapping
- optional velocity saturation
- delegated acceleration computation through the physics engine

That means "geodesic flow" in GSSM is never just a standalone equation on paper. It is the combination of geometry, engine, integrator, and topology helpers.

## Friction Changes The Story

Pure geodesic motion assumes no damping.

The current runtime does include damping:

- base friction from `config.stability.friction`
- optional geometry-returned `mu`
- optional velocity-dependent scaling through `velocity_friction_scale`

Because of that, a more realistic mental model is:

- geometry bends the motion
- friction damps the motion
- external force drives the motion

## Topology Wrapping

When the topology is toroidal, position is wrapped with:

```python
torch.atan2(torch.sin(x), torch.cos(x))
```

This matters because the actual trajectory is defined on a periodic manifold, not on an unconstrained Euclidean line.

So even if the motion is geodesic-like locally, the global path must respect topology.

## About Geodesic Regularization

GSSM does contain physics-aware losses, including geodesic-style regularization paths, but they are not all automatically active in a plain training loop.

Important runtime caveat:

- `PhysicsLoss` can use a geodesic component
- that component depends on data such as `state_info["christoffels"]`
- the default `BaseModel.forward()` contract does not expose `christoffels` by default

So geodesic regularization exists in the codebase, but it should not be documented as a universal always-on loss term for every training script.

## What To Tune First

If the trajectory behavior looks wrong, the most useful knobs are usually:

- `integrator_type`
- `base_dt`
- `friction`
- `velocity_friction_scale`
- topology choice
- whether the task actually needs toroidal supervision or identity readout

These runtime choices affect trajectory behavior more directly than abstract geodesic language alone.

## Practical Interpretation

Use "geodesic flow" as the organizing intuition for why GSSM uses geometry-aware state evolution.

But when reading or debugging the actual model, think in runtime terms:

- forced motion, not free motion
- damped motion, not purely conservative motion
- wrapped topology, not unconstrained coordinates
- integrator behavior, not only continuous-time equations

# Architecture Components

This page summarizes the **current runtime components** of GSSM and how they connect.

For deeper details, use:

- [00_overview.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/00_overview.md)
- [math/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/README.md)
- [runtime/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/runtime/README.md)

## Current Component Hierarchy

The maintained runtime picture is closer to:

```text
gfn.create("gssm", ...)
  -> ModelFactory
     -> embedding
     -> layers: ManifoldLayer x depth
        -> integrator
           -> physics_engine
              -> geometry
              -> optional hysteresis / stochasticity / curiosity / singularity gate
        -> mixer
        -> dynamics_x / dynamics_v
        -> optional layer plugins
     -> hook-attached readout and optional model plugins
```

Important current correction:

- the readout path is not best described as a permanently hardwired `CategoricalReadout` field living directly on the model object,
- in the current runtime it is typically attached through hooks and plugins during model construction.

## Base Model Components

`BaseModel` is the main evolution engine and currently owns:

- `layers`
- `embedding`
- `x0`
- `v0`
- `hooks`
- configuration and sequence-storage behavior

Its job is to:

- resolve forces,
- initialize or reuse state,
- evolve the sequence through layers,
- collect hook-produced logits,
- assemble final state and `state_info`.

## Manifold Layer Components

Each `ManifoldLayer` currently owns:

- `integrator`
- `mixer`
- `norm_x`
- `norm_v`
- `dynamics_x`
- `dynamics_v`
- `plugins`

The actual per-layer flow is:

1. reshape state and force,
2. plugin `pre_integrate`,
3. `integrator.step(...)`,
4. plugin `post_integrate`,
5. mixer,
6. dynamics routing,
7. topology wrap,
8. plugin `finalize`,
9. restore original shape.

So the layer is not just "integrator plus mixer"; it is the main orchestration point for evolution at layer granularity.

## Physics Subsystem

The physics subsystem is currently centered on `ManifoldPhysicsEngine`, which combines:

- geometry curvature,
- centralized friction,
- external force,
- optional ghost force from hysteresis,
- optional stochastic force,
- optional curiosity force,
- optional singularity damping.

Important current detail:

- the engine is the single authority on final friction application,
- geometry may provide a friction term, but the engine decides how that term is combined with config friction.

## Geometry

The geometry layer supplies:

- metric information,
- curvature contribution,
- projection,
- distance,
- sometimes geometry-side friction information.

The most important current families are:

- analytical topologies such as `torus`,
- learned geometries such as `low_rank`.

## Integrators

The factory currently exposes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`
- `adaptive`

Important current default:

- `leapfrog`

## Mixer

The mixer is a required layer component, not an optional plugin.

The runtime currently supports:

- `FlowMixer`
- `GeodesicAttentionMixer`

with partition vs ensemble behavior depending on configuration and return shape.

## Dynamics

The current dynamics registry supports:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

Important current detail:

- `dynamics_x` uses the configured topology,
- `dynamics_v` is always routed as Euclidean tangent-space dynamics.

## Plugins And Hooks

There are two different extension layers in the current architecture.

### Layer plugins

Examples:

- `dynamic_time`
- `fractal`

These modify layer execution directly through:

- `pre_integrate`
- `post_integrate`
- `finalize`

### Model hooks and plugins

Examples:

- readout attachment
- checkpointing
- adjoint
- lensing

These interact with:

- `on_timestep_end`
- `wrap_evolution`
- other lifecycle hooks in `HookManager`

## Practical Reading Order

Read components in this order if you want the current runtime picture:

1. model and hook layer
2. manifold layer
3. physics engine
4. geometry
5. integrator
6. mixer and dynamics
7. plugins

## What This Document Should Not Claim

It would be inaccurate to claim that:

- the model always has a direct fixed readout module in the old form,
- the only dynamics modes are `direct`, `residual`, and `gated`,
- mixers are optional plugins,
- the component hierarchy can be summarized correctly without hooks.

Those claims do not match the current runtime.

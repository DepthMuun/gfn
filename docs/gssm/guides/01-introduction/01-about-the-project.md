# About GSSM

GSSM is the Geodesic State Space Model realization inside the `gfn` framework.

Its central idea is simple:

- represent sequence state as position `x` plus velocity `v`
- evolve that state on a configured manifold
- use geometry, physics, and numerical integration as the state-transition mechanism

This gives GSSM a different structure from standard sequence models that only apply feed-forward updates in a flat latent space.

## What The Runtime Actually Builds

A public creation call such as:

```python
import gfn

model = gfn.create("gssm", vocab_size=32000)
```

goes through the current GSSM factory path and assembles:

- an embedding module
- one or more manifold layers
- a geometry instance
- a physics engine
- an integrator
- a dynamics mode
- a hook-driven readout
- optional plugins such as checkpointing, adjoint, pooling, or lensing

The resulting forward contract is:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

## Why Geometry Matters

Geometry is not just branding in GSSM. It affects:

- how curvature-like terms are computed
- how friction may be produced by the geometry
- how positions are wrapped or constrained
- how readout features are interpreted, especially on torus

The current runtime distinguishes analytical topologies from learned geometries. A fresh model now defaults to analytical torus geometry rather than silently falling back to a learned `reactive` geometry.

## Why Integrators Matter

GSSM evolves state with explicit numerical integrators.

The currently exposed family includes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`

The effective default is `leapfrog`.

This matters because timestep, friction, topology, and loss design interact. GSSM behavior is not determined by architecture alone.

## Why Configuration Needs Care

The codebase contains a few places where a raw schema default is not the same as the effective runtime default after normalization and factory logic.

Examples:

- top-level `rank` is declared as `32`, but the effective built value is often `16` unless you override it explicitly
- `riemannian_type="reactive"` appears in the schema, but does not override `topology.type="torus"` unless explicitly requested
- `velocity_saturation` is disabled by default even though older docs and constants can suggest otherwise

That is why the technical runtime docs exist: they document what the current code path really builds.

## Current Default Shape

A fresh GSSM model currently resolves to something close to:

- topology: `torus`
- geometry: analytical torus
- integrator: `leapfrog`
- `base_dt = 0.1`
- `friction = 0.01`
- embedding mode: `linear`
- readout type: `standard`

These are good starting values, not universal recommendations for every task.

## How To Approach GSSM

If you are new to the project, the safest order is:

1. Start with the public API and effective defaults.
2. Make sure the task target matches the chosen readout.
3. Change geometry or integrator only after the baseline path is correct.
4. Add optional physics modules one at a time.

## Where To Read Next

- `guides/04-guides/01-quick-start-guide.md`
- `guides/02-concepts-core/00-foundations.md`
- `guides/03-reference/00-handbook.md`
- `technical/runtime/00-effective-defaults.md`

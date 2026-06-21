# GSSM Foundations

This guide explains the ideas behind GSSM in the same terms used by the current runtime.

It is intentionally practical:

- `guides/` explains how to think about the system and how to use it.
- `technical/runtime/` and `technical/0_architecture/` are the code-aligned source of truth for defaults, factories, hooks, and implementation details.

## The Core Picture

GSSM models a sequence as the evolution of a latent state with two parts:

- `x`: position on a manifold
- `v`: velocity or momentum in the tangent space

At each timestep, the model converts the input token into a force, applies manifold-aware dynamics, and updates `(x, v)` through one or more `ManifoldLayer`s.

In the current runtime, the public forward path returns:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

`state_info` includes the stored trajectories (`x_seq`, `v_seq`), resolved forces, final state, and the effective mask. That is the structure used by the current physics-aware losses.

## What "Geometry" Means Here

Geometry controls how the state curves as it moves.

Conceptually, GSSM uses Christoffel-like terms to bend trajectories. In code, the geometry module is allowed to return either:

- `gamma`
- `(gamma, mu)`

where `gamma` is the curvature contribution and `mu` is an optional friction term returned by the geometry itself.

That distinction matters because the physics engine is the place where friction is combined and applied. The engine does not assume that all geometries behave identically.

## What "Physics" Means Here

The physics engine computes the effective acceleration by combining:

- geometry-induced curvature
- friction
- external force from the embedding
- optional hysteresis ghost force
- optional stochasticity
- optional curiosity force
- optional singularity damping

At a high level, the runtime follows:

```text
acceleration = -gamma - friction * v + external_force + optional_modules
```

This is a useful mental model, but the exact details depend on the configured geometry, integrator, and optional plugins.

## Effective Defaults

When you create a fresh model with:

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

the current runtime resolves to these effective defaults:

- topology: `torus`
- geometry selection: analytical torus by default
- integrator: `leapfrog`
- `base_dt`: `0.1`
- `friction`: `0.01`
- `velocity_friction_scale`: `0.0`
- `velocity_saturation`: `0.0` (disabled)
- embedding type: `standard`
- embedding mode: `linear`
- readout type: `standard`
- `holographic`: `False`
- effective `rank`: typically `16` unless explicitly overridden

Those are effective defaults, not just schema literals. They come from the full resolution path: schema -> overrides -> config normalization -> factory selection.

## Embeddings And Readout

The input side and output side are independent choices.

### Embedding

The current embedding module supports these main modes:

- `lookup`: plain table lookup
- `linear`: bit expansion of token IDs followed by projection
- `binary`: bit expansion mapped to `[-1, 1]`
- SIREN-style implicit mode through sinusoidal layers
- `continuous`: direct projection of continuous vectors

The schema default is `embedding.type='standard'` with `embedding.mode='linear'`.

### Readout

The current readout builder supports:

- `standard`: categorical projection to vocabulary size
- `implicit`: MLP projection to a configurable output dimension
- `identity`: returns the latent state directly

`holographic=True` no longer changes `readout.type='standard'` into `identity` automatically. If you want latent-state supervision, request `readout.type='identity'` explicitly.

## Integrators And Stability

GSSM evolves state with explicit numerical integrators. The current integrator factory exposes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`

The effective default is `leapfrog`.

For most users, the important practical rule is:

- treat `base_dt`, friction, and topology together
- change one thing at a time
- prefer smaller, controlled overrides instead of copying old benchmark configs blindly

There is no runtime-backed rule such as "every `0.1` of friction always means X" across all solvers and geometries. The actual effect depends on the chosen integrator, timestep, velocity scale, geometry, and optional modules.

## Public Creation Pattern

Use the top-level public API:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=32000,
    dim=512,
    depth=4,
    physics={
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
        },
        "topology": {
            "type": "torus",
        },
    },
)
```

That path is preferred over constructing deep internal components by hand unless you are deliberately extending the runtime.

## How To Read The Rest Of The Docs

Use the other guides for intuition:

- `01-physical-model.md`: high-level physical picture
- `04-dynamic-systems.md`: integrators, friction, and numerical behavior

Use the technical docs when you need runtime truth:

- `technical/runtime/00-effective-defaults.md`
- `technical/runtime/01-hyperparameters.md`
- `technical/0_architecture/`

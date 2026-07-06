# GSSM Architecture Overview

This document describes the **current runtime architecture** of GSSM as implemented in the codebase.

It intentionally avoids frozen benchmark claims and stale "generated from diagnostics" summaries. For defaults and parameter behavior, prefer the runtime-derived notes in:

- `docs/gssm/technical/runtime/00-effective-defaults.md`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

## Public Entry Point

GSSM is created through the public factory:

```python
import gfn

model = gfn.create("gssm", vocab_size=1000)
```

The factory resolves configuration, builds the model stack, attaches plugins, and returns a `ManifoldModel`.

## Runtime Model Shape

At a high level, the model is:

```text
input ids or manual forces
    -> embedding / force resolution
    -> initial latent state (x0, v0)
    -> stack of manifold layers
    -> readout plugin
    -> logits
```

The latent state is represented as:

- `x`: latent position
- `v`: latent velocity

These states evolve through each token step and through each layer.

## Main Components

### `ManifoldModel`

The top-level model is responsible for:

- resolving input forces,
- initializing or reusing latent state,
- evolving the sequence step by step,
- collecting logits through the readout hook system,
- returning final state and state traces.

The current forward contract is:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

`state_info` contains sequence traces and auxiliary information useful for physics-aware losses and diagnostics.

### Embedding / Force Resolution

The first runtime step is not "embedding" in a transformer sense only. The model resolves a **force sequence**:

- from `input_ids` via `FunctionalEmbedding`, or
- from `force_manual` when an external force sequence is supplied directly.

The embedding component supports several modes, including:

- `linear`
- `lookup`
- `binary`
- `siren`
- `continuous`

The current default runtime path is `physics.embedding.mode = "linear"`.

### Initial State

If no external state is supplied, the model initializes from learned parameters:

- `x0`
- `v0`

Both are created by `ModelFactory` and scaled using `initial_spread`.

### `ManifoldLayer`

Each layer contains:

- one geometry instance,
- one physics engine,
- one integrator,
- one mixer,
- one dynamics routing stage,
- optional plugins such as dynamic time or fractal behavior.

The layer receives the current `(x, v)` and one token-step force, then:

1. applies any pre-integration plugins,
2. performs one integration step,
3. mixes head states,
4. routes the proposal through the configured dynamics mode,
5. reapplies topology wrapping when needed,
6. runs finalize hooks.

### Geometry

Geometry determines the Christoffel contribution and, in several cases, a geometry-dependent friction gate.

The current factory supports both:

- analytical topologies such as `torus`, `spherical`, `hyperbolic`,
- learned geometries such as `low_rank`, `reactive`, and `adaptive`.

Important runtime rule:

- analytical topologies now win by default when explicitly declared,
- `riemannian_type` only overrides them when it was explicitly requested by the user.

That avoids the old failure mode where a config could declare `torus` but effectively instantiate `reactive` because of inherited defaults.

### Physics Engine

`ManifoldPhysicsEngine` is the central authority for combining forces.

Its net acceleration is:

```text
net_accel = -christoffel - friction_term + external_force + optional_secondary_forces
```

Optional secondary forces can include:

- hysteresis ghost force,
- stochastic force,
- curiosity force,
- singularity damping.

The engine also centralizes total friction computation so geometry and engine do not both apply damping independently.

### Integrator

The integrator is selected by `physics.stability.integrator_type`.

Current built-in options include:

- `leapfrog`
- `yoshida`
- `verlet`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

Current effective default:

- `leapfrog`

This is resolved by the typed config plus `IntegratorFactory`, not by old guide text.

### Mixer

After the integration step, heads are mixed through the configured mixer.

The layer builder currently supports:

- `FlowMixer`
- `GeodesicAttentionMixer`

The resulting mixed proposal is then passed through the configured dynamics mode.

### Dynamics Routing

Dynamics routing controls how the mixed proposal updates the latent state.

Current modes include:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

This stage operates on flattened state representations and then reshapes back to the head layout.

### Readout

Readout is attached as a plugin that runs on `on_timestep_end`.

Current readout modes:

- `standard`
- `implicit`
- `identity`

Important runtime behavior:

- `holographic=True` no longer silently changes `readout.type='standard'` into `identity`,
- if you want latent-state supervision, request `readout.type='identity'` explicitly.

For toroidal topologies, `standard` and `implicit` readouts consume `[sin(x), cos(x)]` features rather than raw coordinates.

## Sequence Evolution

The internal loop is token-first, layer-second:

```text
for each token step:
    resolve step force
    for each layer:
        integrate
        mix
        route dynamics
    trigger readout hook
```

This means:

- the model maintains a persistent latent state across the sequence,
- readout happens per timestep through hooks,
- optional plugins can modify state evolution without changing the outer model API.

## State Layout

The model uses a head-structured latent state.

In the common `geometry_scope="local"` case:

```text
head_dim = dim / heads
state shape ~= [B, H, D_head]
```

In `geometry_scope="global"`:

```text
head_dim = dim
state shape ~= [B, H, dim]
```

This distinction matters for:

- geometry construction,
- force reshaping,
- mixer behavior,
- readout dimension.

## Configuration Resolution

The runtime configuration path is:

1. instantiate `ManifoldConfig`,
2. apply nested `physics={...}` overrides,
3. normalize flat kwargs,
4. synchronize top-level and nested fields,
5. preserve explicit keys for downstream factory decisions.

Because of that pipeline, some values that look like defaults in one file are not the final effective defaults at runtime.

Examples:

- top-level `rank` and nested `riemannian_rank` are synchronized,
- `topology.type='torus'` does not automatically lose to schema-default `riemannian_type='reactive'`,
- `holographic` is merged with `active_inference.holographic_geometry`.

## Practical Starting Point

For a conservative starting configuration aligned with the current runtime:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    heads=4,
    depth=4,
    initial_spread=0.1,
    physics={
        "topology": {
            "type": "torus",
        },
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
        },
        "readout": {
            "type": "standard",
        },
    },
)
```

This is not presented as a universal optimum. It is simply a clean baseline that matches current defaults and avoids known legacy ambiguities.

## Runtime Caveats

Two important caveats for readers of older documents:

### Continuous embedding

The embedding module supports `mode="continuous"`, but the main `BaseModel.forward()` path currently resolves forces through `self.embedding(input_ids)` unless `force_manual` is provided.

That means continuous-input workflows should be documented carefully and treated as a special path, not as a drop-in replacement for token-id usage in every existing script.

### Defaults vs effective behavior

Do not treat `config/defaults.py` as the sole truth of runtime behavior. Effective defaults come from:

- schema defaults,
- normalization,
- sync rules,
- factory resolution,
- explicit user intent tracking.

## Related Documents

- `docs/gssm/technical/runtime/00-effective-defaults.md`
- `docs/gssm/technical/runtime/01-hyperparameters.md`
- `docs/gssm/guides/03-reference/03-integrators.md`
- `docs/gssm/guides/04-guides/02-advanced-configuration.md`

# GSSM API And Classes Reference

This reference maps the current public creation API to the internal runtime classes that are actually involved in a GSSM model.

It is not a promise that every internal class is part of the long-term public API. For extension work, these names are useful. For normal model creation, prefer `gfn.create("gssm", ...)`.

## Public Creation Path

### Minimal Example

```python
import gfn

model = gfn.create("gssm", vocab_size=1000)
```

### Explicit Example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    heads=4,
    depth=4,
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

## Public Forward Contract

A built GSSM model currently returns:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

`state_info` currently contains:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

That structure is what nearby physics-aware losses and plugins rely on.

## Top-Level Config Fields

These are the main top-level fields on `ManifoldConfig` that matter most in user code:

| Field | Current schema default | Notes |
|---|---:|---|
| `vocab_size` | required | Public entry requirement |
| `dim` | `512` | Total model dimension |
| `depth` | `4` | Number of manifold layers |
| `heads` | `4` | Number of heads |
| `rank` | `32` in schema | Effective built value is usually `16` unless `rank` was explicitly set |
| `integrator` | `leapfrog` | Synchronized with nested stability config |
| `adjoint_enabled` | `False` | Optional wrapped evolution path |
| `holographic` | `False` | Does not implicitly change readout type |
| `initial_spread` | `0.1` | Used for initial state sampling |
| `store_full_sequence` | `True` | Keeps full trajectories in `state_info` |
| `continuous_input_dim` | `None` | Needed for continuous embedding mode when applicable |

## Main Nested Config Groups

### `physics.topology`

| Field | Current default | Notes |
|---|---:|---|
| `type` | `torus` | Effective default topology |
| `R` | `2.0` | Major torus radius |
| `r` | `1.0` | Minor torus radius |
| `riemannian_type` | `reactive` in schema | Does not override torus unless explicitly requested |
| `riemannian_rank` | `16` | Feeds the effective default rank path |
| `geometry_scope` | `local` | Per-head geometry by default |
| `learnable_R` | `True` | Runtime-wired |
| `learnable_r` | `True` | Runtime-wired |

### `physics.stability`

| Field | Current default | Notes |
|---|---:|---|
| `base_dt` | `0.1` | Base timestep |
| `adaptive` | `True` | Adaptive timestep wrapper enabled in config |
| `adaptive_alpha` | `0.1` | Adaptive sensitivity |
| `dt_min` | `0.001` | Minimum timestep |
| `dt_max` | `1.0` | Maximum timestep |
| `enable_trace_normalization` | `True` | Metric normalization support |
| `wrap_x` | `True` | Position wrapping where relevant |
| `friction` | `0.01` | Base friction |
| `velocity_friction_scale` | `0.0` | Disabled by default |
| `velocity_saturation` | `0.0` | Disabled by default |
| `curvature_clamp` | `5.0` | Current schema-backed clamp |
| `friction_mode` | `static` | Current schema value |
| `integrator_type` | `leapfrog` | Effective default solver |
| `base_solver` | `leapfrog` | Underlying solver for adaptive path |
| `toroidal_curvature_scale` | `0.01` | Runtime-wired for torus geometry |

### `physics.embedding`

| Field | Current default | Notes |
|---|---:|---|
| `type` | `standard` | Schema default |
| `mode` | `linear` | Effective default embedding mode |
| `coord_dim` | `16` | Token bit-expansion width |
| `impulse_scale` | `1.0` | Also synchronized with top-level `impulse_scale` |
| `omega_0` | `30.0` | Used by the SIREN-style path |

### `physics.readout`

| Field | Current default | Notes |
|---|---:|---|
| `type` | `standard` | `standard`, `implicit`, or `identity` |
| `out_dim` | `None` | Used by `implicit` readout |
| `hidden_dim` | `None` | Defaults internally to `128` for `implicit` |

### `physics.dynamics`

| Field | Current default | Notes |
|---|---:|---|
| `type` | `direct` | Registered modes: `direct`, `residual`, `mix`, `gated`, `stochastic` |

## Main Runtime Classes

These classes are the most important internal runtime components.

### `ModelFactory`

Responsibility:

- resolves config input
- tracks explicit keys
- applies physics overrides
- normalizes config
- builds the model, readout plugin, and optional plugins

Important runtime detail:

- explicit user intent matters for geometry selection and config synchronization

### `ManifoldModel`

Concrete model class built by the factory.

It inherits the sequence-evolution behavior from `BaseModel` and exposes the public `forward()` contract used by training scripts.

### `BaseModel`

Implements:

- force resolution from embeddings
- batch lifecycle hooks
- state initialization
- the timestep evolution loop
- final `state_info` assembly

This is the class that currently defines the real return structure of `model(...)`.

### `FunctionalEmbedding`

The current embedding implementation supports:

- `lookup`
- `linear`
- `binary`
- SIREN-style implicit path
- `continuous`

In continuous mode, the module expects `continuous_input`, not token IDs.

### `ManifoldPhysicsEngine`

Combines:

- geometry output
- friction
- external force
- hysteresis
- stochasticity
- curiosity
- singularity damping

This is the runtime authority on friction application.

### `ReadoutPlugin`

Hooks into `on_timestep_end` and converts latent state to task outputs.

Available readout modules behind the plugin:

- `CategoricalReadout`
- `ImplicitReadout`
- `IdentityReadout`

## Internal Class Relationships

The current build path is:

```text
gfn.create("gssm", ...)
  -> ModelFactory.create(...)
  -> ManifoldModel(...)
  -> BaseModel.forward(...)
  -> ManifoldLayer(...)
  -> ManifoldPhysicsEngine(...)
  -> integrator + dynamics + hooks
  -> ReadoutPlugin.on_timestep_end(...)
```

## Training Example

```python
import torch
import gfn

model = gfn.create("gssm", vocab_size=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    input_ids, targets = batch

    logits, (_, _), state_info = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Practical Notes

- Prefer `gfn.create("gssm", ...)` over manual assembly unless you are extending the runtime.
- Treat schema defaults and effective defaults as different concepts.
- Do not assume `holographic=True` implies latent-state readout.
- Do not assume `rank=32` is the built default just because it appears at the top level of the schema.

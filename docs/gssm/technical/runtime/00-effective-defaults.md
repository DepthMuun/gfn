# GSSM Effective Defaults

This document describes the **effective runtime defaults** of GSSM as they are resolved by the current code path, not just by static tables.

The important distinction is:

- `config/defaults.py` contains centralized default values and legacy helpers.
- `config/schema.py` defines the dataclass defaults actually used when `ManifoldConfig(...)` is instantiated.
- `models/factory.py`, `config/normalizer.py`, and `geometry/factory.py` further transform and synchronize those values before the model is built.

For that reason, the effective default seen by a freshly created model can differ from the value you would infer by reading only one file.

## Resolution Order

The current runtime resolves configuration in this order:

1. `ManifoldConfig(...)` creates a schema-backed config with dataclass defaults.
2. Optional `physics={...}` overrides are applied through `apply_physics_overrides()`.
3. Flat kwargs are normalized through `normalize_config()`.
4. Sync rules copy selected values between top-level `ManifoldConfig` fields and nested `PhysicsConfig` fields.
5. Geometry selection is resolved by `GeometryFactory`, which uses both topology and the set of explicitly requested keys.

That means "default" must be read as "the value used after this pipeline finishes."

## Source Of Truth

For the current runtime, the most relevant sources are:

- `gfn/realizations/gssm/config/schema.py`
- `gfn/realizations/gssm/config/normalizer.py`
- `gfn/realizations/gssm/models/factory.py`
- `gfn/realizations/gssm/geometry/factory.py`
- `gfn/realizations/gssm/physics/integrators/factory.py`

`config/defaults.py` is useful reference material, but it is **not sufficient by itself** to determine the behavior of a newly created model.

## Effective Defaults Table

The following values describe the effective behavior of `gfn.create("gssm", vocab_size=...)` when no extra overrides are provided.

| Setting | Effective default | Why |
|---|---|---|
| `dim` | `512` at schema level, but the public model factory often starts from `ManifoldConfig(vocab_size=...)` and then syncs nested values as needed | Top-level `ManifoldConfig` default |
| `depth` | `4` | Top-level `ManifoldConfig` default |
| `heads` | `4` | Top-level `ManifoldConfig` default |
| `rank` | `16` effective unless explicitly overridden | `ConfigNormalizer` syncs `physics.topology.riemannian_rank -> config.rank` when `rank` was not explicitly set |
| `topology.type` | `torus` | `TopologyConfig.type` default |
| `topology.riemannian_type` | `reactive` in schema, but it does **not** override torus by default | `GeometryFactory` only lets `riemannian_type` override analytical topologies when it was explicitly requested |
| Effective geometry | `torus` analytical geometry | `GeometryFactory` prefers non-Euclidean declared topologies unless `riemannian_type` was explicit |
| `stability.integrator_type` | `leapfrog` | `StabilityConfig.integrator_type` default, then used by `IntegratorFactory` |
| `integrator` | `leapfrog` | Synchronized from `physics.stability.integrator_type` when top-level `integrator` was not explicit |
| `stability.base_dt` | `0.1` | `DEFAULT_DT` from constants, used by `StabilityConfig` |
| `stability.friction` | `0.01` | `DEFAULT_FRICTION` from constants, used by `StabilityConfig` |
| `stability.velocity_friction_scale` | `0.0` | `StabilityConfig` default, even though `defaults.py` still contains a different reference value |
| `stability.velocity_saturation` | `0.0` | Disabled by default in `StabilityConfig` |
| `stability.enable_trace_normalization` | `True` | `StabilityConfig` default |
| `stability.adaptive` | `True` | `StabilityConfig` default |
| `stability.base_solver` | `leapfrog` | Default base solver for the adaptive integrator |
| `stability.toroidal_curvature_scale` | `0.01` at config level | `StabilityConfig` default, consumed by torus geometry |
| `topology.learnable_R` | `True` | `TopologyConfig` default |
| `topology.learnable_r` | `True` | `TopologyConfig` default |
| `embedding.type` | `standard` | `EmbeddingConfig` default |
| `embedding.mode` | `linear` | `EmbeddingConfig` default |
| `embedding.coord_dim` | `16` | `EmbeddingConfig` default |
| `embedding.omega_0` | `30.0` | `EmbeddingConfig` default, now wired into the builder |
| `impulse_scale` | `1.0` | Top-level default synchronized with `physics.embedding.impulse_scale` |
| `readout.type` | `standard` | `ReadoutConfig` default |
| `holographic` | `False` | Top-level default, merged with `active_inference.holographic_geometry` using logical OR |
| `initial_spread` | `0.1` | `ManifoldConfig` default used to initialize `x0` and `v0` |
| `geometry_scope` | `local` | `TopologyConfig` default |
| `trajectory_mode` | `partition` | `PhysicsConfig` and `ManifoldConfig` default |
| `continuous_input_dim` | `None` | Continuous embedding requires an explicit input dimension when needed |

## Important Default Mismatches

The current codebase contains a few places where a reader can infer different defaults depending on which file they inspect first.

### `rank`

- `ManifoldConfig.rank` is declared as `32`.
- `TopologyConfig.riemannian_rank` is declared as `16`.
- `ConfigNormalizer` synchronizes nested physics values back into the top-level config when the top-level field was not explicitly provided.

Effective result:

- if the user does **not** pass `rank`, the built model will typically end up using `16`.
- if the user passes `rank=...`, that explicit value wins and is pushed into `physics.topology.riemannian_rank`.

### `riemannian_type`

- `TopologyConfig.riemannian_type` defaults to `reactive`.
- `defaults.py` still lists `low_rank` as a reference value.
- `GeometryFactory` no longer lets a non-explicit `riemannian_type` silently override a declared analytical topology such as `torus`.

Effective result:

- untouched configs still instantiate analytical torus geometry when `topology.type='torus'`.
- learned geometries such as `low_rank` or `reactive` only take priority when that choice was requested explicitly in the config path tracked by `_explicit_keys`.

### `velocity_friction_scale`

- `defaults.py` contains a reference value for velocity friction scaling.
- `StabilityConfig` currently defaults it to `0.0`.

Effective result:

- a fresh schema-backed config behaves as if velocity-dependent friction is disabled unless the user enables it.

## What "Explicit" Means

Several factory decisions depend on whether a field was set intentionally by the user, not just on the final value.

The model factory records explicit keys from:

- direct kwargs
- nested config dictionaries
- nested `physics={...}` overrides

This is especially important for:

- geometry selection
- synchronization between top-level and nested config fields

In practice, `topology.type="torus"` plus an untouched schema default `riemannian_type="reactive"` does **not** mean "reactive geometry wins." The code now treats that `riemannian_type` as an inherited default unless the user explicitly requested it.

## Creation Examples

### Minimal creation

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

This path uses:

- analytical torus geometry,
- leapfrog integration,
- `base_dt=0.1`,
- `friction=0.01`,
- standard readout,
- linear functional embedding,
- effective `rank=16` unless overridden.

### Explicit override

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=256,
    rank=32,
    physics={
        "topology": {
            "type": "torus",
            "riemannian_type": "low_rank",
        },
        "stability": {
            "integrator_type": "yoshida",
            "base_dt": 0.05,
        },
    },
)
```

This path changes behavior because the relevant keys become explicit:

- `rank=32` wins over the nested default,
- `riemannian_type="low_rank"` can override the torus analytical default because it was explicitly requested,
- `yoshida` becomes the actual integrator,
- `base_dt=0.05` becomes the base step size seen by the integrator.

## Documentation Rule

When documenting GSSM defaults anywhere else in the repo:

- prefer "effective default" wording,
- cite the runtime resolution path,
- avoid copying raw values from only `defaults.py`,
- call out cases where explicit user intent changes factory selection.

# Factory and Builder Pattern

This document explains how the **current GSSM runtime** assembles a `ManifoldModel` from configuration.

The authoritative implementation lives in:

- `gfn/realizations/gssm/models/factory.py`
- `gfn/realizations/gssm/models/builders/layer_builder.py`
- `gfn/realizations/gssm/models/builders/embedding_builder.py`
- `gfn/realizations/gssm/models/builders/readout_builder.py`
- `gfn/realizations/gssm/models/builders/plugin_builders.py`

## What The Factory Does

`ModelFactory.create(...)` is the main construction path used by the public API:

```python
import gfn

model = gfn.create("gssm", vocab_size=1000)
```

Its job is to:

1. resolve and normalize configuration,
2. preserve explicit user intent for downstream heuristics,
3. build embedding, layers, readout, and optional plugins,
4. initialize learnable latent state parameters,
5. return a fully assembled `ManifoldModel`.

## Supported Creation Flows

The current runtime supports several entry paths.

### 1. Direct typed config

```python
from gfn.realizations.gssm.config.schema import ManifoldConfig
from gfn.realizations.gssm.models.factory import ModelFactory

config = ManifoldConfig(vocab_size=1000, dim=512)
model = ModelFactory.create(config=config)
```

### 2. Public factory with flat overrides

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    heads=4,
    depth=4,
)
```

### 3. Nested physics overrides

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    physics={
        "topology": {"type": "torus"},
        "stability": {"base_dt": 0.05},
    },
)
```

### 4. Dict config reconstruction

If `config` is a plain dictionary, the factory reconstructs `ManifoldConfig` through `from_dict(...)`, including legacy wrapper unwrapping for keys such as `config`, `architecture`, or `model`.

## Real Configuration Resolution

The current resolution path is:

1. interpret `config` if it was passed as string, dict, or typed config,
2. collect explicit keys from direct kwargs and nested mappings,
3. create a base `ManifoldConfig` if none was provided,
4. apply nested `physics={...}` overrides,
5. run `normalize_config(config, kwargs, explicit_keys)`,
6. store `_explicit_keys` on both `config` and `config.physics`,
7. build the model from the normalized config.

This is more precise than a simple precedence slogan because the runtime also uses explicit-key tracking to distinguish:

- inherited schema defaults,
- real user intent.

That distinction matters for:

- geometry selection,
- top-level <-> nested sync rules,
- readout and holographic behavior.

## Builders Used By The Factory

The factory delegates most construction work to specialized builders.

### `EmbeddingBuilder`

Creates a `FunctionalEmbedding` using:

- `config.vocab_size`
- `config.dim`
- `config.physics.embedding.coord_dim`
- `config.physics.embedding.mode`
- `config.impulse_scale`
- `config.physics.embedding.omega_0`
- optional `config.continuous_input_dim`

Important current behavior:

- `impulse_scale` is taken from the top-level config,
- `continuous_input_dim` is only passed if explicitly available.

### `LayerBuilder`

Builds a `ModuleList` of `ManifoldLayer` instances.

For each layer, it creates:

1. geometry via `GeometryFactory.create_with_dim(...)`,
2. `ManifoldPhysicsEngine`,
3. integrator via `IntegratorFactory.create(...)`,
4. mixer via `FlowMixer` or `GeodesicAttentionMixer`,
5. `ManifoldLayer`.

The builder computes dimensions from:

- `config.dim`
- `config.heads`
- `config.physics.topology.geometry_scope`

If `geometry_scope == "global"`:

```text
head_dim = dim
dim_total = heads * dim
```

Otherwise:

```text
head_dim = dim // heads
dim_total = dim
```

### `ReadoutBuilder`

Builds the readout **plugin**, not just a bare projection module.

Current mapping:

- `readout.type = "implicit"` -> `ImplicitReadout`, wrapped in `ReadoutPlugin`
- `readout.type = "identity"` -> `IdentityReadout`, wrapped in `ReadoutPlugin`
- any other value, including `"standard"` -> `CategoricalReadout`, wrapped in `ReadoutPlugin`

Important current behavior:

- `holographic=True` with `readout.type="standard"` only triggers a warning,
- it no longer silently converts the readout into identity mode.

### Optional Plugin Builders

After the base model is created, the factory optionally attaches:

- pooling plugin
- checkpointing plugin
- adjoint plugin
- lensing plugin

These are registered into the model hook system after model construction.

## Actual Assembly Sequence

The current assembly path in `ModelFactory.create(...)` is approximately:

```text
resolve config
collect explicit keys
apply physics overrides
normalize config
compute geometry dimensions
build embedding
build layers
initialize x0 and v0
construct ManifoldModel
build readout plugin and register hooks
build optional plugins and register hooks
return model
```

More concretely:

```python
embedding = EmbeddingBuilder(config).build()
layers = LayerBuilder(config).build()
x0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
v0 = nn.Parameter(torch.randn(1, config.heads, head_dim) * spread)
model = ManifoldModel(layers, embedding, x0, v0, config.holographic, config=config, ...)
readout_plugin = ReadoutBuilder(config, dim_total, topology).build()
readout_plugin.register_hooks(model.hooks)
```

The readout is therefore attached through hooks, not passed directly as a constructor argument to `ManifoldModel`.

## Dependency Chain

The runtime dependency chain is:

```text
config
  -> embedding
  -> layers
       -> geometry
       -> physics engine
       -> integrator
       -> mixer
       -> manifold layer
  -> initial state parameters
  -> model
  -> readout plugin
  -> optional plugins
```

Within each layer:

```text
geometry -> physics engine -> integrator -> mixer -> dynamics routing
```

## Initial State Construction

The factory initializes:

- `x0`
- `v0`

as learnable parameters with shape:

```text
[1, heads, head_dim]
```

Both are sampled from:

```text
Normal(0, initial_spread)
```

using:

- `spread = getattr(config, "initial_spread", 0.1)`

This matches the current runtime default better than older docs that assumed zero-spread initialization.

## What The Factory Does Not Guarantee

The factory improves consistency, but it does not mean every config combination is semantically good.

Examples:

- `continuous` embedding exists at the embedding component level, but the main forward path still needs a compatible call path,
- `holographic=True` does not automatically imply `identity` readout,
- analytical topology and `riemannian_type` may still interact through explicit-key heuristics.

For those cases, always cross-check:

- `technical/runtime/00-effective-defaults.md`
- `technical/runtime/01-hyperparameters.md`

## Why The Builder Pattern Still Matters

The builder pattern is still a good description of the current codebase because it keeps concerns separated:

- embedding construction,
- per-layer assembly,
- readout construction,
- optional plugin attachment.

That separation is why it was possible to change:

- geometry selection heuristics,
- readout behavior,
- continuous embedding parameters,
- plugin attachment,

without rewriting the entire public creation API.

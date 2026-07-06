# Advanced Configuration

This guide explains how to configure GSSM using the **current runtime structure** rather than older config conventions.

For low-level runtime behavior, see:

- `docs/gssm/technical/runtime/00-effective-defaults.md`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

## Configuration Model

GSSM configuration is split between:

- top-level `ManifoldConfig` fields such as `dim`, `depth`, `heads`, `rank`, `integrator`, `initial_spread`, `holographic`
- nested `physics` fields such as topology, stability, embedding, readout, dynamics, hysteresis, singularities, and active inference

In practice, model creation usually looks like:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    depth=4,
    heads=4,
    physics={
        "topology": {"type": "torus"},
        "stability": {"integrator_type": "leapfrog"},
    },
)
```

## Real Configuration Resolution

The current runtime resolves configuration in this order:

1. instantiate `ManifoldConfig`
2. apply nested `physics={...}` overrides
3. normalize flat kwargs into nested fields where possible
4. synchronize top-level and nested values
5. preserve explicit keys for downstream factory decisions

This matters because some fields are **bidirectionally synchronized**:

- `integrator` <-> `physics.stability.integrator_type`
- `impulse_scale` <-> `physics.embedding.impulse_scale`
- `rank` <-> `physics.topology.riemannian_rank`
- `dynamics_type` <-> `physics.dynamics.type`
- `trajectory_mode` <-> `physics.trajectory_mode`
- `coupler_mode` <-> `physics.mixture.coupler_mode`
- `holographic` <-> `physics.active_inference.holographic_geometry`

So the safest way to document or debug a config is to think in terms of **effective runtime configuration**, not only raw input dictionaries.

## Top-Level Fields

These are the most important top-level parameters:

```python
model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    depth=4,
    heads=4,
    rank=32,
    integrator="leapfrog",
    initial_spread=0.1,
    holographic=False,
)
```

### What they do

- `dim`: base latent dimension used to build the state layout
- `depth`: number of manifold layers
- `heads`: number of latent heads
- `rank`: low-rank geometry and mixer rank, synchronized with `physics.topology.riemannian_rank`
- `integrator`: synchronized with `physics.stability.integrator_type`
- `initial_spread`: scale used to initialize learned `x0` and `v0`
- `holographic`: top-level holographic flag, merged with nested active-inference holographic geometry

## Nested `physics` Structure

The most important nested sections are:

- `topology`
- `stability`
- `dynamics`
- `active_inference`
- `embedding`
- `readout`
- `mixture`
- `fractal`
- `hysteresis`
- `singularities`

### Example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    physics={
        "topology": {
            "type": "torus",
            "riemannian_type": "low_rank",
            "learnable_R": True,
            "learnable_r": True,
        },
        "stability": {
            "integrator_type": "yoshida",
            "base_dt": 0.05,
            "friction": 0.01,
            "velocity_friction_scale": 0.02,
            "velocity_saturation": 5.0,
        },
        "embedding": {
            "mode": "linear",
            "coord_dim": 16,
        },
        "readout": {
            "type": "standard",
        },
    },
)
```

## Topology And Geometry

### `physics.topology.type`

This declares the main topology:

- `torus`
- `euclidean`
- `spherical`
- `hyperbolic`
- other registered analytical topologies

### `physics.topology.riemannian_type`

This declares an optional learned geometry override such as:

- `low_rank`
- `reactive`
- `adaptive`

Important runtime behavior:

- analytical topologies such as `torus` now win by default,
- `riemannian_type` only overrides them when it was **explicitly requested**.

That means:

```python
physics={"topology": {"type": "torus"}}
```

does **not** silently become `reactive` just because the schema has `riemannian_type='reactive'`.

### Recommended patterns

Use:

```python
physics={"topology": {"type": "torus"}}
```

when you want analytical torus behavior.

Use:

```python
physics={"topology": {"type": "torus", "riemannian_type": "low_rank"}}
```

when you intentionally want learned geometry to override the analytical torus default.

## Stability And Integrators

The most important runtime stability fields are:

```python
physics={
    "stability": {
        "integrator_type": "leapfrog",
        "base_dt": 0.1,
        "friction": 0.01,
        "velocity_friction_scale": 0.0,
        "velocity_saturation": 0.0,
        "enable_trace_normalization": True,
    }
}
```

### Current baseline

A clean baseline is:

- `integrator_type = "leapfrog"`
- `base_dt = 0.1`
- `friction = 0.01`
- `velocity_friction_scale = 0.0`
- `enable_trace_normalization = True`

### Practical tuning

If training is unstable:

- reduce `base_dt`,
- keep `leapfrog`,
- increase `friction` carefully,
- consider enabling `velocity_saturation`,
- only then try more structural changes.

If dynamics feel too damped:

- reduce `friction`,
- reduce `velocity_friction_scale`,
- keep the topology fixed while tuning,
- compare with a lower-order but simpler solver such as `heun` only when necessary.

## Embedding Configuration

Embedding is configured through:

```python
physics={
    "embedding": {
        "type": "standard",
        "mode": "linear",
        "coord_dim": 16,
        "impulse_scale": 1.0,
        "omega_0": 30.0,
    }
}
```

Important modes:

- `linear`
- `lookup`
- `binary`
- `siren`
- `continuous`

### Important runtime caveat: `continuous`

The embedding component supports `mode="continuous"`, but the main model forward path currently resolves forces through `self.embedding(input_ids)` unless `force_manual` is supplied.

That means continuous-input workflows should be treated as a special path and tested carefully instead of being assumed to work exactly like token-id mode in every existing training script.

## Readout Configuration

Readout is configured through:

```python
physics={
    "readout": {
        "type": "standard",
    }
}
```

Available modes:

- `standard`
- `implicit`
- `identity`

### Important current behavior

`holographic=True` no longer converts `standard` into `identity` automatically.

If you want latent-state output, use:

```python
physics={"readout": {"type": "identity"}}
```

explicitly.

For toroidal topologies, `standard` and `implicit` readouts use `[sin(x), cos(x)]` features internally.

## Dynamics And Mixing

Two related high-level controls are:

- `dynamics_type` or `physics.dynamics.type`
- `coupler_mode` or `physics.mixture.coupler_mode`

Common dynamics modes:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

Use `direct` unless you are intentionally exploring alternative routing behavior.

## Holographic And Identity Paths

There are two distinct ideas here:

- `holographic`: a configuration flag merged between top-level and active-inference config
- `identity` readout: a concrete readout mode that returns latent state directly

Do not assume they are interchangeable.

Recommended rule:

- if you want latent supervision, set `readout.type="identity"`
- if you want standard token logits, keep `readout.type="standard"`

## Practical Configurations

### Conservative baseline

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    dim=512,
    depth=4,
    heads=4,
    initial_spread=0.1,
    physics={
        "topology": {
            "type": "torus",
        },
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
            "enable_trace_normalization": True,
        },
        "embedding": {
            "mode": "linear",
        },
        "readout": {
            "type": "standard",
        },
    },
)
```

### Learned-geometry experiment

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    rank=32,
    physics={
        "topology": {
            "type": "torus",
            "riemannian_type": "low_rank",
            "riemannian_rank": 32,
        },
        "stability": {
            "integrator_type": "yoshida",
            "base_dt": 0.05,
        },
    },
)
```

### Latent-state supervision

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    holographic=True,
    physics={
        "readout": {
            "type": "identity",
        }
    },
)
```

In this case, the downstream loss must be compatible with latent-state output rather than standard token NLL alone.

## Common Mistakes

### Assuming a raw table value is the runtime default

Do not copy values from `config/defaults.py` and assume the model uses them unchanged. The effective runtime configuration also depends on:

- schema defaults,
- normalization,
- sync rules,
- explicit-key tracking,
- geometry and integrator factories.

### Assuming `torus` always means torus analytical geometry

This is only guaranteed when `riemannian_type` was not explicitly used to override it.

### Assuming `holographic=True` implies identity readout

That silent conversion was removed. Request `identity` explicitly.

### Assuming `continuous` embedding is drop-in for every script

Continuous mode exists in the embedding component, but the main forward path still needs careful handling because force resolution is centered on `input_ids` unless `force_manual` is used.

## Documentation Rule

When you add advanced-configuration examples elsewhere:

- prefer public `gfn.create("gssm", ...)`,
- show both top-level and nested config paths,
- use runtime-effective defaults,
- avoid legacy names such as `friction_scale`, `default_friction`, `dt`, or `physics_config` when the current code uses different fields.

# GSSM Handbook

This handbook is the practical reference for working with GSSM from the public API.

It does not freeze one "best benchmark configuration." Instead, it summarizes what the current runtime actually does, which defaults are effective, and which decisions you usually need to make first.

## Public Entry Point

Use the top-level API:

```python
import gfn

model = gfn.create("gssm", vocab_size=32000)
```

That creation path goes through the current config-resolution and factory pipeline, which means the effective behavior comes from more than one file.

## Effective Default Runtime

A fresh `gfn.create("gssm", vocab_size=...)` currently resolves to:

- topology: `torus`
- effective geometry: analytical torus
- integrator: `leapfrog`
- `base_dt = 0.1`
- `friction = 0.01`
- `velocity_friction_scale = 0.0`
- `velocity_saturation = 0.0`
- embedding type: `standard`
- embedding mode: `linear`
- readout type: `standard`
- `holographic = False`
- effective `rank = 16` unless explicitly overridden

Those values are derived from the schema, normalizer, factory synchronization, and geometry selection logic together.

## First Decisions To Make

Most GSSM setups become easier if you decide these items in order.

### 1. Output Contract

Pick the readout according to the supervision target:

- `standard`: categorical logits over `vocab_size`
- `implicit`: learned MLP projection to a task-specific output dimension
- `identity`: raw latent state for latent-space or geometry-aware supervision

Important runtime caveat:

- `holographic=True` no longer auto-switches `standard` readout into `identity`
- if you need latent-state supervision, set `readout.type="identity"` explicitly

### 2. Input Contract

Pick the embedding according to the input modality:

- `lookup`: classic discrete embedding table
- `linear`: bit expansion of token IDs then projection
- `binary`: bit expansion mapped to `[-1, 1]`
- SIREN-style implicit path for sinusoidal coordinate encoding
- `continuous`: direct projection of continuous vectors

The current schema default remains `embedding.mode="linear"`.

### 3. Geometry Contract

Decide whether you want:

- an analytical topology such as `torus`, `euclidean`, `hyperbolic`, or `spherical`
- or a learned geometry such as `low_rank`, `reactive`, or `adaptive`

Important runtime caveat:

- `topology.type="torus"` now wins by default
- `riemannian_type` only overrides the analytical topology when it was explicitly requested

### 4. Numerical Contract

Start conservative:

- `integrator_type="leapfrog"`
- `base_dt=0.1`
- `friction=0.01`

Only widen the configuration once the rest of the training loop is behaving correctly.

## What The Forward Pass Returns

The current model forward contract is:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

`state_info` is the bridge between the model core and physics-aware losses. It includes:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

## Dynamics Overview

GSSM separates three ideas that older docs often mixed together:

- geometry: computes curvature-like effects and may also return friction
- integrator: numerically advances the state
- dynamics mode: merges the proposal back into the persistent state

The current registered dynamics modes are:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

Use `direct` as the simplest baseline unless you have a reason to bias the system toward residual or gated updates.

## Optional Physics Modules

These modules exist in the current runtime but are off by default:

- hysteresis
- singularities
- stochasticity
- curiosity
- dynamic-time plugin
- fractal plugin

They are real runtime features, but they should be introduced intentionally. Enabling several at once can make debugging much harder.

## Loss Selection

GSSM does not have a single universal "best" loss.

In the current runtime, the practical split is:

- categorical tasks: task loss on the readout output, usually with `standard` readout
- latent or manifold-aligned tasks: `identity` or `implicit` readout plus an aligned supervision target
- physics regularization: optional losses that consume `state_info`
- toroidal tasks: torus-aware losses when the target itself lives on periodic coordinates

Do not assume that every training script should use the same loss family. The correct loss depends on whether your target lives in vocabulary space, Euclidean output space, or manifold coordinates.

## Recommended Starting Templates

### Language-Like Discrete Setup

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=50000,
    dim=512,
    depth=4,
    heads=4,
    physics={
        "topology": {"type": "torus"},
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
        },
        "readout": {"type": "standard"},
    },
)
```

### Latent-Space Supervision Setup

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    physics={
        "readout": {"type": "identity"},
        "topology": {"type": "torus"},
        "stability": {"integrator_type": "leapfrog"},
    },
)
```

### Continuous-Input Setup

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1,
    continuous_input_dim=128,
    physics={
        "embedding": {
            "mode": "continuous",
        },
        "readout": {
            "type": "implicit",
            "out_dim": 64,
        },
    },
)
```

## Troubleshooting Order

When a GSSM training run behaves badly, check these in order:

1. Are the targets aligned with the chosen readout?
2. Are the inputs aligned with the chosen embedding mode?
3. Are you relying on an old config snapshot with stale defaults?
4. Are `base_dt`, friction, and topology still close to the effective defaults?
5. Did you enable optional modules before validating the base path?

## Where To Go Next

Use these docs together:

- `01-constants.md` for currently relevant constants and schema defaults
- `02-api-classes.md` for the public-facing class and return-contract map
- `03-integrators.md` for the solver family
- `04-geometries.md` for geometry choices
- `05-dynamics-modes.md` for proposal-merging behavior

For implementation-level details, prefer:

- `technical/runtime/00-effective-defaults.md`
- `technical/runtime/01-hyperparameters.md`
- `technical/0_architecture/`

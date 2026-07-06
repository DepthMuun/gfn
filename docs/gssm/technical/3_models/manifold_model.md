# GSSM Model Stack

This document summarizes the current model-side runtime path for GSSM.

For exact flow details, also see:

- `technical/0_architecture/math/system/factory.md`
- `technical/0_architecture/math/09_forward_pass.md`
- `technical/0_architecture/math/10_backward_pass.md`

## Top-Level Model Classes

The main user-visible object built by the factory is `ManifoldModel`, defined in `models/manifold.py`.

`ManifoldModel` is intentionally thin. It delegates almost all real behavior to `BaseModel`.

### `ManifoldModel`

Role:

- concrete registered GSSM model class
- forwards directly into `BaseModel.forward(...)`

### `BaseModel`

Role:

- resolves forces from embeddings
- initializes state
- triggers lifecycle hooks
- runs the timestep loop across layers
- collects logits and trajectory info
- returns the public forward contract

## Public Forward Contract

The current forward path returns:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

Important runtime fields in `state_info`:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

That return structure is the real contract used by nearby training and loss code.

## Force Resolution

At the start of `BaseModel.forward(...)`:

- `force_manual` wins if provided
- otherwise `input_ids` are passed to the embedding module
- if neither is available, the model raises an error

This is one of the reasons the documented input contract must stay aligned with the chosen embedding mode.

## State Initialization

If a caller provides `state`, it is reused directly.

Otherwise, the current runtime:

1. asks hooks whether a custom initial state is provided
2. falls back to trainable `x0` and `v0`
3. applies `initial_spread` noise to position initialization when enabled

The model also resets certain stateful physics submodules between unrelated batches to avoid unwanted bleed-through.

## Sequence Evolution

The internal `_evolve_sequence(...)` loop does the following at each timestep:

1. apply the timestep mask to the force
2. trigger `on_timestep_start`
3. run each layer in order
4. trigger `on_timestep_end`
5. collect logits from readout hooks
6. store trajectories when `store_full_sequence=True`

The evolution function itself can be wrapped by plugins through the `wrap_evolution` hook.

## Manifold Layers

`ManifoldLayer` is where most of the per-step model mechanics live.

Its runtime responsibilities include:

- mixing state across heads/features
- calling the integrator
- applying the configured dynamics mode
- coordinating optional plugins at the layer level

It should not be described as a Transformer block analogue in a simplistic sense. The update path is explicitly geometry- and integrator-aware.

## Mixers

The current model stack uses mixer modules from `models/components/mixer.py`.

The important split is:

- `FlowMixer`
- `GeodesicAttentionMixer`

These are not model plugins. They are model components used inside the layer construction path.

## Embedding And Readout

### Embedding

The current embedding stack is centered on `FunctionalEmbedding`, which supports:

- `lookup`
- `linear`
- `binary`
- SIREN-style implicit path
- `continuous`

### Readout

Readout is attached through a hook-driven plugin, not by calling a fixed `model.readout(...)` method in the core forward loop.

Current readout module family:

- `CategoricalReadout`
- `ImplicitReadout`
- `IdentityReadout`

`ReadoutPlugin` registers `on_timestep_end` and produces the timestep outputs from latent state.

## Hooks And Plugins

The model owns a `HookManager`.

Current runtime hook usage includes:

- `on_resolve_forces`
- `on_batch_start`
- `state_init`
- `on_timestep_start`
- `on_layer_start`
- `on_layer_end`
- `on_timestep_end`
- `wrap_evolution`
- `on_batch_end`

Older summaries that mention a generic `pre_forward` phase as part of the active runtime are misleading. The main forward path is defined by the hooks actually triggered in `BaseModel`.

## Practical Caveats

- Do not document GSSM as if the model core exposes a fixed standalone readout method.
- Do not describe state persistence as automatic "infinite context" without mentioning explicit state passing and batch resets.
- Do not separate model docs from hook/plugin docs; the runtime path depends on them directly.

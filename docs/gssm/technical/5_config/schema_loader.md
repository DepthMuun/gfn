# GSSM Schema, Loader, And Normalizer

This document describes the current configuration resolution pipeline for GSSM.

It should be read together with:

- `technical/runtime/00-effective-defaults.md`
- `technical/runtime/01-hyperparameters.md`
- `technical/0_architecture/math/system/factory.md`

## Why This Matters

In the current GSSM runtime, "the config" is not a single file.

A built model depends on:

- schema defaults
- physics override application
- flat-kwarg normalization
- top-level <-> nested synchronization
- explicit-key tracking
- factory-level geometry selection

That is why older docs that described the config system only as "schema + loader" were incomplete.

## Schema Layer

The schema lives in `config/schema.py` and defines the typed dataclass structure used by GSSM.

Important dataclasses include:

- `TopologyConfig`
- `StabilityConfig`
- `DynamicsConfig`
- `ActiveInferenceConfig`
- `EmbeddingConfig`
- `ReadoutConfig`
- `PhysicsConfig`
- `TrainerConfig`
- `ManifoldConfig`

The schema defines many defaults, but those defaults are not always the same as the effective built runtime behavior.

## Loader Layer

`config/loader.py` is responsible for converting dictionaries into `PhysicsConfig` and for applying nested overrides.

The two main entry points are:

- `dict_to_physics_config(d)`
- `apply_physics_overrides(cfg, overrides)`

### `dict_to_physics_config`

Creates a fresh `PhysicsConfig()` and fills it from a nested mapping.

Use this when you want schema defaults plus a supplied nested dict.

### `apply_physics_overrides`

Mutates an existing `PhysicsConfig` in place.

Use this when you want to preserve the current config and only patch selected subfields.

This is the path used by `ModelFactory` when the user passes `physics={...}` overrides.

## What The Loader Currently Maps

The loader handles the current nested sub-configs for:

- topology
- stability
- dynamics
- active inference
- hysteresis
- singularities
- embedding
- readout
- mixture
- fractal
- trajectory mode

It also supports some compatibility aliases such as:

- `topology_config`
- `stability_config`
- `dynamics_config`
- `embedding_config`
- `readout_config`
- `major_radius`
- `minor_radius`

## Normalizer Layer

`config/normalizer.py` is the next major step after loading.

Its responsibilities are:

1. map flat kwargs into nested config fields
2. synchronize related parameters between `ManifoldConfig` and `PhysicsConfig`
3. validate the normalized result

### What It Can Map

Current mapping strategies include:

- dotted paths such as `physics.topology.type`
- direct top-level `ManifoldConfig` fields
- direct fields on nested physics sub-configs
- prefix-based mappings such as `topology_type`
- the special `active_inference_*` prefix path

### Synchronization Rules

The normalizer currently synchronizes these paired concepts:

- `integrator` <-> `physics.stability.integrator_type`
- `impulse_scale` <-> `physics.embedding.impulse_scale`
- `rank` <-> `physics.topology.riemannian_rank`
- `dynamics_type` <-> `physics.dynamics.type`
- `trajectory_mode` <-> `physics.trajectory_mode`
- `coupler_mode` <-> `physics.mixture.coupler_mode`
- `holographic` <-> `physics.active_inference.holographic_geometry`

Important caveat:

- `holographic` is handled specially with logical OR semantics between top-level and nested config

## Explicit Keys

One of the most important current runtime features is explicit-key tracking.

`ModelFactory` collects explicit keys from:

- incoming kwargs
- nested config dicts
- nested `physics={...}` mappings

It stores them as `_explicit_keys` on the config objects so downstream logic can tell the difference between:

- a schema default that was never intentionally requested
- a value the user explicitly asked for

This is especially important for:

- geometry selection
- top-level versus nested synchronization

## Validation

The current normalizer performs lightweight validation, including:

- `dim` must be divisible by `heads`
- topology type must be one of the currently accepted labels

This validation is intentionally limited. Many higher-level semantic mismatches are still the responsibility of the caller or downstream runtime code.

## Practical Resolution Order

The effective order seen by `gfn.create("gssm", ...)` is:

1. build or deserialize `ManifoldConfig`
2. apply nested `physics={...}` overrides if provided
3. normalize flat kwargs
4. synchronize top-level and nested fields
5. record `_explicit_keys`
6. let factories choose geometry, integrator, builders, and plugins

That order explains why reading only `schema.py` is not enough to understand what a new model will actually instantiate.

## Practical Caveats

- Do not document raw schema defaults as if they were guaranteed runtime defaults.
- Do not ignore `_explicit_keys`; several current factory decisions depend on them.
- Do not assume aliases or prefix mappings are handled only in the loader; some of the real flattening happens in the normalizer.

# GSSM Troubleshooting Guide

This guide focuses on the problems that actually show up most often when working with the current GSSM runtime.

It deliberately avoids frozen benchmark advice and focuses on the real failure modes caused by config resolution, geometry selection, readout choice, and sequence-state handling.

## Start With The Public Contract

A normal GSSM model is created with:

```python
import gfn

model = gfn.create("gssm", vocab_size=1024)
```

The current forward contract is:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

If your script assumes a different return shape, the rest of the pipeline will usually break downstream.

## 1. Loss Looks Wrong Or Validation Is Misleading

### Typical Cause

The readout and the target representation do not match.

### Check

- `readout.type="standard"` expects vocabulary-space supervision
- `readout.type="implicit"` expects a projected target space you define
- `readout.type="identity"` returns latent state directly and should be paired with a compatible latent or geometry-aware loss

### Fix

Make the target live in the same space as the readout.

If you are supervising toroidal coordinates or latent state, do not keep a plain categorical loss just because it worked in another script.

## 2. Continuous Input Crashes Immediately

### Typical Cause

`embedding.mode="continuous"` was enabled, but the model is still called with token IDs instead of continuous input.

### Check

In the current embedding implementation:

- continuous mode expects `continuous_input`
- discrete modes expect `input_ids`

### Fix

Make the call path consistent with the embedding mode:

```python
# discrete
logits, state, info = model(input_ids)

# continuous mode requires the model path using continuous input upstream
```

Also set `continuous_input_dim` when the projection dimension cannot be inferred safely.

## 3. Geometry Is Not The One You Expected

### Typical Cause

You assumed `riemannian_type="reactive"` would override `topology.type="torus"` automatically.

### What The Runtime Does Now

- analytical topologies win by default
- learned geometries override only when `riemannian_type` was explicitly requested

### Fix

If you want low-rank or reactive geometry, request it explicitly:

```python
physics = {
    "topology": {
        "type": "torus",
        "riemannian_type": "low_rank",
    }
}
```

## 4. Training Becomes Unstable Or Produces NaNs

### First Things To Check

- `integrator_type`
- `base_dt`
- `friction`
- `velocity_friction_scale`
- whether optional modules were enabled all at once

### Safe Baseline

```python
physics = {
    "topology": {"type": "torus"},
    "stability": {
        "integrator_type": "leapfrog",
        "base_dt": 0.1,
        "friction": 0.01,
        "velocity_friction_scale": 0.0,
        "velocity_saturation": 0.0,
    },
    "dynamics": {"type": "direct"},
}
```

### Notes

- the current default is not `dt=0.4`
- `velocity_saturation` is disabled by default
- friction is centralized in the physics engine and can also receive a geometry-returned `mu`

## 5. The Model Trains, But Learns The Wrong Thing

### Typical Cause

The script is mechanically running, but the supervision contract is conceptually wrong.

Common examples:

- next-token cross-entropy used for a target that is not actually next-token prediction
- bounding-box or coordinate regression forced through a categorical readout
- latent-space training attempted through `standard` readout instead of `identity` or `implicit`

### Fix

Audit these items together:

1. target definition
2. readout type
3. loss family
4. geometry/topology choice
5. metric used at evaluation time

## 6. Dimension Mismatch Errors

### Typical Causes

- `dim` not divisible by `heads`
- checkpoint/config mismatch
- continuous input projection size mismatch
- task head expecting a different last dimension than the readout produces

### Checks

```python
assert model.config.dim % model.config.heads == 0
```

Also verify:

- `vocab_size` matches the dataset tokenizer or target space
- `readout.out_dim` matches the downstream target when using `implicit`
- checkpoint and runtime config describe the same embedding mode and dimensions

## 7. Memory Usage Is Too High

### Typical Causes

- batch size too large
- `store_full_sequence=True` while also keeping long sequences
- task code storing extra tensors on top of `state_info`

### Fixes

- reduce batch size
- shorten the sequence length during debugging
- disable unnecessary per-step logging or cached tensors
- consider whether you really need the full trajectory in `state_info`

If your loss only needs the final state, storing the whole sequence may be unnecessary.

## 8. State Handling Feels Wrong Between Batches

### What The Runtime Already Does

`BaseModel.forward()` resets stateful physical components such as hysteresis and curiosity between unrelated batches when those modules are present.

### What You Still Need To Check

- whether your script is reusing `state` intentionally
- whether you are mixing training-time and inference-time state persistence logic

If you do not want cross-call persistence, pass `state=None` and treat each call as a fresh sequence.

## 9. Outputs Exist But Make No Sense

### Typical Causes

- `holographic=True` was assumed to imply identity readout
- stale config copied from older docs
- target decoding logic does not match the current checkpoint or tokenizer

### Important Runtime Caveat

`holographic=True` no longer changes `readout.type="standard"` into `identity` automatically.

If you need raw latent coordinates, request:

```python
physics = {
    "readout": {"type": "identity"}
}
```

## 10. A Good Debugging Order

When something is off, debug in this order:

1. Verify the public model return contract.
2. Verify input modality matches the embedding mode.
3. Verify target space matches the readout type.
4. Revert to analytical torus + leapfrog + direct dynamics.
5. Re-enable optional modules one by one.

## Minimal Sanity Check

```python
import torch
import gfn

model = gfn.create("gssm", vocab_size=256)
input_ids = torch.randint(0, 256, (2, 8))

logits, (x_final, v_final), state_info = model(input_ids)

assert logits.shape[:2] == input_ids.shape
assert x_final.shape == v_final.shape
assert "x_seq" in state_info
assert "v_seq" in state_info
```

If this baseline passes, most remaining bugs are task-specific rather than framework-shape issues.

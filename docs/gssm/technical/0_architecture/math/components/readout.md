# Readout

This document describes the **current readout modules and readout attachment path** used by GSSM.

The relevant implementation lives in:

- `gfn/realizations/gssm/models/components/readout.py`
- `gfn/realizations/gssm/models/builders/readout_builder.py`
- `gfn/realizations/gssm/models/base.py`

## What The Readout Does

The readout maps the current latent manifold state into either:

- categorical logits,
- a learned implicit output space,
- or the latent state itself.

In the present runtime, the readout is not called directly from `BaseModel.forward()` as a hard-coded module attribute. It is usually attached as a `ReadoutPlugin` on:

- `on_timestep_end`

So the model obtains timestep outputs through the hook system.

## Input Shape

The readout modules accept either:

- `[B, D]`
- `[B, H, D_h]`

When the state is multi-head, they flatten it to `[B, H * D_h]` before applying the actual readout mapping.

## Current Readout Types

### `standard`

This is implemented by `CategoricalReadout`.

Behavior:

- flatten latent state if needed,
- if topology is toroidal, replace raw coordinates with `[sin(x), cos(x)]`,
- apply a linear projection to `vocab_size`.

For torus:

```text
x_feat = [sin(x), cos(x)]
logits = Linear(x_feat)
```

For non-torus:

```text
x_feat = x
logits = Linear(x_feat)
```

This is the natural choice for:

- token classification,
- language-style training,
- standard CE / NLL objectives.

### `identity`

This is implemented by `IdentityReadout`.

Behavior:

- flatten latent state if needed,
- return it directly.

This is appropriate only when the downstream loss is compatible with latent coordinates.

Important runtime caveat:

- `IdentityReadout` is not a categorical token readout,
- using it with plain CE/NLL only makes sense if output dimensionality and target interpretation are deliberately compatible.

### `implicit`

This is implemented by `ImplicitReadout`.

Behavior:

- flatten latent state if needed,
- apply torus-aware `[sin(x), cos(x)]` features when topology is toroidal,
- pass through:

```text
Linear -> GELU -> Linear
```

The output dimension is:

- `physics.readout.out_dim` if provided,
- otherwise `vocab_size`.

This is the best fit when:

- the output space should be learned nonlinearly,
- the task is not just plain categorical projection,
- you want a structured regression or latent-alignment head.

## Hook-Based Attachment

The current builder does:

1. build the readout module,
2. wrap it in `ReadoutPlugin`,
3. register that plugin on the model hook manager.

So the actual runtime path is:

```text
latent state -> on_timestep_end hook -> ReadoutPlugin -> output tensor
```

and `BaseModel._evolve_sequence()` collects those returned tensors as timestep outputs.

## `holographic` And Readout

Important current runtime behavior:

- `holographic=True` does **not** automatically convert `standard` into `identity`,
- that silent conversion was removed,
- if you want latent-state output, request `readout.type="identity"` explicitly.

The builder still warns when it sees:

- `holographic=True`
- `readout.type="standard"`

but it keeps the standard categorical readout.

## Why Toroidal Readout Uses `sin/cos`

On a torus, raw angular coordinates are periodic. Direct linear projection across the wrap boundary is discontinuous in coordinate space.

Using:

```text
[sin(x), cos(x)]
```

makes nearby toroidal states remain nearby in feature space even when the raw angle wraps through `-pi` / `pi`.

This is why both:

- `CategoricalReadout`
- `ImplicitReadout`

use torus-aware trigonometric encoding.

## Comparison

| Readout | Output | Current Best Fit |
|---------|--------|------------------|
| `standard` | logits over vocabulary | categorical token tasks |
| `identity` | flattened latent state | latent supervision or coordinate losses |
| `implicit` | learned nonlinear output | structured regression or nontrivial latent mapping |

## What This Document Should Not Claim

It would be inaccurate to claim that:

- readout is always a direct module call from the outer model loop,
- `holographic=True` implies identity behavior,
- identity readout is a safe default for categorical training.

Those statements do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/models/components/readout.py`
- `gfn/realizations/gssm/models/builders/readout_builder.py`
- `gfn/realizations/gssm/models/base.py`
- `docs/gssm/technical/0_architecture/math/system/hooks.md`

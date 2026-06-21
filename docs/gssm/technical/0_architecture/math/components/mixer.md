# Mixer

This document describes the **current mixer runtime** used by `ManifoldLayer`.

The authoritative files are:

- `gfn/realizations/gssm/models/components/mixer.py`
- `gfn/realizations/gssm/models/builders/layer_builder.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

## What The Mixer Does

The mixer runs **after integration** and **before dynamics routing** inside `ManifoldLayer.forward()`.

The current sequence is:

1. integrate each head,
2. mix head states,
3. feed the mixed proposal into the dynamics modules,
4. reshape back to per-head form if needed.

So the mixer is not the final output head and not a hook plugin. It is a required internal component of each manifold layer.

## Two Concrete Mixer Classes

The current code exposes:

- `FlowMixer`
- `GeodesicAttentionMixer`

The builder chooses between them like this:

- if `mixer_type == "attention"`, build `GeodesicAttentionMixer`
- otherwise build `FlowMixer`

This is important because older docs tended to merge all attention behavior into `FlowMixer`, which is no longer the cleanest description of the current builder path.

## `FlowMixer`

`FlowMixer` supports modes:

- `low_rank`
- `default`
- `geodesic`
- `attention`
- `ensemble`

Important current caveat:

- inside `FlowMixer` itself, the modes `low_rank`, `default`, `geodesic`, and `attention` all route to the same partition-style build path,
- the truly separate attention implementation in the current builder is `GeodesicAttentionMixer`.

So the most faithful runtime description is:

- `FlowMixer` provides partition mixing plus ensemble mixing,
- attention as a distinct class lives in `GeodesicAttentionMixer`.

## Partition Path In `FlowMixer`

For non-ensemble modes, `FlowMixer` collapses:

- `[B, H, D_h] -> [B, D]`

for both position and velocity.

### Euclidean partition path

For position:

```text
x_flat = reshape(x)
x_agg = out_proj_x(x_flat)
```

For velocity:

```text
v_flat = reshape(v)
v_agg = out_proj_v(v_flat)
```

### Toroidal partition path

For torus, position mixing uses:

```text
[sin(x), cos(x), tanh(v / 10)]
```

as the feature vector before projection, then wraps back with:

```text
atan2(sin(x_agg), cos(x_agg))
```

Important current detail:

- toroidal position mixing explicitly includes a velocity-derived feature through `tanh(v / 10)`,
- so it is not only a position-only circular average.

### Velocity normalization in partition mode

Current runtime detail:

- partition velocity mixing uses `Identity()` for `mixed_norm_v`,
- not `RMSNorm`.

The reason is explicit in code comments: preserve momentum magnitude information instead of sphericalizing the mixed velocity.

## Ensemble Path In `FlowMixer`

In `ensemble` mode, `FlowMixer` preserves head structure:

- input `[B, H, D_h]`
- output `[B, H, D_h]`

It computes:

- softmax head weights from `ensemble_attn`,
- a consensus center,
- a small coupling update with coefficient `0.1`.

For torus:

- center uses circular averaging through `atan2(sum(w sin), sum(w cos))`,
- head-to-center deltas are wrapped angular differences.

For velocity:

- it computes a weighted center in ordinary linear space.

This mode is the right description when the model wants to keep separate head trajectories instead of collapsing them immediately.

## `GeodesicAttentionMixer`

`GeodesicAttentionMixer` is a separate class used by the builder when `mixer_type == "attention"`.

Its path is:

1. project `q`, `k`, `v`,
2. compute pairwise head distances,
3. turn negative distances into attention weights,
4. mix across heads,
5. flatten to `[B, D]`,
6. apply final output projection.

For torus:

- pairwise distance uses wrapped angular differences,
- mixed position uses `atan2(sum(w sin), sum(w cos))`.

For Euclidean:

- pairwise distance is ordinary squared-distance style attention.

So this is the actual current geodesic-attention implementation, not just a label on `FlowMixer`.

## Relation To `ManifoldLayer`

After mixing:

- if the mixer returns `[B, D]`, the layer runs the partition dynamics path and redistributes back into heads,
- if the mixer returns `[B, H, D_h]`, the layer runs the ensemble-preserving path.

This is why the output shape of the mixer matters so much to downstream behavior.

## Practical Guidance

Use partition-style mixing when:

- you want a single aggregated latent proposal per layer,
- the model behaves more like standard head aggregation.

Use ensemble mode when:

- you want to preserve headwise trajectories longer,
- you care about soft consensus instead of immediate collapse.

Use `mixer_type="attention"` when:

- you explicitly want the dedicated `GeodesicAttentionMixer` path.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- all attention behavior is implemented inside `FlowMixer`,
- velocity mixing always uses RMS normalization,
- the mixer is an optional plugin rather than a required layer component.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/models/components/mixer.py`
- `gfn/realizations/gssm/models/builders/layer_builder.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

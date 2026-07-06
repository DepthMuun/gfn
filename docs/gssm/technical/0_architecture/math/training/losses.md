# Losses

This document describes the **current loss classes actually registered** in GSSM and the caveats that matter in the present runtime.

The relevant implementation lives in:

- `gfn/realizations/gssm/losses/factory.py`
- `gfn/realizations/gssm/losses/generative.py`
- `gfn/realizations/gssm/losses/physics.py`
- `gfn/realizations/gssm/losses/toroidal.py`

## What Exists In The Current Runtime

The loss registry currently imports and registers families including:

- generative losses
- physics-informed losses
- toroidal losses
- detection losses
- regularization losses

The main documented classes in the current base path are:

- `ManifoldGenerativeLoss` registered as `generative`
- `PhysicsLoss` registered as `physics`
- `PhysicsInformedLoss` registered as `generative_physics`
- `ToroidalLoss` registered as `toroidal` and `toroidal_distance`
- `ToroidalCategoricalLoss`
- `ToroidalVelocityLoss`

## Loss Factory

`LossFactory.create(config)` expects either:

- a string loss type, or
- a config dictionary containing at least `type`

Example:

```python
loss = LossFactory.create({"type": "generative", "mode": "nll"})
```

If a key is unknown, the factory falls back to `generative`.

## `ManifoldGenerativeLoss`

This is the default generative loss family and supports multiple modes:

- `nll`
- `mse`
- `cosine`
- `toroidal`
- `hybrid`

### `nll`

This is the standard categorical path:

```python
F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
```

Optional extras:

- `label_smoothing`
- `entropy_coef`

This is the most natural choice when:

- the model uses categorical readout,
- outputs are token logits,
- targets are token ids.

### `mse`

This is a plain MSE path over outputs.

Important current caveat:

- if predictions are `[B, S, V]` logits and targets are `[B, S]`, the implementation only unsqueezes the target,
- so this mode should be treated as a specialized continuous-output path, not as a standard token-training default.

### `cosine`

This computes cosine distance on continuous outputs.

If the target is integer token ids, the implementation falls back to `nll`.

### `toroidal`

This computes wrapped angular error:

```text
diff_wrapped = atan2(sin(diff), cos(diff))
loss = mean(diff_wrapped^2)
```

Important current caveat:

- if the prediction is logits, the implementation converts logits to angles through a softmax-weighted average over evenly spaced angles,
- this is a runtime heuristic, not a mathematically exact inverse of categorical readout.

### `hybrid`

This combines:

- categorical `nll`
- a toroidal regularization term derived from `state_info["x_seq"]` when available

Important current caveat:

- the toroidal part only activates when `state_info` contains `x_seq`,
- otherwise the hybrid path degenerates mostly to the `nll` side.

## `PhysicsLoss`

`PhysicsLoss` is a weighted sum of up to three primitives:

- geodesic regularization
- Hamiltonian conservation
- kinetic regularization

### Geodesic term

The implementation uses:

```text
L_geo = mean(christoffels^2)
```

Important current caveat:

- the code requires `state_info["christoffels"]`,
- the default forward path does **not** populate `christoffels` automatically.

So geodesic regularization exists in code, but it is **not automatically active** in the standard forward path unless some custom wrapper or plugin adds that tensor to `state_info`.

### Hamiltonian term

The implementation uses only kinetic energy:

```text
H = 0.5 * ||v||^2
L_ham = mean(var_t(H))
```

Important current caveat:

- this is not a full potential-plus-kinetic Hamiltonian,
- it is a kinetic-energy stability surrogate.

### Kinetic term

The implementation penalizes only energy above a threshold:

```text
KE = 0.5 * ||v||^2
L_kin = mean(relu(KE - max_kinetic))
```

This is the most runtime-reliable of the three physics terms because it only depends on `v_seq`.

## `PhysicsInformedLoss`

`PhysicsInformedLoss` registered as `generative_physics` computes:

```text
cross_entropy + lambda_physics * physics_loss
```

and optionally subtracts an entropy bonus.

This is the cleanest built-in path when you want:

- categorical token training,
- plus optional physics regularization.

Important caveat:

- the physics part is only as rich as the available `state_info` fields,
- so in the default forward path the Hamiltonian and kinetic parts are easier to activate than the geodesic one.

## `ToroidalLoss`

`ToroidalLoss` is the main geometry-aware angular loss family.

Supported modes currently include:

- `circular`
- `mse`
- `riemannian`
- `hybrid`
- `phase`

### `circular`

Wrapped angular distance:

```text
diff_wrapped = atan2(sin(diff), cos(diff))
dist = diff_wrapped^power
```

This is the most direct toroidal loss.

### `riemannian`

This mode assumes coordinates come in `(theta, phi)` pairs and uses a torus-inspired metric:

```text
ds^2 = r^2 dtheta^2 + (R + r cos(theta))^2 dphi^2
```

Important caveat:

- if the last dimension is odd, the implementation falls back to circular behavior.

### `hybrid`

This mixes:

- circular wrapped distance
- vector consistency via `sin/cos`

### `phase`

This uses:

```text
1 - cos(diff)
```

which is often a smooth periodic objective.

## `ToroidalCategoricalLoss`

This loss is for the case where:

- predictions are logits or angular coordinates,
- targets are token ids,
- the task is still interpreted through toroidal token-angle mapping.

Important current caveats:

- token ids are mapped to evenly spaced angles unless `learnable_tokens=True`,
- the learnable token-angle path is effectively one-dimensional,
- this is a specialized loss, not a universal replacement for standard token CE.

## `ToroidalVelocityLoss`

This regularizes `state_info["v_seq"]` by penalizing excessive velocity magnitude.

It is simple and reliable because it only requires velocity traces.

## Practical Guidance

Use `generative` with `mode="nll"` when:

- you have categorical readout,
- you are training a normal token-prediction model.

Use `generative_physics` when:

- you want standard CE plus optional physics regularization.

Use `toroidal` or `toroidal_distance` when:

- outputs and targets are already meaningfully interpretable as toroidal coordinates.

Use `toroidal_categorical` only when:

- you intentionally want a token-to-angle bridge,
- you accept the current heuristic mapping behavior.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- geodesic regularization is always active in standard training,
- the physics loss always sees full geometric state,
- toroidal loss is the right default for every categorical pipeline,
- the generative toroidal mode reconstructs categorical tokens exactly.

Those are stronger claims than the current runtime supports.

## Runtime Cross-References

- `gfn/realizations/gssm/losses/factory.py`
- `gfn/realizations/gssm/losses/generative.py`
- `gfn/realizations/gssm/losses/physics.py`
- `gfn/realizations/gssm/losses/toroidal.py`
- `docs/gssm/technical/0_architecture/math/10_backward_pass.md`

# Optimizers

This document explains the **current optimizer utilities actually implemented** in GSSM.

The relevant code lives in:

- `gfn/realizations/gssm/training/optimizer.py`
- `gfn/realizations/gssm/training/__init__.py`

## What Exists In The Current Runtime

The training package currently exposes:

- `RiemannianAdam`
- `RiemannianSGD`
- `create_optimizer(...)`
- `make_gfn_optimizer(...)`
- `all_parameters(...)`

These utilities are public through:

```python
from gfn.realizations.gssm.training import (
    RiemannianAdam,
    RiemannianSGD,
    create_optimizer,
    make_gfn_optimizer,
)
```

## What `RiemannianAdam` Really Does

`RiemannianAdam` is currently a **thin extension of `torch.optim.Adam`**.

Its special behavior is not a full general-purpose Riemannian optimizer. Instead, after the base optimizer step, it optionally wraps selected position parameters back onto a torus:

```python
p.data = torch.atan2(torch.sin(p.data), torch.cos(p.data))
```

This wrap happens only when both conditions hold:

1. `geometry_type == "torus"`
2. the parameter group has `is_position=True`

So the current semantics are:

- Euclidean behavior -> standard Adam update
- Toroidal position parameters -> Adam update plus angular wrapping

## What `RiemannianSGD` Really Does

`RiemannianSGD` follows the same idea but extends `torch.optim.SGD` instead of `AdamW`.

Again, the current special behavior is:

- run the normal SGD step,
- then wrap toroidal position groups if `is_position=True`.

## Important Current Limitation

Despite the names, these optimizers do **not** currently implement:

- parallel transport,
- manifold-aware momentum transport,
- general retractions for arbitrary geometries,
- geometry-specific updates beyond torus angle wrapping.

So the most accurate mental model is:

- standard Euclidean optimizer core,
- plus optional torus projection for marked position parameters.

## Parameter Grouping In `create_optimizer(...)`

`create_optimizer(model, ...)` scans `model.named_parameters()` and builds two buckets:

### Position parameters

A parameter is treated as a toroidal position parameter if:

- its name contains `"x0"`, or
- its name contains `"position"`

These parameters are placed in a group with:

- `is_position=True`

### Other parameters

Everything else is placed in a normal group with:

- `is_position=False`

If the selected optimizer class is one of the Riemannian variants, this flag controls whether post-step torus wrapping is applied.

## What `make_gfn_optimizer(...)` Does

`make_gfn_optimizer(...)` is the higher-level helper for **dual-group optimization**.

It splits parameters into:

### Physics-sensitive parameters

The function places the following into the physics group:

- parameters named `x0`
- parameters named `v0`
- parameters named `impulse_scale`
- parameters whose names contain `"gate"`

This group gets:

- `lr = lr * physics_lr_scale`
- `weight_decay = 0.0`

### Base network parameters

Everything else goes into the base group with:

- `lr = lr`
- `weight_decay = weight_decay`

Important current detail:

- parameters from `extra_modules` are appended to the global named-parameter pool,
- but the physics group is collected only from `manifold.named_parameters()`,
- so `extra_modules` contribute to the base network group unless they are handled separately outside this helper.

## Default Optimizer Choice In `make_gfn_optimizer(...)`

The current helper defaults to:

- `optimizer_cls = torch.optim.AdamW`

That means the helper does **not** automatically choose `RiemannianAdam`.

If you want torus wrapping behavior through the helper, you must pass a compatible optimizer class explicitly.

## Typical Usage Patterns

### Simple torus-aware optimizer

```python
from gfn.realizations.gssm.training import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-3,
    geometry_type="torus",
)
```

### Name-based grouped optimizer

```python
from gfn.realizations.gssm.training import create_optimizer, RiemannianAdam

optimizer = create_optimizer(
    model,
    {
        "type": "riemannian_adam",
        "lr": 1e-3,
        "geometry": "torus",
        "weight_decay": 0.0,
    },
)
```

### Dual-group optimizer

```python
from gfn.realizations.gssm.training import make_gfn_optimizer

optimizer = make_gfn_optimizer(
    model,
    lr=1e-3,
    physics_lr_scale=10.0,
    weight_decay=1e-4,
)
```

## When The Riemannian Variants Matter

The Riemannian wrappers matter mainly when:

- the topology is toroidal,
- you want `x0` or other marked position parameters wrapped back to `[-pi, pi]`,
- you are using name-based or explicit parameter groups that preserve `is_position=True`.

They matter much less when:

- the topology is Euclidean,
- no parameter group is marked as positional,
- you use plain `AdamW` or `SGD`.

## Practical Interpretation

### Why wrap position parameters?

On a torus, positions are angular coordinates. A raw optimizer step can move them outside the canonical range.

The current optimizer handles this by mapping them back with:

$$\theta \mapsto \operatorname{atan2}(\sin\theta, \cos\theta)$$

which keeps angular coordinates in a wrapped representation compatible with toroidal dynamics.

### Why dual learning rates?

The helper assumes that some parameters have a more direct effect on the physical trajectory:

- initial state,
- velocity seed,
- impulse gain,
- gate parameters.

So it optionally trains them with:

- larger learning rate,
- no weight decay.

This is a training heuristic encoded in the helper, not a theorem about all GSSM workloads.

## What This Document Should Not Claim

It would be inaccurate to claim that the current runtime:

- always uses `RiemannianAdam`,
- always uses dual-group optimization,
- implements fully general manifold optimization.

Those are stronger statements than the current code supports.

## Runtime Cross-References

- `gfn/realizations/gssm/training/optimizer.py`
- `gfn/realizations/gssm/training/__init__.py`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

# Plugins - Dynamic Time

This document describes the **current `DynamicTimePlugin` implementation** used inside `ManifoldLayer`.

The authoritative code lives in:

- `gfn/realizations/gssm/models/plugins/dynamic_time.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

## What The Plugin Does

`DynamicTimePlugin` is a **layer plugin**, not a `HookManager` plugin.

Its only active integration point in the current runtime is:

- `pre_integrate(...)`

There it replaces the scalar layer timestep with a per-head effective timestep tensor.

## Current Runtime Path

Inside `ManifoldLayer.forward()` the plugin is called as:

```python
x_3d, v_3d, dt_eff = plugin.pre_integrate(x_3d, v_3d, dt_eff, f_3d)
```

So the dynamic-time path affects the integrator by changing:

- the effective `dt`

before the integrator step runs.

## How Parameters Are Built

During `setup()`, the plugin reads from:

- `layer.config.stability.base_dt`
- `layer.config.stability.dt_min`
- `layer.config.stability.dt_max`
- `layer.heads`
- `layer.head_dim`
- `layer.topology`

It then creates:

- one learnable scalar `dt_param` per head,
- one gating module per head.

## Base Timestep Parameterization

The learnable base timestep is:

```text
dt_base = softplus(dt_params)
```

and is clamped to:

```text
[dt_min, dt_max]
```

Current defaults read from stability config are typically:

- `base_dt = 0.1`
- `dt_min = 1e-4`
- `dt_max = 0.5`

### Initialization detail

The plugin initializes each head around:

```text
target_dt = base_dt / 0.9
```

and adds a small per-head offset of `0.05` in parameter space.

So the heads do not start identically.

## Gating Modes

The plugin supports two gating paths.

### Standard gating

Uses only position:

```python
gate_h = gating_h(x[:, i])
```

### Thermo gating

Uses both position and velocity:

```python
gate_h = gating_h(x[:, i], v[:, i])
```

This is controlled by:

- `dynamic_time_type == "thermo"`

Any other value falls back to the standard path.

## Effective Timestep

The current implementation computes:

```text
dt_eff = clamp(softplus(dt_params), dt_min, dt_max) * gates
```

with shapes:

- `dt_base`: `[1, H, 1]`
- `gates`: `[B_eff, H, 1]`
- `dt_eff`: `[B_eff, H, 1]`

So the timestep is:

- learnable per head,
- state-dependent per batch element,
- broadcast across the head feature dimension.

## What The Plugin Does Not Do

The current implementation does **not**:

- change the force directly,
- modify `x` or `v` in `pre_integrate`,
- implement a generic curvature-derived adaptive ODE solver,
- call `post_integrate`, `pre_mix`, `post_mix`, or `finalize` with dynamic-time-specific logic.

It is narrower than the old conceptual description: it is specifically a per-head learned timestep gate.

## Practical Interpretation

The most faithful interpretation of the current code is:

- every head gets a learnable baseline speed,
- the current state gates that speed,
- the resulting per-head timestep is passed into the integrator.

This is not the same as the separate `adaptive` integrator in `physics.stability.integrator_type="adaptive"`.

The two systems are different:

- `adaptive` integrator changes `dt` from acceleration magnitude at the solver level,
- `DynamicTimePlugin` changes `dt` through learned per-head gating before the solver step.

## When It Is Worth Using

Dynamic time is most plausible when:

- different heads truly specialize into different dynamical regimes,
- a single scalar timestep feels too restrictive,
- you want a learned per-head speed control path.

It is less compelling when:

- you need the simplest possible solver path,
- you want highly transparent timestep behavior,
- you already rely on the explicit `adaptive` integrator and do not want two timestep adaptation mechanisms interacting.

## Configuration

The plugin is enabled through the nested dynamic-time config path used by the layer plugin registry.

A representative configuration is:

```python
physics = {
    "active_inference": {
        "dynamic_time": {
            "enabled": True,
            "type": "thermo",
        }
    },
    "stability": {
        "base_dt": 0.1,
        "dt_min": 1e-4,
        "dt_max": 0.5,
    },
}
```

Important current caveat:

- the plugin code reads `dynamic_time_type` from the plugin config object,
- while user-facing configs often describe this simply as `type`,
- so this path should always be verified against the normalized config if behavior looks inconsistent.

## Runtime Cross-References

- `gfn/realizations/gssm/models/plugins/dynamic_time.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

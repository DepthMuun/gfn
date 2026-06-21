# Plugins - Fractal

This document describes the **current `FractalPlugin` implementation** used by `ManifoldLayer`.

The important source files are:

- `gfn/realizations/gssm/models/plugins/fractal.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

## What The Plugin Is Supposed To Do

Conceptually, the plugin is meant to provide a micro-manifold refinement path:

- detect high-curvature or high-velocity regions,
- run an auxiliary micro-manifold evolution,
- blend that refined state back into the main trajectory.

That conceptual idea is still visible in the code.

## What The Current Runtime Actually Does

The plugin is a **layer plugin** and only participates through:

- `finalize(...)`

inside `ManifoldLayer.forward()`.

Its actual logic is:

1. if the plugin is disabled, do nothing,
2. if `micro_manifold is None`, do nothing,
3. otherwise estimate curvature from velocity norm,
4. compute a sigmoid tunnel gate,
5. call `micro_manifold(x, v)`,
6. blend the micro state back into `(x, v)`.

## Critical Current Caveat

In the present implementation:

- `setup()` does **not** build a micro-manifold,
- `self.micro_manifold` starts as `None`,
- nothing in the default path shown here assigns it automatically.

That means:

- the plugin class exists,
- the finalize logic exists,
- but in the ordinary runtime path it is effectively a no-op unless some external code injects a real `micro_manifold`.

This is the most important thing to document accurately.

## Current Formula

When a micro-manifold is actually present, the plugin computes:

### Curvature estimate

```text
curvature_est = mean_h(||v_h||)
```

implemented as average velocity norm across heads.

### Tunnel gate

```text
tunnel_gate = sigmoid((curvature_est - threshold) * slope)
```

### Blend

```text
x_out = x + tunnel_gate * (x_f - x) * alpha
v_out = v + tunnel_gate * (v_f - v) * alpha
```

where `(x_f, v_f)` comes from:

```python
x_f, v_f = self.micro_manifold(x, v)
```

## Parameters Used Today

The plugin currently stores:

- `threshold`
- `alpha`
- `slope`

with current defaults:

- `threshold = 1.0`
- `alpha = 0.1`
- `slope = 1.0`

These affect behavior only if a micro-manifold is actually present.

## What The Plugin Does Not Currently Do

The present implementation does **not**:

- automatically build a sub-manifold in `setup()`,
- modify the timestep,
- observe force directly,
- attach any hook-based behavior,
- guarantee any extra compute in the default path.

So it should not be documented as if it were an always-on refinement engine.

## Best Interpretation

The most faithful interpretation is:

- this is a prepared extension point for fractal or multiscale refinement,
- its blending rule is implemented,
- but the runtime still needs an external micro-manifold assignment for it to become active.

## When It Would Matter

The plugin becomes meaningful only when a calling path supplies `micro_manifold`.

Then it can be useful for:

- localized refinement,
- multiscale experiments,
- high-velocity regime smoothing through blended micro-evolution.

Without that extra setup, enabling the config flag alone is not enough to get the intended effect.

## Configuration

A representative config is:

```python
physics = {
    "fractal": {
        "enabled": True,
        "threshold": 1.0,
        "alpha": 0.1,
        "slope": 1.0,
    }
}
```

Important current caveat:

- this config enables the plugin instance,
- but it does not by itself create the micro-manifold that the plugin needs to do any refinement.

## Runtime Cross-References

- `gfn/realizations/gssm/models/plugins/fractal.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

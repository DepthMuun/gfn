# Geometry Reference

This guide summarizes the geometry choices that the current GSSM runtime can build.

For exact implementation details, use:

- `technical/0_architecture/math/03_geometry.md`
- `technical/0_architecture/math/geometry/`
- `technical/runtime/00-effective-defaults.md`

## How Geometry Is Selected

The geometry factory uses two config keys:

- `physics.topology.type`
- `physics.topology.riemannian_type`

Current runtime rule:

- analytical topologies such as `torus`, `hyperbolic`, and `spherical` win by default
- learned geometries such as `low_rank`, `reactive`, and `adaptive` only override an analytical topology when `riemannian_type` was explicitly requested

This matters because the schema default still says `riemannian_type="reactive"`, but a fresh GSSM model with `topology.type="torus"` now instantiates torus geometry by default.

## Effective Default Geometry

For a fresh:

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

the effective geometry is:

- topology: `torus`
- geometry: analytical torus

not a learned reactive geometry.

## Geometry Contract

The geometry module may return either:

- `gamma`
- `(gamma, mu)`

where:

- `gamma` is the Christoffel-like curvature term
- `mu` is an optional friction contribution returned by the geometry

The physics engine is responsible for interpreting that contract and combining friction consistently.

## Main Geometry Families

### Torus

Use when periodic structure is a natural fit.

Typical uses:

- cyclic or angular targets
- modulo-like structure
- wrapped latent coordinates
- toroidal losses or latent coordinate supervision

Key runtime points:

- torus is the effective default geometry
- radii `R` and `r` are runtime-wired and can be learnable
- `toroidal_curvature_scale` is now an active runtime knob
- torus-aware readouts can use `sin/cos` feature expansion

Typical config:

```python
physics = {
    "topology": {
        "type": "torus",
        "R": 2.0,
        "r": 1.0,
        "learnable_R": True,
        "learnable_r": True,
    },
    "stability": {
        "toroidal_curvature_scale": 0.01,
    },
}
```

### Euclidean

Use when you want the simplest flat-space baseline.

Key runtime points:

- no toroidal wrapping
- no analytical torus curvature
- useful when you want to remove periodic structure from the equation

Typical config:

```python
physics = {
    "topology": {
        "type": "euclidean",
    }
}
```

### Hyperbolic

Use when the problem naturally benefits from negatively curved analytical geometry.

This is an analytical geometry choice, not just a conceptual recommendation. If you set `topology.type="hyperbolic"`, that analytical topology now wins by default unless you explicitly ask for a learned geometry override.

### Spherical

Use when closed positive-curvature structure is a better match than a torus or flat space.

As with hyperbolic geometry, this is selected through `topology.type`.

### Low-Rank

Use when you want a learned Riemannian geometry rather than a fixed analytical topology.

Key runtime points:

- rank comes from `physics.topology.riemannian_rank`
- low-rank geometry is learned
- the runtime includes Python and optional CUDA paths for nearby low-rank operations
- low-rank must be explicitly requested if you also declare an analytical topology

Typical config:

```python
physics = {
    "topology": {
        "type": "torus",
        "riemannian_type": "low_rank",
        "riemannian_rank": 32,
    }
}
```

### Reactive

Use when you want a learned geometry path designed around reactive curvature behavior.

Key runtime point:

- `reactive` is no longer silently chosen over `torus` just because it appears as the schema default for `riemannian_type`

### Adaptive

Use when you want a learned geometry that adapts more explicitly with configuration-driven plasticity.

This is an experimental learned-geometry choice compared with the more straightforward analytical topologies.

## Geometry Scope

`physics.topology.geometry_scope` controls whether geometry is applied per head or on the full dimension:

- `local`: default, per-head geometry
- `global`: geometry sees the full model dimension per head path

The current model factory uses this setting to decide the dimension passed into the geometry builder.

## Selection Guide

Use `torus` when:

- the task has periodic structure
- you want the current default, best-supported analytical path
- you plan to use toroidal supervision or wrapped coordinates

Use `euclidean` when:

- you want the simplest non-periodic baseline
- you need to isolate whether toroidal behavior is helping or hurting

Use a learned geometry (`low_rank`, `reactive`, `adaptive`) when:

- you explicitly want the geometry itself to be learned
- you are prepared to tune more than the default analytical path usually requires

## Minimal Examples

### Default analytical torus

```python
import gfn

model = gfn.create("gssm", vocab_size=1024)
```

### Explicit low-rank override

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    physics={
        "topology": {
            "type": "torus",
            "riemannian_type": "low_rank",
            "riemannian_rank": 32,
        }
    },
)
```

### Euclidean baseline

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    physics={
        "topology": {"type": "euclidean"},
    },
)
```

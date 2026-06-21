# GSSM Geometry Base

This document describes the current geometry contract used by the GSSM runtime.

For file-level details of specific implementations, also see:

- `technical/0_architecture/math/03_geometry.md`
- `technical/0_architecture/math/geometry/`
- `technical/runtime/00-effective-defaults.md`

## Core Role

Geometry is the layer that tells the physics engine how curvature and, optionally, geometry-driven friction should behave.

In the current runtime, geometry is not just a metric lookup. It participates directly in the acceleration path consumed by `ManifoldPhysicsEngine`.

## Base Class

All geometry implementations inherit from `gfn/realizations/gssm/geometry/base.py`:

```python
class BaseGeometry(nn.Module):
    ...
```

The base class stores:

- `config`
- `return_friction_separately`
- `topology_type`

and provides default implementations for:

- `metric_tensor(x)`
- `christoffel_symbols(x)`
- `compute_kinetic_energy(x, v)`
- `compute_potential_energy(x)`
- `forward(x, v, force=None)`
- `project(x)`
- `dist(x1, x2)`

## Runtime Contract

The geometry `forward()` path may return either:

- a tensor representing the geometry contribution
- a tuple `(gamma, mu)`

where:

- `gamma` is the Christoffel-like curvature term
- `mu` is an optional friction contribution

The physics engine is responsible for interpreting that result and combining `mu` with the fallback friction from config.

That tuple-return contract is important because several current docs from older versions incorrectly assumed a single universal geometry output shape.

## BaseGeometry Behavior

The base implementation is intentionally conservative:

- `metric_tensor(x)` returns ones
- `christoffel_symbols(x)` returns zeros
- `project(x)` is identity
- `dist(x1, x2)` is Euclidean norm

Its `forward()` implementation computes a simplified acceleration-like quantity and, when `return_friction_separately` is enabled, returns a tuple whose second component is zero friction.

Concrete geometries override the parts they actually need.

## Factory Selection

Geometry instances are created through `GeometryFactory`.

The selection keys are:

- `physics.topology.type`
- `physics.topology.riemannian_type`

Current runtime rule:

- analytical topologies such as `torus`, `hyperbolic`, and `spherical` win by default
- learned geometries such as `low_rank`, `reactive`, and `adaptive` only override when `riemannian_type` was explicitly requested

This behavior depends on `_explicit_keys` propagated from `ModelFactory`.

## Effective Default

For a fresh:

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

the effective geometry is analytical torus geometry, not a learned `reactive` geometry, even though `TopologyConfig.riemannian_type` defaults to `reactive` in the schema.

## Geometry Families In The Current Runtime

The factory imports and registers these geometry modules:

- `euclidean`
- `torus`
- `low_rank`
- `adaptive`
- `reactive`
- `hyperbolic`
- `holographic`
- `hierarchical`
- `spherical`

They do not all play the same role:

- some are analytical topologies
- some are learned geometries
- some are more experimental extension points

The important distinction for runtime behavior is whether a geometry is selected by declared topology or by explicit learned-geometry override.

## Dimension Handling

The main model factory usually constructs geometry with `create_with_dim(...)`, not the simpler `create(...)` helper.

That matters because geometry usually runs on per-head tensors:

- local scope: per-head dimension
- global scope: full model dimension

`geometry_scope` decides which dimension is passed into the geometry builder.

## Practical Caveats

- Do not infer geometry behavior from the schema alone.
- Do not assume `riemannian_type` always wins over `topology.type`.
- Do not assume every geometry returns only curvature without friction.
- Treat torus as the current best-supported analytical default path.

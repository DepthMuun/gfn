# Euclidean Geometry

This document describes the **current `EuclideanGeometry` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/geometry/euclidean.py`

## What It Does

`EuclideanGeometry` is the simplest registered geometry in the current runtime.

Its behavior is exactly what the code suggests:

- metric tensor is identity-like,
- Christoffel symbols are zero,
- projection is identity,
- distance is ordinary Euclidean norm.

## Metric

The implementation returns:

```python
torch.ones_like(x)
```

for `metric_tensor(x)`.

So in the current runtime the Euclidean metric is represented as a diagonal per-coordinate weight tensor of ones, not as an explicit dense identity matrix.

That distinction matters because other runtime utilities, such as metric-aware normalization, are written to accept diagonal forms too.

## Curvature

`christoffel_symbols(x)` returns:

```python
torch.zeros_like(x)
```

So the Euclidean geometry contributes no analytical curvature term of its own.

In practical engine terms, this means the acceleration will come from:

- friction,
- external force,
- optional hysteresis,
- optional stochasticity,
- optional curiosity,

but not from Euclidean geometric curvature.

## Projection

`project(x)` is identity.

So Euclidean geometry itself does not wrap or constrain coordinates.

Important practical caveat:

- if you use Euclidean geometry, boundedness must come from other mechanisms,
- not from geometry projection.

## Distance

`dist(x1, x2)` is:

```python
torch.norm(x1 - x2, dim=-1)
```

So the distance path is the standard Euclidean norm in the last dimension.

## Return Contract

Unlike torus or low-rank, `EuclideanGeometry` does not override `forward(...)`.

That means it inherits the fallback `BaseGeometry.forward(...)` behavior.

Important consequence:

- in practice, with Euclidean Christoffels equal to zero, the base path yields zero geometry acceleration and zero geometry-side friction output.

So the engine will usually rely on:

- fallback friction from config,
- and the non-geometry force terms.

## When It Is A Good Fit

Euclidean geometry makes sense when:

- you want the simplest flat baseline,
- periodic structure is not meaningful,
- you explicitly do not want analytical curvature.

It is less safe when:

- you want bounded latent coordinates,
- you rely on topology itself for stability,
- toroidal wrapping is part of the intended inductive bias.

## Normalization Interaction

With Euclidean topology, the registry currently chooses:

- identity for position,
- metric-aware velocity normalization if geometry is passed through,
- otherwise tangent velocity normalization.

So even though the geometry is flat, velocity control can still come from the normalization stack.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- Euclidean geometry itself provides any wrapping,
- Euclidean geometry provides curvature-based friction,
- Euclidean space is automatically safe just because the geometry is simple.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/geometry/euclidean.py`
- `gfn/realizations/gssm/geometry/base.py`
- `docs/gssm/technical/0_architecture/math/components/normalization.md`

# Geometry

This document describes the **current geometry runtime contract** used by GSSM.

The authoritative files are:

- `gfn/realizations/gssm/geometry/base.py`
- `gfn/realizations/gssm/geometry/factory.py`
- `gfn/realizations/gssm/geometry/torus.py`
- `gfn/realizations/gssm/geometry/low_rank.py`

## What Geometry Means In The Current Runtime

In GSSM, a geometry module provides the manifold-specific part of the dynamics:

- metric information,
- curvature / Christoffel-like contribution,
- projection back to the manifold,
- distance function,
- sometimes a geometry-side friction signal.

The geometry does not by itself perform the whole evolution step. It supplies structure that the physics engine and integrator consume.

## Base Contract

`BaseGeometry` currently defines:

- `metric_tensor(x)`
- `compute_kinetic_energy(x, v)`
- `compute_potential_energy(x)`
- `forward(x, v, force=None)`
- `project(x)`
- `dist(x1, x2)`

Important current runtime detail:

`forward(...)` may return either:

- a tensor,
- or a tuple `(gamma, mu)`

This tuple form is important because many modern geometries separate:

- curvature contribution,
- friction contribution.

## Base Implementation Caveat

The `BaseGeometry.forward(...)` implementation is only a fallback template. It uses a simplified pointwise `-gamma * v^2` pattern and optional `force / g`.

It should not be treated as the definitive mathematical behavior of the specialized geometries.

The real runtime behavior comes from the concrete subclasses.

## Factory Selection

The geometry factory now follows this logic:

- prefer the declared analytical or topological `topology.type`,
- allow learned `riemannian_type` to override only when it was explicitly requested,
- avoid silently replacing `torus` with a learned geometry just because of schema defaults.

This is one of the key current runtime fixes.

So the docs should no longer imply that `riemannian_type='reactive'` automatically wins in all cases.

## Geometry Families That Matter Most In The Current Runtime

### `torus`

`torus` maps to `ToroidalRiemannianGeometry`.

It provides:

- analytical torus-style metric,
- paired-coordinate curvature,
- optional learnable radii,
- toroidal projection,
- optional friction gate,
- optional CUDA fast path.

This is the most important analytical topology in the current runtime.

### `flat_torus`

Defined in the same torus file, but with:

- periodic wrapping,
- flat metric,
- zero analytical curvature,
- separate friction handling.

This is useful when you want periodic coordinates without full toroidal curvature.

### `low_rank`

`LowRankRiemannianGeometry` is the main learned geometry path documented in the current code.

Its behavior is:

- learn low-rank basis tensors `U` and `W`,
- produce curvature-like terms from bilinear contractions of velocity,
- optionally use trace normalization,
- return `(gamma, mu)` with separate friction gate.

Important current detail:

- this is an approximation-oriented learned geometry,
- not an exact closed-form analytical manifold.

### Other registered geometries

The factory imports several additional geometries through registration, including learned and analytical variants such as:

- `reactive`
- `adaptive`
- `hyperbolic`
- `holographic`
- `hierarchical`
- `spherical`

This document does not restate their full mathematics unless that behavior has been revalidated line by line. For runtime accuracy, the safest claims should center on the geometries already audited directly.

## Metric Tensor In Practice

In the present runtime, `metric_tensor(x)` is used for things such as:

- kinetic-energy computation,
- metric-aware velocity normalization,
- topology-dependent geometric reasoning.

The metric may come back as:

- diagonal per-coordinate weights,
- or a denser matrix form,

depending on the geometry implementation.

So it is safer to document the interface than to overgeneralize one specific tensor shape.

## Projection

`project(x)` is the geometry-side way to map coordinates back onto the manifold.

Examples:

- torus uses wrapped angular projection,
- Euclidean-style geometries usually use identity.

In the current runtime, projection can also be mirrored at the integrator helper level through torus-aware topology resolution, so docs should not pretend projection happens in only one place.

## Distance

`dist(x1, x2)` is geometry-specific.

Examples:

- torus uses wrapped angular distance,
- Euclidean uses norm difference,
- learned geometries may still use simpler approximations unless explicitly overridden.

## Friction And Geometry

Modern GSSM geometries may provide geometry-side friction information.

Examples already validated in code:

- torus returns `(gamma, mu)`
- low-rank returns `(gamma, mu)`

This means friction is no longer a purely separate scalar concept outside geometry. But the final application of friction still belongs to the physics engine.

## Practical Interpretation

The current geometry layer is best understood as:

- the provider of manifold structure,
- the source of curvature-aware acceleration terms,
- and sometimes the source of geometry-conditioned damping signals.

The most important runtime distinction is:

- analytical topology, such as `torus`,
- versus learned geometry, such as `low_rank`.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- every geometry returns only Christoffel symbols,
- geometry selection always follows schema defaults literally,
- the base geometry implementation represents the exact behavior of all subclasses.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/geometry/base.py`
- `gfn/realizations/gssm/geometry/factory.py`
- `docs/gssm/technical/0_architecture/math/geometry/torus.md`
- `docs/gssm/technical/runtime/00-effective-defaults.md`

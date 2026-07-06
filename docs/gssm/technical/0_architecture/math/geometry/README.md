# Geometries

This directory contains the geometry-specific notes for the GSSM runtime.

The safest way to read this folder is:

- use [03_geometry.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/03_geometry.md) for the shared runtime contract,
- then use the files in this folder for geometry-specific details.

## What A Geometry Means In The Current Runtime

A geometry module can provide:

- a metric tensor,
- a curvature or Christoffel-like contribution,
- a projection method,
- a distance function,
- and sometimes a geometry-side friction signal.

Important current caveat:

- modern geometries may return `(gamma, mu)`, not just one tensor,
- the physics engine is still the component that decides how friction is finally applied.

## Most Relevant Geometries Right Now

### `torus`

Documented in [torus.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/torus.md).

This is the most important analytical topology in the current runtime.

Properties:

- bounded through periodic wrapping,
- analytical toroidal curvature,
- optional learnable radii,
- optional torus-aware friction gating.

### `euclidean`

Documented in [euclidean.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/euclidean.md).

Properties:

- flat metric,
- zero Christoffel symbols,
- no wrapping,
- simplest baseline geometry.

### `low_rank`

Documented in [low_rank.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/low_rank.md).

Properties:

- learned low-rank curvature approximation,
- optional CUDA fused path,
- separate friction gate,
- can behave toroidally or Euclidean-like depending on the configured topology.

Important current caveat:

- `low_rank` is not inherently bounded on its own,
- boundedness depends on the active topology behavior, especially whether toroidal projection is in play.

## Selection Guidance

Use `torus` when:

- you want the main analytical bounded topology,
- periodic coordinates are a good fit,
- you want the most explicit toroidal behavior.

Use `euclidean` when:

- you want the simplest flat baseline,
- periodic structure is not needed,
- you want to remove analytical curvature entirely.

Use `low_rank` when:

- you want a learned geometry approximation,
- you care about scaling or CUDA support,
- exact analytical curvature is less important than flexibility or efficiency.

## Important Runtime Note

The geometry factory now prefers:

- the declared `topology.type`

unless a learned override such as `riemannian_type` was explicitly requested.

So this folder should not describe geometry choice as if one learned default always overrides the declared topology.

## Reading Order

1. [torus.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/torus.md)
2. [euclidean.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/euclidean.md)
3. [low_rank.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/geometry/low_rank.md)

## Runtime Cross-References

- `gfn/realizations/gssm/geometry/factory.py`
- `gfn/realizations/gssm/geometry/base.py`
- `docs/gssm/technical/0_architecture/math/03_geometry.md`

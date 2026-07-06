# Low-Rank Geometry

This document describes the **current `LowRankRiemannianGeometry` runtime**.

The authoritative code is:

- `gfn/realizations/gssm/geometry/low_rank.py`

## What It Is In The Current Code

`LowRankRiemannianGeometry` is a learned geometry that approximates curvature through low-rank parameters:

- `U`
- `W`

and contracts them against velocity to produce a curvature-like tensor `gamma`.

The code-level description is more specific than a generic "full Christoffel approximation": it is a bilinear learned contraction over velocity, not an exact symbolic computation of Levi-Civita derivatives.

## Core Learned Parameters

For single-head mode:

- `U`: `[D, R]`
- `W`: `[D, R]`

For multi-head mode:

- `U`: `[H, D, R]`
- `W`: `[H, D, R]`

These are initialized with very small random noise to break symmetry.

## Curvature Path

The main learned contraction is implemented through:

```text
v_r = v @ U
sq  = v_r * v_r
gamma = sq @ W^T
```

or the multi-head equivalent with `einsum`.

So the actual runtime behavior is closer to:

- project velocity into rank space,
- square or bilinearly combine that representation,
- project back to feature space.

This is more faithful than presenting it as a generic dense Christoffel tensor decomposition.

## Connection Method

The class also exposes `connection(v, w, x=None)`, which computes a bilinear contraction:

```text
Gamma(v, w)
```

using both `v` and `w`.

Important current detail:

- `forward(...)` uses the self-connection style path through squared projected velocity,
- while `connection(...)` is the explicit bilinear helper.

So the doc should not pretend there is only one single formula used everywhere.

## Trace Normalization

The current implementation supports:

- `enable_trace_normalization`

When active, `_normalize(gamma)` subtracts the mean in the vector case, and uses a symmetry-preserving diagonal correction in the matrix case.

This is a real runtime feature and affects the curvature term before the final `tanh` clamp.

## Final Clamp

After normalization, the non-CUDA path applies:

```text
gamma = clamp_val * tanh(gamma / clamp_val)
```

So curvature is softly bounded by `curvature_clamp`.

## Friction Path

Low-rank geometry also builds a `FrictionGate`.

Important current runtime behavior:

- low-rank always returns `(gamma, mu)`,
- the physics engine is the single authority on how friction is ultimately applied.

This was an explicit design fix to avoid double-applying damping.

## Topology Interaction

`low_rank` can run under different topological interpretations because it reads:

- `config.topology.type`

If topology is toroidal, gate features become:

```text
[sin(x), cos(x)]
```

If topology is non-toroidal, gate features use raw `x`.

It also changes:

- `project(x)`
- `dist(x1, x2)`

according to whether topology is toroidal.

Important current caveat:

- `low_rank` is a learned geometry family,
- but boundedness and projection behavior still depend on the topology mode it is paired with.

So it is inaccurate to describe low-rank as inherently periodic or inherently bounded by itself.

## Metric Tensor

The current metric path is implicit and diagonal-like:

```text
g_diag ~= sum_r U^2
```

The code returns per-coordinate metric scale derived from `U`, broadcast to the input shape.

This is important because other runtime systems, such as metric-aware velocity normalization, use this diagonal metric approximation directly.

## CUDA Path

The class supports a fused CUDA path through:

- `LowRankChristoffelFunction`
- `low_rank_christoffel_fwd`
- `low_rank_christoffel_bwd`

This is one reason low-rank matters in the current runtime: it has a practical fast path for CUDA-heavy workloads.

## `low_rank_paper`

The same file also defines:

- `PaperLowRankRiemannianGeometry`

This variant changes the internal nonlinear contraction in `forward(...)` and still returns `(gamma, mu)`.

So when documenting low-rank behavior in general, it is worth remembering that the repo currently contains at least two related learned low-rank variants.

## When It Is A Good Fit

Use low-rank when:

- you want a learned geometry rather than a fixed analytical one,
- you care about scalable curvature approximation,
- you want access to the CUDA fused path.

It is less appropriate when:

- you need a closed-form analytical manifold,
- you want the simplest possible flat baseline,
- you want the explicit toroidal geometry rather than a learned approximation.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- low-rank is always bounded,
- low-rank computes exact full Christoffel symbols,
- low-rank returns only one tensor,
- low-rank behavior is independent of topology.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/geometry/low_rank.py`
- `gfn/realizations/gssm/physics/components/friction.py`
- `docs/gssm/technical/0_architecture/math/03_geometry.md`

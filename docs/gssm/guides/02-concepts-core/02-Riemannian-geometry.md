# Riemannian Geometry In GSSM

This guide explains what geometry means in the current GSSM runtime.

For implementation details, use:

- `technical/1_geometry/base.md`
- `technical/0_architecture/math/03_geometry.md`
- `technical/0_architecture/math/geometry/`

## The Main Idea

GSSM evolves latent state on a chosen manifold rather than assuming one flat latent space for every task.

In practical terms, geometry affects:

- how trajectories bend
- how distances are interpreted
- whether coordinates wrap
- whether extra friction can come from the geometry itself

## Metrics vs. Runtime Geometry

In differential geometry, a Riemannian manifold is usually described by a metric tensor that defines local inner products and therefore lengths, distances, and curvature.

That mathematical picture is still useful for intuition, but the current GSSM runtime does not always operate by constructing a full metric tensor and analytically deriving everything from it.

Instead, different geometry implementations provide the behavior needed by the physics engine directly.

## Current Geometry Contract

In the runtime, the geometry module may return:

- `gamma`
- `(gamma, mu)`

where:

- `gamma` is the Christoffel-like curvature contribution
- `mu` is optional geometry-provided friction

That means geometry is part of the actual acceleration path, not just a passive measurement tool.

## Geometry Families

The current factory can build both analytical topologies and learned geometries.

### Analytical Topologies

These are selected primarily through `physics.topology.type`:

- `torus`
- `euclidean`
- `hyperbolic`
- `spherical`

### Learned Geometries

These are selected through `physics.topology.riemannian_type` when explicitly requested:

- `low_rank`
- `reactive`
- `adaptive`
- plus more experimental entries such as `holographic` and `hierarchical`

## Important Runtime Rule

The current factory no longer lets `riemannian_type="reactive"` silently override `topology.type="torus"` just because it appears in the schema defaults.

Current effective behavior:

- analytical topology wins by default
- learned geometry override only wins when explicitly requested

This is why a fresh GSSM model currently builds torus geometry by default.

## Torus As The Default

For a plain:

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

the effective geometry is analytical torus geometry.

That means:

- position is wrapped periodically
- torus-aware readouts and toroidal losses are available when appropriate
- `R`, `r`, `learnable_R`, `learnable_r`, and `toroidal_curvature_scale` are real runtime knobs

## Low-Rank Geometry

Low-rank geometry is the main learned-geometry family used when you want trainable curvature instead of a fixed analytical topology.

Why it matters:

- it reduces parameter and compute cost relative to a dense full-geometry picture
- it exposes `riemannian_rank`
- it has both Python and nearby optimized paths in the broader runtime

What matters most in user-facing terms is not the exact derivation but the fact that low-rank geometry must now be requested explicitly when used together with a declared analytical topology.

## Curvature Bounding

The runtime still includes curvature-control mechanisms such as `curvature_clamp`, but the exact effect depends on the geometry implementation and where the clamp is applied.

So the safe interpretation is:

- curvature is bounded for numerical stability
- the clamp is a runtime safeguard, not a proof of exact Riemannian curvature control

## Geometry Scope

`physics.topology.geometry_scope` affects the dimensional view used by the geometry builder:

- `local`: per-head geometry
- `global`: geometry sees the full model dimension

This setting matters because GSSM layers usually operate on head-partitioned state rather than on one single flat latent tensor.

## Practical Reading Guide

When choosing geometry, ask these questions in order:

1. Does the task naturally want periodic coordinates?
2. Do I want an analytical topology or a learned geometry?
3. Does the loss live in vocabulary space, Euclidean space, or manifold coordinates?
4. Do I need toroidal wrapping, toroidal loss, or identity readout?

That sequence is usually more useful than starting from abstract curvature theory alone.

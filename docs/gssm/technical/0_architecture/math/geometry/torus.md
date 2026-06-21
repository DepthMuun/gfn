# Torus Geometry

This document describes the **current analytical torus geometry implementation** used by GSSM.

The authoritative code lives in:

- `gfn/realizations/gssm/geometry/torus.py`
- `gfn/realizations/gssm/geometry/factory.py`

## What The Runtime Means By "Torus"

The registered geometry key is:

- `torus`

and it maps to:

- `ToroidalRiemannianGeometry`

There is also a separate registered geometry:

- `flat_torus`

which keeps toroidal wrapping but removes analytical Christoffel curvature.

## Coordinate Convention

The analytical torus implementation assumes the last latent dimension is organized in paired coordinates:

```text
(theta_0, phi_0), (theta_1, phi_1), ...
```

This is why the implementation requires:

- even feature dimension

and raises an error otherwise.

So the most faithful statement is:

- the torus geometry is analytical,
- pairwise,
- and generalized across multiple `(theta, phi)` pairs rather than being limited to only one 2D torus.

## Metric Used

For each `(theta, phi)` pair, the implementation follows the standard torus-style metric:

```text
g_theta = r^2
g_phi   = (R + r cos(theta))^2
```

In code, `metric_tensor(x)` fills:

- even indices with `r^2`
- odd indices with `(R + r cos(theta))^2`

## Learnable Radii

The current implementation supports both:

- `learnable_R`
- `learnable_r`

If enabled, `R` and `r` are `nn.Parameter`s.

If disabled, they are stored as non-trainable buffers.

Important current detail:

- the implementation does not "wrap radii to positive" after every optimizer step,
- it simply stores the configured initial values and lets autograd update them.

So it would be inaccurate to document a hard absolute-value postprocessing step that the code does not perform.

## Christoffel / Curvature Contribution

The geometry computes a curvature term `gamma` from:

- `x`
- `v`
- `R`
- `r`
- `toroidal_curvature_scale`

For each `(theta, phi)` pair, the implementation builds:

- a theta component driven by `v_phi^2`
- a phi component driven by `v_phi * v_theta`

The exact expressions are consistent with torus-style Christoffel structure, but the runtime also multiplies them by:

- `toroidal_curvature_scale`

This means curvature strength is explicitly tunable in the current code.

## Return Contract

This is a very important runtime detail.

`ToroidalRiemannianGeometry.forward(...)` does **not** return only a single acceleration tensor. Its current contract is:

```text
return gamma, mu
```

Where:

- `gamma` is the pure curvature contribution
- `mu` is a friction gate

The physics engine is responsible for applying the damping term using that returned friction information.

So the geometry no longer silently mixes curvature and damping into one inseparable tensor.

## Friction Gate

The torus geometry constructs:

- `x_in = [sin(x), cos(x)]`

and feeds it into a `FrictionGate`.

The friction mode comes from:

- `config.stability.friction_mode`

Current supported gate modes come from the friction module, including:

- `static`
- `mlp`

The geometry therefore supports state-aware friction gating, but it does so through the shared friction component rather than through a bespoke torus-only formula written inline in the docs.

## Active Inference And Singularity Modulation

The current torus implementation also contains optional modifiers for:

- active inference reactive curvature scaling
- singularity-related potential gating

When enabled through config, these can rescale `gamma` further.

This means the final torus curvature path is not just the classical textbook Christoffel expression; it can be modulated by additional runtime mechanisms.

## Position Projection

The torus projection method is:

```python
atan2(sin(x), cos(x))
```

This is used to wrap coordinates back into a periodic angular representation.

The same wrapped-distance idea also appears in:

- `dist(x1, x2)`

which computes the norm of the wrapped angular difference.

## CUDA Path

If the optional CUDA extension is available and tensors are on CUDA, the geometry can use:

- `toroidal_cuda.forward(...)`

Otherwise it falls back to the Python / PyTorch implementation.

So the analytical torus path has both:

- a Python fallback,
- an optional CUDA fast path.

## Factory Selection

The geometry factory now prefers:

- declared analytical `topology.type="torus"`

unless the user explicitly requested a learned override through:

- `topology.riemannian_type`

This is one of the key runtime fixes in the current version.

So a plain torus config no longer silently becomes `reactive` just because the schema default contains a learned geometry type.

## `flat_torus`

The same file also defines `FlatToroidalRiemannianGeometry`.

Its behavior is different:

- metric is flat,
- curvature tensor is zero,
- toroidal projection still exists,
- a friction gate is still returned.

This is useful when you want periodic wrapping without the full analytical torus curvature.

## Practical Interpretation

Use analytical `torus` when you want:

- periodic bounded coordinates,
- explicit toroidal curvature,
- learnable radii,
- torus-aware readout behavior.

Use `flat_torus` when you want:

- periodic latent coordinates,
- but not the full analytical curvature term.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- torus geometry is only a single 2D torus with one `(theta, phi)` pair,
- radii are forcibly projected positive after every update,
- friction is just a simple scalar formula hardcoded in the torus file,
- the torus forward path returns only one tensor.

Those claims do not match the current implementation.

## Runtime Cross-References

- `gfn/realizations/gssm/geometry/torus.py`
- `gfn/realizations/gssm/geometry/factory.py`
- `gfn/realizations/gssm/physics/components/friction.py`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

# GSSM Hyperparameters

This document explains how key GSSM hyperparameters behave in the current runtime.

The goal is not to assign magical interpretations such as "every 0.1 means exactly X", because many parameters interact non-linearly with:

- the chosen integrator,
- the active geometry,
- the current velocity norm,
- sequence-dependent external force,
- topology-specific wrapping.

Instead, this guide documents:

- where each parameter lives,
- how the runtime uses it,
- what practical effect to expect,
- which interactions matter most.

## Read This First

For many physics-related parameters, the important unit is not the parameter alone but the combination:

```text
effective update ~= integrator(base_dt, acceleration, friction, topology, velocity)
```

That means:

- doubling `friction` does not produce a fixed universal change,
- lowering `base_dt` changes both stability and how strongly damping is felt per step,
- toroidal geometry changes the meaning of distance and readout features,
- learned geometry and analytical geometry do not react identically to the same scalar setting.

## `friction`

### Config Path

`physics.stability.friction`

### Runtime Role

`ManifoldPhysicsEngine` treats friction as part of the total damping coefficient:

```text
mu_total = friction_fallback + mu_geo
```

Where:

- `friction_fallback` is the scalar config value,
- `mu_geo` is any extra geometry-provided friction gate.

The engine then builds a damping term proportional to velocity:

```text
friction_term = mu_total * v
net_accel = -christoffel - friction_term + external_and_optional_forces
```

### Practical Meaning

- larger `friction` damps momentum faster,
- lower `friction` lets velocity persist longer,
- if it is too high, the system becomes overdamped and trajectories flatten,
- if it is too low, oscillation and velocity growth become more likely.

### Why `0.1` Is Not A Universal Unit

The visible effect of `friction=0.1` depends on:

- `base_dt`,
- the chosen integrator,
- the current `||v||`,
- whether `velocity_friction_scale` is active,
- whether geometry already contributes `mu_geo`.

In leapfrog, for example, friction appears inside divisors like:

```text
v_half = (...) / (1 + 0.5 * dt * mu1 + eps)
v_next = (...) / (1 + dt * mu_avg + eps)
```

So the same friction value damps more aggressively when `dt` is larger.

### Working Heuristic

- `0.0` to `0.01`: very light damping
- around `0.01`: current baseline default
- `0.05` and above: noticeably stronger damping
- `0.1` and above: use carefully, especially with larger `dt`

These are practical ranges, not exact physical units.

## `velocity_friction_scale`

### Config Path

`physics.stability.velocity_friction_scale`

### Runtime Role

If this value is positive, friction increases with normalized velocity magnitude:

```text
v_norm = ||v|| / sqrt(D)
mu_total = mu_total * (1 + velocity_friction_scale * v_norm)
```

### Practical Meaning

- `0.0` disables velocity-dependent friction,
- larger values make fast trajectories self-damp more strongly,
- useful as a safety mechanism against velocity explosions,
- can also suppress legitimate long-range momentum if overused.

### Interaction

This parameter matters much more when:

- velocities already become large,
- `friction` is non-zero,
- the task generates strong external impulses.

## `velocity_saturation`

### Config Path

`physics.stability.velocity_saturation`

### Runtime Role

If `velocity_saturation > 0`, the integrator applies differentiable tanh saturation:

```text
v_sat = tanh(v / saturation) * saturation
```

Otherwise it falls back to hard clamping with `MAX_VELOCITY`.

### Practical Meaning

- `0.0` means "disabled",
- small positive values produce smooth soft-clamping,
- useful when you want to keep gradients differentiable while limiting velocity magnitude,
- too small a value can collapse dynamics into a narrow regime.

## `base_dt`

### Config Path

`physics.stability.base_dt`

### Runtime Role

This is the base integration step used by the selected solver unless a per-step override is supplied.

### Practical Meaning

- larger `base_dt` means bigger moves through state space per integration step,
- smaller `base_dt` means more conservative numerical evolution,
- large values can improve speed but increase instability risk,
- small values improve stability but may slow effective state transport.

### Adaptive Case

If the `adaptive` integrator is used, `base_dt` becomes the upper reference value and the actual step is:

```text
dt_eff = base_dt / (1 + alpha * ||accel||)
```

So the true per-step size shrinks automatically in high-acceleration regions.

## `integrator_type`

### Config Path

`physics.stability.integrator_type`

### Runtime Role

Resolved by `IntegratorFactory`.

Available built-in choices include:

- `leapfrog`
- `yoshida`
- `verlet`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

### Practical Summary

- `leapfrog`: default production baseline, symplectic, efficient
- `yoshida`: higher-order symplectic, more expensive
- `heun`: non-symplectic RK-style method, often simpler for debugging or non-conservative regimes
- `adaptive`: wrapper that changes `dt` dynamically and delegates actual stepping to `base_solver`

### Choosing Between `leapfrog` And `yoshida`

- choose `leapfrog` when you want a stable default and moderate cost,
- choose `yoshida` when you want a higher-order symplectic path and can afford extra compute,
- choose `heun` when preserving symplectic structure is less important than a simpler explicit solver behavior.

## `adaptive_alpha`, `dt_min`, `base_solver`

### Config Path

- `physics.stability.adaptive_alpha`
- `physics.stability.dt_min`
- `physics.stability.base_solver`

### Runtime Role

These only matter when the selected integrator is `adaptive`.

`adaptive_alpha` controls how aggressively `dt` shrinks in high-acceleration regions.

`dt_min` is the lower clamp for the adaptive step.

`base_solver` selects the underlying integrator used once `dt_eff` is computed.

### Practical Meaning

- higher `adaptive_alpha` means stronger responsiveness to curvature and acceleration,
- lower `dt_min` allows more aggressive shrinkage, which can help stability but increase cost,
- `base_solver="leapfrog"` keeps the adaptive path closer to the standard symplectic baseline.

## `topology.type`

### Config Path

`physics.topology.type`

### Runtime Role

This is the primary geometry selector:

- `torus`
- `euclidean`
- `spherical`
- `hyperbolic`
- other registered analytical topologies

### Practical Meaning

- `torus` wraps positions with `atan2(sin(x), cos(x))`,
- toroidal distances and readouts respect periodic boundaries,
- Euclidean space does not wrap and behaves as an unbounded latent space,
- analytical topologies are currently preferred by the factory unless a learned geometry override was explicitly requested.

## `riemannian_type`

### Config Path

`physics.topology.riemannian_type`

### Runtime Role

This is a secondary selector used by `GeometryFactory`.

Important: it does **not** automatically override the declared topology anymore.

### Practical Meaning

- if `topology.type='torus'` and `riemannian_type` was not explicitly requested, torus analytical geometry wins,
- if the user explicitly requests `riemannian_type='low_rank'` or `reactive`, that learned geometry can override the analytical default.

### Why This Matters

This behavior prevents a config from saying "torus" while silently instantiating a learned geometry because of an inherited schema default.

## `toroidal_curvature_scale`

### Config Path

`physics.stability.toroidal_curvature_scale`

### Runtime Role

In analytical torus geometry, this scalar multiplies the Christoffel contribution derived from toroidal coordinates.

### Practical Meaning

- larger values strengthen toroidal curvature forces,
- smaller values make torus dynamics behave closer to a flatter periodic space,
- setting it too high can make toroidal effects dominate the external force.

## `learnable_R` And `learnable_r`

### Config Path

- `physics.topology.learnable_R`
- `physics.topology.learnable_r`

### Runtime Role

In torus geometry, `R` and `r` become trainable parameters when these flags are enabled. Otherwise they are registered as buffers.

### Practical Meaning

- enabling them lets the model learn the torus radii from data,
- disabling them freezes manifold shape and makes experiments more controlled,
- learned radii are useful when topology scale should adapt to the task.

## `initial_spread`

### Config Path

Top-level `initial_spread`

### Runtime Role

Used by `ModelFactory` to initialize the learnable initial position and velocity states:

```text
x0 ~ N(0, initial_spread)
v0 ~ N(0, initial_spread)
```

### Practical Meaning

- larger values inject more initial variation into latent trajectories,
- smaller values start closer to a near-zero manifold state,
- too small can encourage overly uniform early dynamics,
- too large can make the first steps noisy or unstable.

The current default is `0.1`, which is already a stability-minded value compared with older zero-spread conventions.

## `embedding.mode`

### Config Path

`physics.embedding.mode`

### Runtime Role

This controls how inputs are converted into force-like manifold embeddings:

- `lookup`
- `linear`
- `binary`
- `siren`
- `continuous`

### Practical Meaning

- `linear`: bit expansion of token ids plus projection, current default runtime mode
- `binary`: binary mapping normalized to `[-1, 1]`
- `lookup`: standard learned lookup embedding
- `siren`: sinusoidal implicit representation driven by `omega_0`
- `continuous`: direct projection of continuous vectors `[B, T, D_in]`

### Important Note

`continuous` is not just "another token embedding." It bypasses vocabulary lookup and expects real-valued inputs through `continuous_input`.

## `omega_0`

### Config Path

`physics.embedding.omega_0`

### Runtime Role

Passed into the SIREN-based embedding path and used in `SineLayer`.

### Practical Meaning

- higher values increase oscillatory frequency in the implicit embedding,
- lower values make the field smoother,
- this parameter only matters for SIREN-style embedding paths.

## `readout.type`

### Config Path

`physics.readout.type`

### Runtime Role

Resolved by `ReadoutBuilder`.

Available built-in modes:

- `standard`
- `implicit`
- `identity`

### Practical Meaning

- `standard`: categorical logits through a linear projection
- `implicit`: MLP projection from latent state to a target output space
- `identity`: return the latent state directly for latent-space supervision

### Toroidal Detail

For `standard` and `implicit` readouts on torus, the readout does **not** consume raw coordinates directly. It first expands the latent state to:

```text
[sin(x), cos(x)]
```

This makes the readout periodic-aware.

## `holographic`

### Config Path

Top-level `holographic` and `physics.active_inference.holographic_geometry`

### Runtime Role

These are synchronized with logical OR semantics:

```text
final_holographic = top_level_holographic or physics_holographic_geometry
```

### Important Current Behavior

`holographic=True` no longer silently converts `readout.type='standard'` into `identity`.

If you want latent-state output, request:

```text
readout.type = "identity"
```

explicitly.

## `dim`, `heads`, `rank`, `geometry_scope`

### Config Paths

- top-level `dim`
- top-level `heads`
- top-level `rank`
- `physics.topology.geometry_scope`

### Runtime Role

These define the state layout used by layers and geometry.

If `geometry_scope='local'`:

```text
head_dim = dim / heads
dim_total = dim
```

If `geometry_scope='global'`:

```text
head_dim = dim
dim_total = heads * dim
```

### Practical Meaning

- `local` scope partitions the latent dimension across heads,
- `global` scope gives each head a full-dimensional geometric space,
- `rank` controls low-rank geometry and mixer capacity,
- because of config synchronization, the effective default `rank` is currently `16` unless you override it explicitly.

## Safe Documentation Pattern

When documenting a hyperparameter elsewhere in the repo, use this pattern:

1. Name the config path.
2. State the effective default, not just the raw table value.
3. Quote the runtime formula if the parameter is physics-facing.
4. Call out important interactions.
5. Only then add practical heuristics.

That keeps the documentation accurate even when implementation details are subtle.

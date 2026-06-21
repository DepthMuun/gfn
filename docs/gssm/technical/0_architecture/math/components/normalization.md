# Manifold Normalization

This document describes the **current geometry-aware normalization runtime** used by `ManifoldLayer`.

The authoritative files are:

- `gfn/realizations/gssm/physics/normalization.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

## What Exists In The Current Runtime

The current normalization registry exposes:

- `position_torus`
- `position_euclidean`
- `velocity_tangent`
- `velocity_metric`
- `feature_hidden`
- `identity`

The layer normally uses:

- one normalization for position,
- one normalization for velocity,

selected through `ManifoldNormalizationRegistry.get_for_topology(...)`.

## Position vs Velocity

The runtime intentionally separates these two cases.

### Position

Position is normalized according to topology:

- torus -> wrapped angular projection,
- Euclidean -> identity.

### Velocity

Velocity normalization is treated as a tangent-space operation:

- metric-aware if geometry is available,
- otherwise a tangent-space clamp / RMS normalization fallback.

This matches the code more closely than older docs that described position and velocity normalization as one generic process.

## Position Normalization Paths

### Torus position normalization

`TorusPositionNormalization` uses:

```python
atan2(sin(x), cos(x))
```

This is the same wrapped-angle pattern used elsewhere in the runtime for toroidal projection.

### Euclidean position normalization

`EuclideanPositionNormalization` is just identity.

Important current detail:

- the runtime does **not** apply generic RMS-style normalization to Euclidean positions in this registry.

## Velocity Normalization Paths

### `TangentVelocityNormalization`

This path does:

1. hard clamp velocity to `[-MAX_VELOCITY, MAX_VELOCITY]`
2. apply `nn.RMSNorm`

So the current tangent fallback is:

```text
v_out = RMSNorm(clamp(v))
```

### `MetricAwareVelocityNormalization`

If geometry is available, the registry chooses this stricter path.

It uses `geometry.metric_tensor(context_x)` and scales the velocity only when the metric norm exceeds the maximum allowed magnitude.

Important current detail:

- this path does **not** apply RMSNorm after the metric-aware scaling,
- it simply rescales or clamps magnitude in metric space and returns the result.

That is an important difference from the older conceptual docs.

## Registry Selection Logic

The current registry logic is:

```text
if is_velocity:
    if geometry is available:
        use velocity_metric
    else:
        use velocity_tangent
else:
    if topology is torus:
        use position_torus
    else:
        use position_euclidean
```

This is exactly the logic `ManifoldLayer` uses when normalization is enabled.

## How `ManifoldLayer` Uses It

`ManifoldLayer` reads:

- `config.stability.enable_trace_normalization`

If enabled, it builds:

- `norm_x` from topology and geometry,
- `norm_v` from topology and geometry.

If disabled, it uses:

- `identity`

Important current caveat:

- the name `enable_trace_normalization` is historical and a bit misleading,
- enabling it activates the registry-selected normalization stack more broadly, not just one isolated trace-normalization operation.

## Relationship To Velocity Saturation

Velocity saturation also exists in integrators through:

- `stability.velocity_saturation`

That is a separate mechanism from the normalization registry.

So the docs should not collapse these into one single feature:

- registry normalization happens in the layer / dynamics path,
- velocity saturation happens in the integrator path.

## Practical Interpretation

The current normalization system is best understood as:

- topology-aware position cleanup,
- geometry-aware or tangent-aware velocity control,
- selected centrally by the registry,
- injected into `ManifoldLayer` dynamics routing.

It is not a generic "normalize everything everywhere" system.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- metric-aware velocity normalization always ends with RMSNorm,
- Euclidean position uses a learned or RMS-style normalization,
- `velocity_saturation` is the same thing as manifold normalization.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/normalization.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `gfn/realizations/gssm/physics/integrators/base.py`

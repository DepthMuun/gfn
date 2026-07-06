# Singularities

This document describes the **current singularity-handling runtime**.

The authoritative code is:

- `gfn/realizations/gssm/physics/components/singularities.py`
- `gfn/realizations/gssm/physics/engine.py`

## What Exists In The Current Runtime

There are two related components:

- `SingularityGate`
- `SingularityDetector`

The engine path currently uses:

- `SingularityGate`

not the full detector workflow by default.

That distinction matters because older docs tended to describe the detector logic as if it were always active in the main acceleration path.

## `SingularityGate`

`SingularityGate` is the main runtime damping mechanism.

It computes:

```text
gate = sigmoid(slope * (abs(metric_component) - threshold))
```

and can apply that gate to:

- velocity through `damp_velocity(...)`
- force through `damp_force(...)`

So the live runtime singularity behavior is primarily a smooth sigmoid damping gate around a supplied scalar metric measure.

## Engine Usage

Inside `ManifoldPhysicsEngine.compute_acceleration(...)`, singularity damping only happens when:

- singularities are enabled,
- `self.singularity_gate` exists,
- `metric_component` is explicitly passed to the engine call.

Then the engine does:

```text
net_accel = singularity_gate.damp_force(net_accel, metric_component)
```

Important current caveat:

- if no `metric_component` is provided, the singularity gate does not affect the acceleration path.

So the docs should not imply that singularity damping is always automatically active whenever the config flag is enabled.

## `SingularityDetector`

`SingularityDetector` still exists and analyzes a metric tensor by:

- determinant magnitude,
- minimum eigenvalue magnitude,
- taking the minimum of those measures.

It can produce:

- a binary singularity mask,
- or a scalar measure through `get_metric_component(...)`.

Important current caveat:

- this detector is a real component,
- but it is not automatically inserted into the engine's main acceleration path in the validated runtime flow here.

## Configuration Reality

The schema currently exposes:

- `enabled`
- `epsilon`
- `strength`
- `threshold`

In the gate path:

- `threshold` is used directly,
- `strength` becomes `slope = strength * 20.0`.

Important current caveat:

- `epsilon` matters in the detector helper logic,
- not in the basic gate formula itself.

## Practical Interpretation

The safest current description is:

- singularity handling is available,
- the main runtime protection is a smooth damping gate,
- its effect depends on whether a metric-component signal is actually passed through the active path.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- singularity detection is always active in every forward pass,
- the engine always computes determinant and eigenvalues internally before damping,
- enabling the config flag alone guarantees active damping in all acceleration calls.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/components/singularities.py`
- `gfn/realizations/gssm/physics/engine.py`
- `docs/gssm/technical/0_architecture/math/01_physics_engine.md`

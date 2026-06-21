# Constants Reference

This guide lists the constants that are still relevant to the current GSSM runtime and, more importantly, explains how to read them correctly.

## Read This First

Not every value in `constants.py` is an effective model default.

For GSSM, the actual behavior seen by a new model comes from:

- `constants.py`
- `config/schema.py`
- `config/normalizer.py`
- `models/factory.py`
- `geometry/factory.py`

So this file should be read together with:

- `technical/runtime/00-effective-defaults.md`
- `technical/runtime/01-hyperparameters.md`

## Numerical Stability Constants

These values come directly from `gfn/realizations/gssm/constants.py`.

| Constant | Current value | Role |
|---|---:|---|
| `EPS` | `1e-8` | Generic small epsilon used in numerical helpers |
| `EPSILON_STANDARD` | `1e-7` | Standard denominator protection |
| `EPSILON_SMOOTH` | `1e-9` | Extra-small smoothing epsilon |
| `EPSILON_STRONG` | `1e-6` | Stronger numerical safeguard |
| `CLAMP_MIN_STRONG` | `1e-4` | Lower bound for guarded clamp paths |
| `INF` | `1e12` | Large sentinel value |

## Timestep And Stability Constants

| Constant | Current value | Notes |
|---|---:|---|
| `MIN_DT` | `0.001` | Schema default for `stability.dt_min` |
| `MAX_DT` | `1.0` | Schema default for `stability.dt_max` |
| `DEFAULT_DT` | `0.1` | Feeds the schema default for `stability.base_dt` |
| `DEFAULT_FRICTION` | `0.01` | Feeds the schema default for `stability.friction` |
| `DEFAULT_PLASTICITY` | `0.05` | Feeds the schema default for active-inference plasticity |
| `MAX_VELOCITY` | `10.0` | Compatibility constant; not the same as the effective default of `velocity_saturation` |

Important caveat:

- `MAX_VELOCITY = 10.0` exists in constants
- `stability.velocity_saturation` defaults to `0.0` in the schema

That means a fresh config behaves as if velocity saturation is disabled, even though an older reader could mistakenly infer otherwise by looking only at constants.

## Geometry And Topology Constants

| Constant | Current value | Role |
|---|---:|---|
| `CURVATURE_CLAMP` | `5.0` | Schema default for `stability.curvature_clamp` |
| `TOROIDAL_MAJOR_RADIUS` | `1.0` | Low-level torus reference constant |
| `TOROIDAL_MINOR_RADIUS` | `0.3` | Low-level torus reference constant |
| `TOROIDAL_PERIOD` | `2 * pi` | Periodic wrap interval |
| `TOROIDAL_CURVATURE_SCALE` | `0.1` | Constant exists, but the current schema default for `stability.toroidal_curvature_scale` is `0.01` |

Important caveat:

- constants define one toroidal curvature reference value
- the schema currently defaults `stability.toroidal_curvature_scale` to `0.01`

For user-facing configs, prefer the schema-backed runtime value, not the raw constant snapshot.

## Friction And Gating Constants

| Constant | Current value | Role |
|---|---:|---|
| `FRICTION_SCALE` | `0.1` | Global reference scale used by some low-level paths |
| `VELOCITY_FRICTION_SCALE` | `0.01` | Reference constant only |
| `GATE_BIAS_OPEN` | `2.0` | Open-gate initialization bias |
| `GATE_BIAS_CLOSED` | `-2.0` | Closed-gate initialization bias |

Important caveat:

- `VELOCITY_FRICTION_SCALE` in constants is not the effective default used by a fresh config
- `StabilityConfig.velocity_friction_scale` defaults to `0.0`

## Singularity Constants

| Constant | Current value | Role |
|---|---:|---|
| `SINGULARITY_THRESHOLD` | `0.5` | Schema default for singularity threshold |
| `BLACK_HOLE_STRENGTH` | `3.0` | Schema default for singularity strength |
| `SINGULARITY_GATE_SLOPE` | `10.0` | Low-level gate slope constant |

## Topology Name Constants

These are the canonical names used throughout the runtime:

| Constant | Value |
|---|---|
| `TOPOLOGY_TORUS` | `"torus"` |
| `TOPOLOGY_SPHERE` | `"spherical"` |
| `TOPOLOGY_HYPERBOLIC` | `"hyperbolic"` |
| `TOPOLOGY_EUCLIDEAN` | `"euclidean"` |

## Dynamics Mode Constants

These are the canonical registered dynamics labels:

| Constant | Value |
|---|---|
| `DYNAMICS_DIRECT` | `"direct"` |
| `DYNAMICS_RESIDUAL` | `"residual"` |
| `DYNAMICS_MIX` | `"mix"` |
| `DYNAMICS_GATED` | `"gated"` |
| `DYNAMICS_STOCHASTIC` | `"stochastic"` |

## What To Use In Practice

For user configuration, prefer schema-level names and the public API:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=2048,
    physics={
        "stability": {
            "base_dt": 0.1,
            "friction": 0.01,
            "integrator_type": "leapfrog",
        },
        "topology": {
            "type": "torus",
        },
        "dynamics": {
            "type": "direct",
        },
    },
)
```

Avoid building user-facing configuration by copying raw constants mechanically. The effective runtime defaults are determined after normalization and factory selection.

# Dynamics Modes Reference

This guide describes how GSSM merges an integrator proposal back into the persistent state.

It covers the current registered dynamics family:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

## What A Dynamics Mode Does

Inside a `ManifoldLayer`, the physics + integrator path produces proposals for position and velocity. The dynamics mode decides how those proposals become the next actual state.

Conceptually:

```text
proposal -> dynamics mode -> next state
```

This is separate from:

- the embedding, which creates external force
- the geometry, which creates curvature terms
- the integrator, which numerically evolves the system

## Current Runtime Behavior

The registered modes live in `gfn/realizations/gssm/physics/dynamics/` and all inherit from `BaseDynamics`.

Each mode also applies topology-aware normalization through the shared base helpers, so the exact behavior differs between toroidal position state and Euclidean tangent-space velocity state.

## `direct`

### Meaning

`direct` uses the proposal as the next state and then applies the shared normalization logic.

### Mental Model

```text
next_state = normalize(proposal)
```

### When To Use It

- simplest baseline
- closest to "trust the integrator proposal"
- good first choice when debugging a training script

## `residual`

### Meaning

`residual` computes a residual from current state to proposal, normalizes that residual, scales it with a learnable parameter, and adds it back to the current state.

### Mental Model

```text
residual = proposal - current
next_state = current + learned_scale * normalize(residual)
```

On toroidal position state, the residual is computed through wrapped angular difference rather than naive subtraction.

### When To Use It

- when you want smoother updates than `direct`
- when the model should keep more continuity between successive states

## `mix`

### Meaning

`mix` learns an interpolation coefficient between current state and proposal, then applies an additional learnable change scale.

### Mental Model

```text
interpolated = alpha * current + (1 - alpha) * proposal
next_state = current + change_scale * normalize(interpolated - current)
```

On toroidal position state, interpolation is done with circular `sin/cos` blending rather than straight Euclidean averaging.

### When To Use It

- when you want a softer state transition than `direct`
- when you want learned retention without the full gate network of `gated`

## `gated`

### Meaning

`gated` learns a sigmoid gate from the concatenation of current state and proposal, then uses that gate to mix the two.

### Mental Model

```text
g = sigmoid(W[current; proposal])
next_state = normalize(g * proposal + (1 - g) * current)
```

### When To Use It

- when the model should decide contextually how much to preserve
- when the update policy should depend on the current content rather than on a single global mixing scalar

## `stochastic`

### Meaning

`stochastic` adds learnable Gaussian noise to the proposal path before normalization.

### Mental Model

```text
base = proposal or residual-style proposal
next_state = normalize(base + sigma * noise)
```

The implementation keeps `sigma` positive with a softplus transform.

### When To Use It

- when you explicitly want exploration in the state update
- when deterministic proposal merging collapses too aggressively

## Position vs Velocity

GSSM treats position and velocity differently:

- position dynamics can be topology-aware, especially on torus
- velocity dynamics live in tangent-space and are treated as Euclidean in the current runtime

This matters because the same named dynamics mode may behave differently on `x` and `v` due to the shared normalization and topology helpers.

## Selection Heuristics

Start with these simple rules:

| Goal | Suggested mode | Why |
|---|---|---|
| Stable baseline | `direct` | Smallest amount of extra moving parts |
| Smoother carry-over | `residual` | Keeps explicit continuity with current state |
| Learned interpolation | `mix` | Simpler than full gating |
| Content-aware retention | `gated` | Gate depends on current state and proposal |
| Exploration | `stochastic` | Adds controlled noise |

## Minimal Example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    physics={
        "dynamics": {
            "type": "direct",
        }
    },
)
```

## Practical Advice

- Change dynamics mode only after the base loss, readout, and target contract are already correct.
- If you are debugging a task mismatch, `direct` is the easiest mode to reason about.
- If you switch to toroidal supervision or identity readout, re-check whether the chosen dynamics mode is still giving you the right latent behavior.

# Dynamic Systems And Stability

This guide describes the current GSSM dynamics layer and stability knobs at a practical level.

For implementation details, use:

- `technical/0_architecture/math/02_integrators.md`
- `technical/0_architecture/math/08_dynamics.md`
- `technical/runtime/01-hyperparameters.md`

## State Update In GSSM

Each timestep follows the same broad pattern:

1. Convert the input token into a force through the embedding module.
2. Evolve `(x, v)` inside each `ManifoldLayer`.
3. Let the physics engine compute acceleration from geometry, friction, and optional modules.
4. Let the configured integrator update the state.
5. Apply the configured dynamics mode to accept or blend the proposal.
6. Produce logits through the readout hook on `on_timestep_end`.

The important practical consequence is that "dynamics" and "integrator" are not the same thing:

- the integrator decides how the differential update is numerically solved
- the dynamics mode decides how the proposal is merged back into the persistent state

## Integrator Family

The current integrator factory exposes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`

The effective default is `leapfrog`.

### Practical Summary

- `leapfrog`: default and the safest starting point for most training runs
- `verlet`: close to leapfrog, still symplectic, simpler update path
- `yoshida`: fourth-order symplectic option with higher per-step cost
- `forest_ruth`: fourth-order symplectic alternative
- `omelyan`: the current runtime implements a PEFRL-style fourth-order path, not the older second-order description that existed in previous docs
- `rk4` and `heun`: non-symplectic alternatives that are still available when you want a more standard ODE-style baseline

## Friction In The Current Runtime

Friction is centralized in `ManifoldPhysicsEngine`.

The engine combines:

- `config.stability.friction` as the fallback base friction
- optional `mu` returned by the geometry itself
- optional velocity scaling through `velocity_friction_scale`

That means the runtime does not assume a single universal active-inference friction gate on every path.

The effective formula is closer to:

```text
mu_total = friction_fallback + mu_geo
mu_total = mu_total * velocity_scaling_if_enabled
```

with velocity scaling only applied when `velocity_friction_scale > 0`.

Current effective defaults:

- `friction = 0.01`
- `velocity_friction_scale = 0.0`

So a fresh model uses base friction, but velocity-dependent drag is off unless you enable it.

## Timestep And Adaptation

The schema default is:

- `base_dt = 0.1`
- `adaptive = True`
- `adaptive_alpha = 0.1`
- `base_solver = "leapfrog"`

In practice:

- `base_dt` sets the reference step size
- the adaptive path is a timestep wrapper around a base solver
- the dynamic-time plugin is a separate mechanism and should not be confused with the adaptive integrator

There is no runtime-backed universal rule like "use `dt=0.4` for GSSM." Old benchmark-specific recommendations should not be treated as framework defaults.

## Velocity Control

Velocity control is split across different parts of the runtime:

- friction is handled in the physics engine
- topology wrapping is handled by the integrator and boundary helpers
- velocity saturation is available as an integrator-side clamp

The current schema default is:

- `velocity_saturation = 0.0`

That means saturation is disabled by default. Older docs that described a default of `100.0` no longer match the current runtime-backed config.

## Optional Stability Modules

Several optional modules can affect the dynamics:

- `hysteresis`: adds a ghost-force term and keeps internal state across timesteps within a batch
- `singularities`: can damp forces or velocities near problematic metric regions
- `stochasticity`: adds Brownian or OU-style noise, but only when enabled and when the engine receives `dt`
- `curiosity`: adds an exploration force when enabled

These are real runtime modules, but they are off by default.

## Dynamics Modes

GSSM currently registers five dynamics modes:

- `direct`
- `residual`
- `mix`
- `gated`
- `stochastic`

### `direct`

`direct` takes the proposal and normalizes it according to topology.

Use it when you want the cleanest "proposal becomes state" behavior.

### `residual`

`residual` computes a residual between the current state and the proposal, normalizes that residual, scales it with a learnable parameter, and adds it back to the current state.

On toroidal position states, the residual is computed with wrapped angular differences.

### `mix`

`mix` interpolates between current state and proposal with a learnable mixing coefficient.

On toroidal position states, interpolation is done through circular `sin/cos` blending rather than naive linear averaging.

### `gated`

`gated` uses a learnable sigmoid gate over `[current_state; proposal]` and mixes the two state candidates accordingly.

### `stochastic`

`stochastic` adds learnable Gaussian noise on top of the proposal path before normalization.

## Stability Checklist

If training becomes unstable, check these first:

- lower `base_dt`
- keep `integrator_type="leapfrog"` until the rest of the setup is stable
- avoid enabling multiple optional physics modules at once
- verify whether `readout.type`, loss, and target representation actually match the task
- enable velocity-dependent friction only intentionally
- do not assume old benchmark values are global defaults

## Minimal Example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    physics={
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
            "velocity_friction_scale": 0.0,
            "velocity_saturation": 0.0,
        },
        "dynamics": {
            "type": "direct",
        },
    },
)
```

This is a conservative starting point because it stays close to the effective default runtime path.

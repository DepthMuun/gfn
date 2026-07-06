# Integrator Reference

This guide describes the integrators available in the current GSSM runtime and how to choose among them.

For formulas and parameter-level behavior, see:

- `docs/gssm/technical/runtime/01-hyperparameters.md`

## Current Default

The effective runtime default integrator is:

- `leapfrog`

This is resolved from `physics.stability.integrator_type` through `IntegratorFactory`.

It is **not** `yoshida`, and the baseline step size is **not** `0.4`. The current schema-backed default is:

- `base_dt = 0.1`
- `integrator_type = "leapfrog"`

## Available Integrators

The current built-in integrators are:

- `leapfrog`
- `yoshida`
- `verlet`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

## Symplectic Integrators

Symplectic integrators are usually the best fit when you want the latent dynamics to behave like a structured physical flow rather than a generic ODE update.

### `leapfrog`

`leapfrog` is the current default and the safest production baseline.

Why it is a good default:

- second-order symplectic update,
- stable for many sequence tasks,
- lower cost than higher-order symplectic methods,
- explicit support in the current optimized runtime path.

In the Python fallback path, leapfrog also applies damping in a way that depends on both `dt` and total friction, so it behaves better than a naive undamped kick-drift-kick implementation.

Use `leapfrog` when:

- you want a conservative default,
- you are training generative or sequence models,
- you want a good balance between stability and cost.

### `yoshida`

`yoshida` is a higher-order symplectic integrator built from a composition of sub-steps.

Use `yoshida` when:

- you want a more accurate symplectic path,
- you can afford more compute per step,
- you are running geometry-sensitive experiments where the extra order is worthwhile.

Trade-off:

- more expensive than `leapfrog`,
- more internal sub-steps,
- still subject to instability if `base_dt` and damping are poorly chosen.

### `verlet`

`verlet` is another second-order symplectic option.

Use it when:

- you want a simple symplectic baseline distinct from leapfrog,
- you are comparing behavior across second-order integrators.

### `forest_ruth`

`forest_ruth` is a fourth-order symplectic method.

Use it when:

- you want a high-order symplectic alternative to `yoshida`,
- you are benchmarking integrator sensitivity,
- runtime cost is secondary to trajectory quality.

### `omelyan`

`omelyan` is a second-order symplectic method with different coefficients than plain leapfrog.

Use it when:

- you want a symplectic path with slightly different long-horizon error behavior,
- you are doing controlled comparisons among symplectic solvers.

## Non-Symplectic Integrators

These methods can still be useful, but they are no longer described as the primary default path.

### `heun`

`heun` is a second-order explicit trapezoidal Runge-Kutta method.

Use `heun` when:

- you want a simpler explicit solver,
- preserving symplectic structure is less important,
- you are debugging or testing a non-conservative regime.

Trade-off:

- easier to reason about as a generic ODE solver,
- does not preserve symplectic structure,
- can behave differently from leapfrog under the same friction and topology settings.

### `rk4`

`rk4` is the standard fourth-order Runge-Kutta method.

Use `rk4` when:

- you want a familiar high-order explicit baseline,
- you are benchmarking against standard ODE solvers,
- exact symplectic structure is not the priority.

Trade-off:

- not symplectic,
- more compute per step than simple second-order methods.

## Adaptive Integrator

### `adaptive`

`adaptive` is not a standalone physical scheme in the same sense as leapfrog or heun. It is a wrapper that computes:

```text
dt_eff = base_dt / (1 + alpha * ||accel||)
```

and then delegates the actual step to `base_solver`.

Relevant config:

- `physics.stability.integrator_type = "adaptive"`
- `physics.stability.base_solver`
- `physics.stability.adaptive_alpha`
- `physics.stability.dt_min`

Use `adaptive` when:

- you want the timestep to shrink automatically in high-acceleration regions,
- you want more protection against local instability,
- you can tolerate extra variability in effective step size.

The safest pairing is usually:

- `integrator_type = "adaptive"`
- `base_solver = "leapfrog"`

## Choosing An Integrator

### Good default

Use `leapfrog` when you do not have a strong reason to choose otherwise.

### Higher-order symplectic path

Use `yoshida` or `forest_ruth` when:

- you want higher-order symplectic behavior,
- the extra compute cost is acceptable.

### Simpler explicit solver

Use `heun` when:

- you want a non-symplectic solver that is easy to interpret,
- you are debugging interactions among force terms.

### Dynamic timestep

Use `adaptive` when:

- acceleration varies strongly across the trajectory,
- a fixed `base_dt` is either too aggressive in hard regions or too conservative everywhere else.

## Common Interactions

Integrator behavior depends strongly on:

- `physics.stability.base_dt`
- `physics.stability.friction`
- `physics.stability.velocity_friction_scale`
- `physics.stability.velocity_saturation`
- `physics.topology.type`

Two rules are especially important:

- a larger `base_dt` makes all integrators behave more aggressively,
- stronger damping changes leapfrog-like schemes non-linearly because friction enters the update denominators.

## Configuration Examples

### Minimal explicit integrator choice

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    physics={
        "stability": {
            "integrator_type": "leapfrog",
        }
    },
)
```

### Higher-order symplectic example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    physics={
        "stability": {
            "integrator_type": "yoshida",
            "base_dt": 0.05,
        }
    },
)
```

### Adaptive timestep example

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1000,
    physics={
        "stability": {
            "integrator_type": "adaptive",
            "base_solver": "leapfrog",
            "base_dt": 0.1,
            "adaptive_alpha": 0.1,
            "dt_min": 0.001,
        }
    },
)
```

## Troubleshooting

If training becomes unstable:

- reduce `base_dt`,
- prefer `leapfrog` over more aggressive alternatives,
- increase damping carefully through `friction` or `velocity_friction_scale`,
- enable `velocity_saturation` if velocity spikes are the main issue.

If the model becomes too damped or too slow:

- reduce `friction`,
- reduce `velocity_friction_scale`,
- keep `leapfrog` but try a slightly different `base_dt`,
- only then consider switching to another integrator.

## Documentation Rule

When writing about GSSM integrators elsewhere in the repo:

- treat `leapfrog` as the current default,
- avoid stale claims such as `yoshida` default or `dt=0.4` baseline,
- describe integrator choice together with `base_dt` and damping, not in isolation.

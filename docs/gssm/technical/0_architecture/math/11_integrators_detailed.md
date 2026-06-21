# Integrators - Detailed Runtime Notes

This file is now a **runtime-oriented companion index** for the integrator docs.

The solver-specific pages are the canonical source for implementation details:

- [leapfrog.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/leapfrog.md)
- [verlet.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/verlet.md)
- [yoshida.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/yoshida.md)
- [forest_ruth.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/forest_ruth.md)
- [omelyan.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/omelyan.md)
- [rk4.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/rk4.md)
- [heun.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/heun.md)

Use [02_integrators.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/02_integrators.md) for the shared runtime contract.

## Shared Base Contract

All current integrators inherit from `BaseIntegrator` and expose:

```python
def step(self, x, v, force=None, dt=None, steps=1, **kwargs) -> Dict[str, torch.Tensor]:
    ...
```

The important shared helpers are:

- `_get_acceleration(...)`
- `_resolve_topology(...)`
- `_clamp_velocity(...)`
- `_resolve_friction_mu(...)`

Important current runtime detail:

- `_clamp_velocity(...)` supports differentiable tanh saturation when `velocity_saturation > 0`,
- `_resolve_topology(...)` handles torus wrapping directly,
- not every solver uses `_resolve_friction_mu(...)` in the same way.

## Current Solver Families

### Symplectic family

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`

### Non-symplectic family

- `rk4`
- `heun`

### Adaptive wrapper

- `adaptive`

Important current caveat:

- `adaptive` is a wrapper around a base solver with dynamic `dt`,
- not a separate textbook embedded error-estimator family in the current implementation.

## Runtime Differences That Matter

### Leapfrog vs Verlet

- `leapfrog` has an explicit friction-aware split update,
- `verlet` uses the simpler `x + v dt + 0.5 a dt^2` path.

So they are related, but not the same implementation.

### Yoshida vs Forest-Ruth

- both are fourth-order symplectic,
- `yoshida` currently has an optional fused CUDA path for low-rank geometries,
- `forest_ruth` does not expose that extra specialization here.

### Omelyan

Important current correction:

- the current `OmelyanIntegrator` is PEFRL-style **4th order**,
- not the older 2nd-order variant described in stale docs.

### RK4 and Heun

These are non-symplectic, but they still use:

- topology wrapping for predicted and final positions,
- base-class velocity clamping on intermediate or final velocity states.

So they are fully integrated with the runtime safety scaffolding.

## Fast Paths

The current codebase includes fused CUDA specialization at least for:

- `leapfrog`
- `yoshida`

under low-rank CUDA-compatible conditions.

This means the effective runtime behavior of an integrator may depend on:

- geometry family,
- CUDA availability,
- whether external force is present,
- tensor device.

## Practical Ranking

For the current runtime, the safest practical summary is:

- `leapfrog` -> main default
- `yoshida` / `forest_ruth` / `omelyan` -> higher-order symplectic alternatives
- `verlet` -> simpler second-order symplectic alternative
- `rk4` / `heun` -> non-symplectic alternatives
- `adaptive` -> timestep wrapper over another solver

## What This Document Should Not Do

This file should not duplicate full solver formulas and freeze them in place if the implementation changes.

That is why the per-solver pages are now the canonical source for exact runtime behavior.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/base.py`
- `gfn/realizations/gssm/physics/integrators/factory.py`
- `docs/gssm/technical/0_architecture/math/integrators/README.md`

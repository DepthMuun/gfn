# GSSM Integrators

This folder contains solver-specific notes that complement the main runtime overview in [02_integrators.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/02_integrators.md).

## Current Factory Scope

The current integrator factory registers and resolves:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

Important current default:

- the factory default is `leapfrog`

If an unknown integrator key is requested, the runtime falls back to `leapfrog`.

## What This Folder Is Best For

Use these files for:

- solver-specific behavior,
- symplectic vs non-symplectic differences,
- implementation caveats that are easier to explain per integrator.

Use [02_integrators.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/02_integrators.md) for:

- the shared runtime contract,
- the role of `BaseIntegrator`,
- current default selection,
- the difference between `adaptive` and the dynamic-time plugin.

## Most Important Files

Start with:

1. [leapfrog.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/leapfrog.md)
2. [yoshida.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/yoshida.md)
3. [verlet.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/integrators/verlet.md)

Those are the most relevant symplectic paths in the current codebase.

## Practical Guidance

Use `leapfrog` when:

- you want the current standard runtime path,
- you want the documented default,
- you care about stable geometry-aware training.

Use `yoshida` when:

- you want a higher-order symplectic solver,
- you accept higher cost.

Use `adaptive` when:

- you intentionally want the wrapper that rescales `dt` from acceleration magnitude.

Use `rk4` or `heun` when:

- you are intentionally exploring non-symplectic alternatives.

## Important Caveat

Some older descriptions count force evaluations as if every solver were implemented in the pure textbook form.

That is not always the best description of the current runtime, because:

- some solvers include explicit friction handling,
- some have CUDA fused fast paths,
- `adaptive` delegates to a base solver,
- topology wrapping and velocity saturation are handled in shared helper logic.

So this folder should not be read as a purely textbook catalogue detached from the actual implementation.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/factory.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/02_integrators.md`

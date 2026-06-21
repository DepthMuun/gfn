# GSSM Code Analysis

This file is now a **maintained runtime audit summary** rather than a frozen full-code dump.

The older version mixed correct structure with stale implementation claims, outdated defaults, and obsolete module descriptions. The maintained documentation now prefers:

- focused runtime docs,
- component-specific notes,
- explicit caveats where code and schema diverge.

## What This File Is For

Use this page when you want a compact answer to:

- how the current GSSM runtime is assembled,
- which subsystems matter most,
- where the main implementation truth lives.

Use the deeper docs for exact details:

- [00_overview.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/00_overview.md)
- [01_components.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/01_components.md)
- [math/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/README.md)
- [runtime/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/runtime/README.md)

## Current Assembly Path

The current runtime path is:

```text
gfn.create("gssm", ...)
  -> ModelFactory.build()
     -> embedding
     -> manifold layers
        -> integrator
           -> physics engine
              -> geometry
              -> optional physical submodules
        -> mixer
        -> dynamics routing
        -> optional layer plugins
     -> hook-attached readout
     -> optional model plugins
```

Important current correction:

- the readout path is best understood as hook-attached rather than as a permanently hardwired old-style direct model field.

## Main Runtime Modules

### Models

The main model runtime is centered on:

- `models/base.py`
- `models/manifold_layer.py`
- `models/factory.py`

Key realities:

- `BaseModel.forward()` resolves forces from `input_ids` or `force_manual`,
- state can be reused or initialized from learned `x0` and `v0`,
- logits are usually collected from timestep-end hooks,
- the evolution loop can be wrapped by plugins such as checkpointing or adjoint.

### Physics

The main physics runtime is centered on:

- `physics/engine.py`
- `physics/integrators/`
- `physics/components/`
- `physics/dynamics/`

Key realities:

- the engine is the authority on friction application,
- geometry may return `(gamma, mu)` rather than only `gamma`,
- auxiliary modules are optional and config-dependent,
- singularity damping only matters when the relevant metric component is supplied.

### Geometry

The geometry runtime is centered on:

- `geometry/base.py`
- `geometry/factory.py`
- analytical geometries such as `torus`
- learned geometries such as `low_rank`

Key realities:

- geometry selection now respects explicit keys and topology intent better than the older default chain,
- `torus` is no longer silently overridden just because a learned `riemannian_type` default exists.

### Hooks And Plugins

The extension system is centered on:

- `models/hooks.py`
- model plugin builders
- layer plugins

Key realities:

- layer plugins such as `dynamic_time` and `fractal` modify per-layer execution,
- model plugins such as adjoint and checkpointing wrap or extend sequence evolution,
- the evolution loop is intentionally hook-aware.

## Important Runtime Corrections

These were the biggest inaccuracies in the older analysis:

- calling `Omelyan` a second-order method even though the current implementation is PEFRL-style fourth order,
- treating readout as a purely direct old-style module instead of a hook-attached runtime path,
- flattening all dynamics into only `direct/residual/gated` and omitting active runtime modes such as `mix` and `stochastic`,
- implying geometry selection follows schema defaults literally without the newer explicit-key logic,
- describing adaptive integration as if it were a textbook embedded error-estimator path rather than the current timestep wrapper.

## Best Reading Order

If you want a real code-aligned map of the system, read in this order:

1. [00_overview.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/00_overview.md)
2. [01_components.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/01_components.md)
3. [math/09_forward_pass.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/09_forward_pass.md)
4. [math/10_backward_pass.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/10_backward_pass.md)
5. [runtime/00-effective-defaults.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/runtime/00-effective-defaults.md)
6. [runtime/01-hyperparameters.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/runtime/01-hyperparameters.md)

## What This File Should Not Try To Be

This file should not try to be:

- a frozen line-by-line code dump,
- a benchmark report,
- a diagnostic snapshot tied to one historical run,
- or a substitute for the more specific runtime docs.

It is now a navigation and audit summary for the maintained documentation set.

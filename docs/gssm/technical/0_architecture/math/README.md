# GSSM Mathematical Foundations

This folder contains the **mathematical and conceptual layer** of the GSSM documentation.

It is meant to complement, not replace, the runtime-derived documentation in:

- `docs/gssm/technical/runtime/00-effective-defaults.md`
- `docs/gssm/technical/runtime/01-hyperparameters.md`
- `docs/gssm/technical/0_architecture/00_overview.md`

Use this folder when you want to understand:

- the equations behind the physics engine,
- the intuition behind each integrator and geometry,
- how embedding, readout, hooks, and plugins fit together conceptually,
- how gradients and physics-informed losses are supposed to work at a system level.

Use the `technical/runtime/` folder when you need:

- effective defaults,
- config precedence,
- exact factory behavior,
- current hyperparameter semantics,
- code-accurate runtime caveats.

## Scope

This folder is intentionally deeper and more mathematical than the user-facing guides.

Some documents here describe the idealized behavior of a component. When a mathematical description and a current runtime detail diverge, prefer the runtime-derived documents and the code.

## Directory Structure

### Core Physics

| File | Topic | What It Explains |
|------|-------|------------------|
| `01_physics_engine.md` | Physics Engine | Net acceleration, geometric force, friction, auxiliary forces |
| `03_geometry.md` | Geometry Overview | Christoffel symbols, curvature, topology families |
| `04_hysteresis.md` | Hysteresis | Memory mechanism and ghost-force intuition |
| `05_singularities.md` | Singularities | Damping and numerical protection around singular regions |
| `06_stochasticity.md` | Stochasticity | Brownian / OU-style perturbations |
| `07_curiosity.md` | Curiosity | Exploration-oriented auxiliary force |
| `08_dynamics.md` | Dynamics | Routing modes such as direct, residual, mix, and gated |

### Integrators

See `integrators/` for per-integrator notes.

Current runtime-supported integrators include:

- `leapfrog`
- `yoshida`
- `verlet`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

Current effective default:

- `leapfrog`

### Data Flow

| File | Topic | What It Explains |
|------|-------|------------------|
| `forward_pass_conceptual.md` | Forward Pass | High-level intuition for sequence evolution |
| `backward_pass_conceptual.md` | Backward Pass | High-level intuition for gradient flow |
| `09_forward_pass.md` | Forward Pass | Detailed step-by-step runtime walk |
| `10_backward_pass.md` | Backward Pass | Detailed gradient path through the current architecture |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Mixer | `components/mixer.md` | Multi-head mixing and aggregation |
| Embedding | `components/embedding.md` | Token or continuous input -> force mapping |
| Readout | `components/readout.md` | Latent state -> logits or latent output |
| Normalization | `components/normalization.md` | Position / velocity normalization behavior |

### Plugins And System

| Component | File | Purpose |
|-----------|------|---------|
| Dynamic Time | `plugins/dynamic_time.md` | Adaptive timestep intuition |
| Fractal | `plugins/fractal.md` | Recursive micro-manifold refinement |
| Hooks | `system/hooks.md` | Lifecycle injection points used by plugins |
| Factory | `system/factory.md` | How the runtime assembles a complete model |

### Training

| Component | File | Purpose |
|-----------|------|---------|
| Losses | `training/losses.md` | Physics-informed and toroidal loss concepts |
| Optimizers | `training/optimizers.md` | Riemannian optimizers and dual-group optimization |

### Geometry Details

| Geometry | File | Notes |
|----------|------|-------|
| Torus | `geometry/torus.md` | Periodic analytical geometry |
| Euclidean | `geometry/euclidean.md` | Flat unbounded geometry |
| Low-Rank | `geometry/low_rank.md` | Learned geometry approximation |

## Quick Reference

### Fundamental Acceleration Equation

At a conceptual level, the physics engine combines:

$$a_{net} = -\Gamma(x,v) - \mu v + F_{ext} + F_{aux}$$

Where `F_aux` may include:

- hysteresis ghost force,
- stochastic force,
- curiosity force,
- singularity-related damping.

The exact runtime composition lives in the physics engine and depends on which components are enabled.

### State Variables

GSSM evolves:

- position `x`
- velocity `v`

The model processes a force sequence token by token and updates `(x, v)` through each layer.

### Symplectic Integrators

Symplectic methods preserve the phase-space structure better than generic explicit ODE solvers.

Main symplectic methods documented here:

- Leapfrog
- Yoshida
- Verlet
- Forest-Ruth
- Omelyan

Main non-symplectic methods documented here:

- Heun
- RK4

## Reading Guide

### To understand the model from first principles

1. `forward_pass_conceptual.md`
2. `01_physics_engine.md`
3. `integrators/leapfrog.md`
4. `03_geometry.md`
5. `backward_pass_conceptual.md`

### To understand the current runtime path

1. `docs/gssm/technical/0_architecture/00_overview.md`
2. `docs/gssm/technical/runtime/00-effective-defaults.md`
3. `docs/gssm/technical/runtime/01-hyperparameters.md`
4. `09_forward_pass.md`
5. `10_backward_pass.md`

### To understand specific subsystems

| Interest | Read |
|----------|------|
| Physics forces | `01_physics_engine.md` |
| Integrator behavior | `02_integrators.md` and `integrators/` |
| Geometry selection and curvature | `03_geometry.md` and `geometry/` |
| Hooks and extensibility | `system/hooks.md` |
| Model assembly | `system/factory.md` |
| Loss behavior | `training/losses.md` |
| Optimizer behavior | `training/optimizers.md` |

## Important Caveats

### Mathematical truth vs runtime truth

This folder emphasizes mathematical structure and conceptual explanation.

The runtime may still differ in details because of:

- config normalization,
- explicit-key tracking,
- factory heuristics,
- plugin attachment,
- codepaths that prioritize backward compatibility.

### Continuous embedding

The embedding component supports `mode="continuous"`, but the main runtime path still resolves forces from `input_ids` unless `force_manual` or a custom continuous-input path is used.

That means the mathematical description of continuous embedding is valid, but user-facing examples must still be checked against the actual call path.

## Cross-References

- `../00_overview.md`
- `../../runtime/00-effective-defaults.md`
- `../../runtime/01-hyperparameters.md`
- `../../guides/03-reference/03-integrators.md`
- `../../guides/04-guides/02-advanced-configuration.md`

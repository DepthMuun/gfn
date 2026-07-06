# Physical Model

This guide explains the physical intuition behind GSSM using the terms that match the current runtime.

It is intentionally approximate at the conceptual level. For the exact implementation path, prefer:

- `technical/2_physics/engine.md`
- `technical/3_models/manifold_model.md`
- `technical/0_architecture/math/01_physics_engine.md`

## Core State

GSSM evolves a latent state with two parts:

- `x`: position on the chosen manifold
- `v`: velocity or momentum in the tangent space

Each input token is converted into an external force, and the model updates `(x, v)` through geometry-aware dynamics plus a numerical integrator.

At the public API level, the forward pass returns:

```python
logits, (x_final, v_final), state_info = model(input_ids)
```

## Why The Physical Analogy Helps

The physics analogy is useful because it separates different roles clearly:

- geometry bends trajectories
- the physics engine combines curvature, friction, and external force
- the integrator advances the state
- the dynamics mode decides how the proposal becomes the next persistent state

This is more precise than treating GSSM as a generic recurrent block with physics-inspired naming.

## Hamiltonian Intuition

The useful mental model is still Hamiltonian or near-Hamiltonian motion:

- position stores where the latent state is
- velocity stores how that state is moving
- the input injects force
- friction damps motion

In a purely conservative system, motion persists and phase-space volume is preserved.
In the current GSSM runtime, friction is part of the real update path, so the system is better understood as a damped or conformal-symplectic-like state evolution rather than a strict frictionless Hamiltonian system.

## Geometry In The Runtime

Geometry affects the update through Christoffel-like curvature terms.

The current geometry contract may return:

- `gamma`
- `(gamma, mu)`

where:

- `gamma` is the curvature contribution
- `mu` is an optional geometry-produced friction term

The physics engine then combines this with the fallback friction from config.

This matters because the runtime does not assume one single universal learned friction gate on every path.

## External Force

The input side of GSSM is a force-generation path.

Depending on the embedding configuration, the model can derive force from:

- token IDs through `lookup`, `linear`, or `binary` style embeddings
- continuous inputs through `embedding.mode="continuous"`
- SIREN-style implicit coordinate mappings

So the physical metaphor is literal in one important sense: the embedding path is not just feature extraction, it feeds the external force term that drives state evolution.

## Friction And Forgetting

Friction is one of the main ways the model controls persistence.

Low friction:

- allows motion to persist longer
- can help tasks that benefit from trajectory carry-over
- can also cause unwanted drift

Higher friction:

- damps motion faster
- helps the model settle
- can also suppress useful dynamics if pushed too far

The current runtime combines base friction and geometry-returned friction inside `ManifoldPhysicsEngine`.

## Optional Physical Modules

Several optional modules extend the base physical picture:

- hysteresis
- singularities
- stochasticity
- curiosity

These are real runtime components, but they are not part of the default path of a fresh GSSM model.

## Integrator Choice

Once the engine has defined the acceleration, the integrator advances the state.

The current family includes:

- `leapfrog`
- `verlet`
- `yoshida`
- `forest_ruth`
- `omelyan`
- `rk4`
- `heun`

The effective default is `leapfrog`.

That default matters more than older “best integrator” claims copied from older benchmark notes.

## Practical Interpretation

The best way to think about the current GSSM physical model is:

- a geometry-aware latent dynamical system
- driven by input-derived force
- damped by configurable friction
- evolved by an explicit numerical integrator
- optionally regularized by state-aware losses

That is more faithful to the runtime than describing the model as a single fixed Hamiltonian solver or as a universal physics-informed loss machine.

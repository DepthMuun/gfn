# Backward Pass - Conceptual Explanation

This document explains the **current gradient story** of GSSM at a conceptual level.

For exact runtime behavior, the authoritative sources are:

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/components/adjoint.py`
- [10_backward_pass.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/10_backward_pass.md)

## High-Level Picture

In the current runtime, gradients normally flow through standard PyTorch autograd over:

- embedding or manual force input,
- manifold-layer evolution,
- timestep-end readout hooks,
- final loss computation.

So the default conceptual story is:

```text
loss -> logits -> timestep-end hook outputs -> layer evolution -> forces/state -> parameters
```

## Default Backward Path

Without optional plugins, `BaseModel._evolve_sequence()` executes an ordinary differentiable Python loop.

That means gradients flow through:

- every layer call,
- every mixer,
- every dynamics module,
- every integrator step,
- every physics-engine acceleration call,
- any differentiable hook-produced readout tensors.

The runtime does not implement a special handwritten backward pass for the default path.

## Hook-Driven Readout Gradients

Because logits are usually collected from:

- `on_timestep_end`

the backward path conceptually includes:

- gradients from the readout hook outputs,
- back into the current `x` and `v`,
- then through the timestep and layer history.

So the readout is part of the differentiable graph, but its exact structure depends on the hooked module that produced the tensor.

## State Evolution Gradients

Conceptually, gradients backpropagate through:

1. timestep-end readout,
2. the layer stack for that timestep,
3. previous timesteps through recurrent state reuse,
4. the initial state or provided continuation state.

This is the closest conceptual analogue to backpropagation through time in the current runtime.

## Layer-Level Gradient Flow

Inside a manifold layer, the main differentiable blocks are:

- plugin pre-integrate transforms,
- integrator step,
- plugin post-integrate transforms,
- mixer,
- dynamics routing,
- topology wrap,
- plugin finalize transforms.

Important current caveat:

- topology wrap for torus uses differentiable `atan2(sin(x), cos(x))`,
- but some runtime safety operations such as hard clamp fallback can create non-smooth behavior when saturation is not in the differentiable tanh mode.

## Physics-Engine Gradient Flow

Conceptually, gradients flow through:

- geometry curvature terms,
- friction handling,
- external force addition,
- optional auxiliary modules such as hysteresis, stochasticity, curiosity, or singularity gates when active.

Important current caveat:

- not every optional physics module is active in every path,
- and some components require extra state or inputs to matter at runtime.

So the safest doc language is:

- gradients flow through whichever physics modules are actually instantiated and invoked in the current model path.

## Initial State And Force Parameters

In the current default initialization path:

- `x0` and `v0` are learnable parameters,
- `x` receives optional random perturbation from `initial_spread`,
- gradients still flow back to the underlying learned parameters through the differentiable graph.

If force comes from embeddings:

- gradients flow into embedding parameters through the force sequence.

If force comes from `force_manual`:

- gradients flow into that tensor if it requires grad,
- but not into the embedding path because it was bypassed.

## Adjoint Path

The current runtime does include an optional adjoint plugin:

- built by `AdjointBuilder`,
- registered through `model.hooks`,
- attached to `wrap_evolution`.

Important current caveat:

- the implementation currently imports `odeint_adjoint` if available, but inside the wrapper it calls `torchdiffeq.odeint` directly for the actual sequence solve,
- the plugin also fixes `method='euler'` to match the discrete-step interpretation.

So the most faithful statement is:

- there is an adjoint-style optional evolution wrapper in the runtime,
- but the current implementation should not be documented as a pure canonical `odeint_adjoint` path with no caveats.

## Checkpointing And Wrapped Evolution

Because `BaseModel._evolve_sequence()` exposes:

- `wrap_evolution`

other memory-oriented or evolution-wrapping plugins can also alter the backward path indirectly by changing how the forward graph is constructed.

So the backward story is partly hook-configurable, not only hardcoded.

## Practical Failure Modes

The most realistic conceptual failure modes are:

- vanishing gradients across long evolution horizons,
- exploding gradients under large timesteps or weak velocity control,
- topology or saturation interactions that make optimization rougher,
- memory pressure when full trajectories are stored.

These are runtime-shaped issues, not just abstract neural-network pathologies.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- GSSM always uses a custom handwritten backward solver,
- the adjoint path is always active,
- the current adjoint implementation is a perfect one-to-one `odeint_adjoint` path,
- gradients always flow through every optional physics module regardless of config.

Those claims do not match the current runtime.

# Backward Pass - Current Gradient Flow

This document describes how gradients flow through the **current GSSM runtime**.

The relevant implementation lives in:

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `gfn/realizations/gssm/models/components/adjoint.py`
- `gfn/realizations/gssm/training/optimizer.py`
- the selected integrator, geometry, and loss modules

## 1. Default Gradient Mechanism

The default training path uses standard PyTorch autograd over the unrolled discrete forward pass.

In the common case:

```python
logits, (x_final, v_final), state_info = model(input_ids)
loss = criterion(logits, targets)
loss.backward()
```

There is no hand-written backward for the default discrete evolution loop.

## 2. Main Gradient Path

In the standard generative path, gradients flow roughly like:

```text
loss
  <- logits
  <- readout plugin
  <- latent state trajectory
  <- manifold layers
  <- integrator steps
  <- physics engine
  <- embedding / force path
  <- x0, v0 and other trainable parameters
```

## 3. Differentiable Parts Of The Forward Pass

### Embedding / force encoding

When the model uses:

```python
all_forces = self.embedding(input_ids)
```

gradients flow into the active embedding path, for example:

- embedding table weights,
- bit-projection weights,
- SIREN weights,
- `impulse_scale`.

### Initial state

The learned state seeds:

- `x0`
- `v0`

participate in gradient flow because the forward pass expands them into the trajectory.

If `initial_spread > 0`, random noise is added to `x`, but gradients still flow to the underlying parameter through the additive expression.

### Masked force application

The timestep force:

```python
force = fs[:, i] * ms[:, i]
```

is differentiable with respect to `fs`.

### Layer evolution

Each layer call is differentiable as long as the chosen submodules use differentiable tensor ops, which is the case for the normal runtime path.

## 4. Gradient Flow Through `ManifoldLayer`

Inside each layer, gradients flow through:

1. reshape / view operations,
2. layer plugins such as `pre_integrate` or `post_integrate`,
3. the integrator step,
4. the mixer,
5. the dynamics routing blocks,
6. topology wrapping,
7. layer finalization plugins.

### Reshapes

Operations such as:

```python
x.reshape(...)
v.reshape(...)
```

do not block gradient flow.

### Mixer and dynamics blocks

The mixer and routing modules are ordinary differentiable neural components, so gradients reach:

- mixer parameters,
- residual / gated / stochastic routing parameters,
- any geometry-aware normalization used inside those paths.

### Topology wrapping

Toroidal wrapping uses differentiable trigonometric operations such as:

```python
torch.atan2(torch.sin(x), torch.cos(x))
```

This keeps the path differentiable almost everywhere while maintaining periodic coordinates.

## 5. Gradient Flow Through The Integrator

The selected integrator differentiates through repeated tensor operations on:

- `x`
- `v`
- `force`
- effective timestep `dt`

Examples of differentiable operations used in the current integrators:

- additions and weighted sums,
- multiplications by `dt`,
- divisions by damping terms,
- velocity clamping,
- topology wrapping,
- calls into `_get_acceleration(...)`.

So the backward pass naturally propagates through the whole discrete solver stack.

## 6. Gradient Flow Through The Physics Engine

The physics engine computes acceleration from:

- geometry contribution,
- friction,
- optional external force,
- optional auxiliary modules.

That means gradients can flow into:

- learnable geometry parameters such as `R` and `r`,
- friction-related coefficients when they are parameterized,
- hysteresis modules,
- curiosity modules,
- stochastic module parameters,
- the incoming force representation.

Important nuance:

- random samples themselves are not differentiable,
- but learnable parameters that scale or shape those samples can still receive gradients.

## 7. Readout Gradient Path

The outer model loop does not directly call `self.readout(...)`.

Instead, readout is usually injected via:

- `ReadoutPlugin`
- registered on `on_timestep_end`

So the effective path is:

```text
loss <- timestep logits <- readout plugin <- latent state
```

For `CategoricalReadout`, gradients reach:

- the linear projection weights,
- the latent state before projection,
- and therefore the full preceding trajectory.

For `IdentityReadout`, gradients bypass the categorical projection and go directly to the latent state, so the loss must be compatible with that latent representation.

## 8. Optional Wrapped Evolution: Adjoint Plugin

The codebase includes an `AdjointPlugin` that hooks into:

- `wrap_evolution`

and replaces the default evolution function with an ODE-style wrapper when `torchdiffeq` adjoint support is importable.

Important current implementation detail:

- the plugin is named and registered as an adjoint wrapper,
- but the actual wrapper currently calls `torchdiffeq.odeint` standard integration internally.

So the most accurate description is:

- optional ODE-wrapped evolution exists,
- it changes the forward/backward path,
- but the present implementation is not a simple guarantee of pure `odeint_adjoint` memory behavior.

## 9. What `state_info` Means For Losses

The current forward path returns a `state_info` dictionary that includes:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

This matters because some loss terms depend on fields that are **not** populated by default.

### Example: `PhysicsLoss`

`PhysicsLoss` can use:

- `christoffels`
- `x_seq`
- `v_seq`

But in the default model forward path:

- `x_seq` and `v_seq` are available,
- `christoffels` are not populated automatically.

So the geodesic component only becomes active if some custom path or plugin explicitly provides those tensors in `state_info`.

## 10. Parameters That Commonly Receive Gradients

The exact set depends on config, but typical trainable parameters include:

- embedding parameters
- `impulse_scale`
- `x0`
- `v0`
- mixer weights
- dynamics routing parameters
- geometry parameters such as `R` and `r` when learnable
- readout weights for non-identity readouts
- gate parameters used by dynamic-time or routing components

## 11. Operations That Do Not Carry Gradients Normally

Examples of things that do not themselves provide gradient signal:

- random draws such as `torch.randn_like(...)`
- boolean conditions
- control-flow branches
- non-selected hook results

That said, tensors used inside those branches may still influence later differentiable computations.

## 12. Checkpointing And Memory

The current codebase also supports checkpoint-style tooling through plugins and training utilities, but checkpointing is not the same mechanism as the adjoint wrapper.

At a high level:

- checkpointing trades compute for memory by recomputing forward segments,
- the adjoint wrapper changes the evolution path itself.

These two ideas should not be conflated in documentation.

## 13. Practical Gradient Failure Modes

### Vanishing gradients

Typical causes in the current runtime:

- long unrolled sequences,
- overly damped dynamics from strong friction,
- very small effective timesteps,
- weak signal in the force path.

### Exploding gradients

Typical causes:

- large timesteps,
- weak velocity control,
- unstable geometry / solver combinations,
- long sequence-depth products without clipping.

### Misleading physical-loss expectations

A frequent documentation mistake is to assume that every physics loss component is active just because the class exists.

In practice, the backward effect depends on whether the forward path actually provided the needed state tensors.

## 14. Backward Summary

```text
loss.backward()
  -> readout parameters or latent-output loss
  -> latent trajectory
  -> per-layer mixer / dynamics / integrator / plugins
  -> physics engine and geometry
  -> force encoding path
  -> learned initial state and other trainable constants
```

## Runtime Cross-References

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `gfn/realizations/gssm/models/components/adjoint.py`
- `gfn/realizations/gssm/losses/physics.py`
- `gfn/realizations/gssm/training/optimizer.py`

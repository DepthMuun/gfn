# Forward Pass - Current Runtime Data Flow

This document describes the **current forward path** implemented in:

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `gfn/realizations/gssm/models/components/readout.py`
- the selected integrator and physics engine

## High-Level Structure

At runtime, the default forward path is:

```text
input_ids or force_manual
  -> force sequence
  -> initial state (x, v)
  -> sequence evolution over timesteps
       -> per-layer manifold updates
  -> hook-based readout
  -> logits + final state + state_info
```

## Entry Point: `BaseModel.forward(...)`

The current public model forward accepts:

```python
forward(
    input_ids=None,
    attention_mask=None,
    state=None,
    force_manual=None,
    **kwargs,
)
```

and returns:

```python
(logits, (x_final, v_final), state_info)
```

## 1. Force Resolution

The current runtime resolves forces in this order:

```python
if force_manual is not None:
    all_forces = force_manual
elif input_ids is not None:
    all_forces = self.embedding(input_ids)
else:
    raise ValueError(...)
```

So the default path is still:

- token ids in,
- `FunctionalEmbedding` produces continuous forces out.

Important current caveat:

- even though `FunctionalEmbedding` supports `mode="continuous"`,
- the main path still calls `self.embedding(input_ids)` unless you provide a compatible alternative path such as `force_manual`.

## 2. Attention Mask Handling

The current code builds:

```python
mask = attention_mask.unsqueeze(-1).float()
```

when an attention mask is supplied, otherwise it creates an all-ones mask.

That means masking happens at the force level through:

```python
force = fs[:, i] * ms[:, i]
```

inside the timestep loop.

## 3. Batch Start Hook

Before state initialization, the model triggers:

```python
self.hooks.trigger("on_batch_start", batch_size=batch_size, device=all_forces.device)
```

This is a model-level lifecycle event, not a layer plugin call.

## 4. State Initialization

State initialization happens in one of three ways.

### External state

If `state` is provided:

```python
x, v = state
```

### Hook override

If no explicit state is passed, the model gives `state_init` a chance to override initialization:

```python
init_res = self.hooks.trigger("state_init", batch_size=batch_size)
if init_res:
    x, v = init_res[-1]
```

### Default learned initialization

Otherwise, the model expands learnable parameters:

```python
x = self.x0.expand(batch_size, self.x0.shape[1], self.x0.shape[2])
v = self.v0.expand(batch_size, self.v0.shape[1], self.v0.shape[2])
```

and optionally adds noise to `x` when `initial_spread > 0`.

Current shapes:

- `x0`, `v0`: `[1, heads, head_dim]`
- expanded state: `[B, heads, head_dim]`

## 5. Evolution Function Selection

`BaseModel.forward()` does not always call `_evolve_sequence()` directly.

Instead, it first lets hooks wrap the evolution function:

```python
wrapped_evolution = self.hooks.trigger("wrap_evolution", evolution_fn=evolve_fn)
if wrapped_evolution:
    evolve_fn = wrapped_evolution[-1]
```

This is how the optional adjoint plugin replaces the default discrete loop with its own wrapped evolution path.

## 6. Default Sequence Evolution Loop

In the default path, `_evolve_sequence()` iterates over timesteps:

```python
for i in range(l_seq_len):
    force = fs[:, i] * ms[:, i]
```

At each timestep, the model:

1. triggers `on_timestep_start`,
2. runs all manifold layers,
3. triggers `on_timestep_end`,
4. stores logits and state snapshots.

## 7. Timestep Start Hook

The current loop allows `on_timestep_start` to modify the force.

If a callback returns:

- a tensor -> it is added to the force,
- a dict containing `"force"` -> it replaces the force.

This makes `on_timestep_start` the main model-level hook for force preprocessing.

## 8. Layer Stack

For each timestep, the model iterates through:

```python
for layer in self.layers:
```

and around each layer call it triggers:

- `on_layer_start`
- `on_layer_end`

The layer itself remains the main site of physics evolution.

## 9. `ManifoldLayer.forward(...)`

Each `ManifoldLayer` receives:

- `x`
- `v`
- `force`

with either:

- batch shape `[B, H, D]`, or
- sequence shape `[B, S, H, D]`

and normalizes everything internally to `[B_eff, H, D]`.

### 9.1 Force reshaping

The exact reshaping depends on:

- input rank,
- `geometry_scope`,
- whether force arrives already partitioned by head or flattened across heads.

So it is not always accurate to think of force as a single fixed `[B, D] -> [B, H, D]` reshape. The code handles:

- 2D force `[B, H*D]`
- 3D force `[B, H, D]`
- sequence forms of both
- global-scope force broadcast

### 9.2 Layer plugins: `pre_integrate`

The layer keeps a `ModuleDict` of layer plugins and applies:

```python
for plugin in self.plugins.values():
    x_3d, v_3d, dt_eff = plugin.pre_integrate(x_3d, v_3d, dt_eff, f_3d)
```

This is where dynamic time-step logic lives in the current runtime.

Important distinction:

- this is **not** the same as `HookManager`,
- these are layer plugins called directly by `ManifoldLayer.forward()`.

### 9.3 Integrator step

The core update is:

```python
res = self.integrator.step(x_3d, v_3d, force=f_3d, dt=dt_eff)
x_stepped, v_stepped = res["x"], res["v"]
```

The integrator is selected by config and may be:

- `leapfrog`
- `yoshida`
- `verlet`
- `forest_ruth`
- `omelyan`
- `heun`
- `rk4`
- `adaptive`

### 9.4 Layer plugins: `post_integrate`

After the integrator step:

```python
for plugin in self.plugins.values():
    x_stepped, v_stepped = plugin.post_integrate(...)
```

### 9.5 Mixing

The stepped states are then mixed:

```python
x_mix, v_mix = self.mixer(x_stepped, v_stepped)
```

The mixer is a required layer component, not a plugin.

### 9.6 Dynamics routing

The mixed proposal is combined with the reference state through:

- `self.dynamics_x`
- `self.dynamics_v`

depending on the configured dynamics type.

The current code supports flat routing over reshaped head states and then restores `[B_eff, H, D]`.

### 9.7 Topology wrapping

After routing:

```python
x_next = self.integrator._resolve_topology(x_next)
```

For toroidal coordinates this wraps angles back into a valid periodic representation.

### 9.8 Layer plugins: `finalize`

Finally:

```python
for plugin in self.plugins.values():
    x_next, v_next = plugin.finalize(x_next, v_next)
```

This is where the current fractal plugin would act if enabled and fully configured.

## 10. Physics Engine Contribution

Inside the integrator, acceleration ultimately comes from the physics engine:

```python
net_accel = -christoffel - friction_term
if force is not None:
    net_accel = net_accel + force
```

and then optionally adds enabled auxiliary modules such as:

- hysteresis
- stochasticity
- curiosity

The exact combination depends on the instantiated physics engine and config.

## 11. Readout Generation

The current model does not call a readout module directly in the outer forward loop.

Instead, readout is hook-driven:

```python
step_res = self.hooks.trigger("on_timestep_end", x=local_x, v=local_v)
for r in step_res:
    if isinstance(r, torch.Tensor):
        l_logits.append(r)
```

In the common path, `ReadoutPlugin` provides those tensors.

### Categorical readout

`CategoricalReadout`:

- flattens `[B, H, D]` to `[B, H*D]`,
- uses `[sin(x), cos(x)]` features for torus,
- uses raw latent features for Euclidean-type readout,
- returns `[B, vocab_size]`.

## 12. Sequence Outputs

The evolution loop collects:

- timestep logits
- `x` states
- `v` states

and stacks them into:

- `logits`: `[B, S, V]`
- `x_seq`: `[B, S, H, D]`
- `v_seq`: `[B, S, H, D]`

## 13. Returned `state_info`

The current forward path returns a `state_info` dictionary containing:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

This matters because downstream training losses only have access to what the forward path actually stores here.

## 14. Forward Summary

```text
token ids or force_manual
  -> force sequence
  -> initial (x, v)
  -> optional wrapped evolution function
  -> for each timestep:
       -> timestep-start hooks
       -> for each layer:
            -> layer-start hooks
            -> layer plugin pre_integrate
            -> integrator step
            -> layer plugin post_integrate
            -> mixer
            -> dynamics routing
            -> topology wrapping
            -> layer plugin finalize
            -> layer-end hooks
       -> timestep-end hooks
  -> stack logits and state trajectories
  -> batch-end hooks
  -> return logits, final state, state_info
```

## Runtime Cross-References

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- `gfn/realizations/gssm/models/components/readout.py`
- `gfn/realizations/gssm/physics/engine.py`
- `docs/gssm/technical/0_architecture/math/system/hooks.md`

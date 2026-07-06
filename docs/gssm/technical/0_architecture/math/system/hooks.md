# Hooks System

This document describes the **current HookManager-based lifecycle** used by `ManifoldModel`.

It is important to distinguish two extension systems in the current runtime:

1. model-level hooks managed by `HookManager`,
2. layer-level plugins attached directly to `ManifoldLayer.plugins`.

They are related, but they are not the same mechanism.

## What Hooks Are

`HookManager` stores named callback lists and lets model components inject logic into the forward pass without rewriting the whole outer loop.

At runtime:

- callbacks are registered by name,
- `trigger(...)` executes all callbacks in registration order,
- non-`None` return values are collected into a list.

## Hook Names Defined By `HookManager`

The current manager defines these hook slots:

| Hook Name | Defined in manager | Triggered by current model path | Purpose |
|-----------|--------------------|----------------------------------|---------|
| `pre_forward` | yes | no in current `BaseModel.forward()` | reserved slot, currently unused by the main forward path |
| `state_init` | yes | yes | override initial `(x, v)` |
| `wrap_evolution` | yes | yes | replace or wrap `_evolve_sequence()` |
| `on_batch_start` | yes | yes | batch-level setup |
| `on_timestep_start` | yes | yes | inspect or modify force before layer stack |
| `on_layer_start` | yes | yes | inject layer kwargs or observe pre-layer state |
| `on_layer_end` | yes | yes | inspect post-layer state |
| `on_timestep_end` | yes | yes | produce logits or collect timestep outputs |
| `on_batch_end` | yes | yes | cleanup or final plugin reporting |

Important current detail:

- `pre_forward` exists in the manager but is not triggered by the current `BaseModel.forward()` implementation.

## Where Hooks Are Triggered

### In `BaseModel.forward()`

The current runtime triggers:

- `on_batch_start`
- `state_init`
- `on_batch_end`

### In `BaseModel._evolve_sequence()`

The current runtime triggers:

- `on_timestep_start`
- `on_layer_start`
- `on_layer_end`
- `on_timestep_end`
- `wrap_evolution`

### In the adjoint wrapper

If the optional adjoint plugin is active, the wrapped evolution path reproduces:

- `on_timestep_start`
- `on_layer_start`
- `on_layer_end`
- `on_timestep_end`

inside its ODE-style evolution wrapper.

## How Registration Works

Plugins register callbacks through:

```python
def register_hooks(self, manager):
    manager.register("on_timestep_end", self.on_timestep_end)
```

The readout system is the main example:

- `ReadoutPlugin.register_hooks(...)`
- registers on `on_timestep_end`
- returns a tensor of logits from the current latent state

## How Results Are Used

### `state_init`

`state_init` can replace the default learned initial state:

```python
init_res = self.hooks.trigger("state_init", batch_size=batch_size)
if init_res:
    x, v = init_res[-1]
```

The last returned value wins.

### `on_timestep_start`

This hook can modify the force used at the current timestep.

The current runtime accepts:

- a tensor, which is added to the force,
- a dict containing `"force"`, which replaces the current force.

### `on_layer_start`

This hook is mainly used to mutate `layer_kwargs` before calling the layer.

The runtime passes:

- `layer`
- `layer_kwargs`
- `x`
- `v`

### `on_layer_end`

This hook is observational in the current model loop. The runtime triggers it with:

- `layer`
- `x`
- `v`
- `extra_info`

but does not directly use returned values to mutate state.

### `on_timestep_end`

This is the main output-production hook.

The base loop does:

```python
step_res = self.hooks.trigger("on_timestep_end", x=local_x, v=local_v)
for r in step_res:
    if isinstance(r, torch.Tensor):
        l_logits.append(r)
```

So any callback that returns a tensor at this stage can contribute timestep outputs.

### `wrap_evolution`

This hook can replace the whole sequence evolution function.

Current main use:

- `AdjointPlugin`

The model takes the last returned wrapper:

```python
wrapped = self.hooks.trigger("wrap_evolution", evolution_fn=evolve_fn)
if wrapped:
    evolve_fn = wrapped[-1]
```

## Hooks vs Layer Plugins

This distinction is critical for the current codebase.

### HookManager plugins

These are model-level lifecycle extensions, such as:

- readout plugin,
- adjoint plugin,
- pooling plugin,
- checkpointing plugin,
- lensing plugin.

They interact with `ManifoldModel.hooks`.

### Layer plugins

These are attached directly to `ManifoldLayer.plugins` and do **not** use `HookManager`.

Current examples:

- `DynamicTimePlugin`
- `FractalPlugin`

They expose methods such as:

- `pre_integrate`
- `post_integrate`
- `finalize`

and are called manually inside `ManifoldLayer.forward()`.

Important current detail:

- the `LayerPlugin` base class also defines `pre_mix` and `post_mix`,
- but the current `ManifoldLayer.forward()` does not call those methods.

## Common Current Use Cases

### Readout

`ReadoutPlugin` registers `on_timestep_end` and converts the current latent state into:

- categorical logits,
- implicit readout output,
- or identity output.

### Adjoint evolution wrapping

`AdjointPlugin` registers `wrap_evolution` if `torchdiffeq` adjoint support is importable.

Important current implementation detail:

- the plugin is gated on `odeint_adjoint` availability,
- but inside the wrapper it currently calls `torchdiffeq.odeint` standard integration for the actual trajectory solve.

So the runtime behavior should be described as:

- optional ODE-wrapped evolution path,
- implemented through the adjoint plugin interface,
- not a guaranteed pure adjoint-memory path in every environment.

### State initialization

Custom initialization logic can be injected through `state_init`.

This is the cleanest hook for:

- persistent state warm starts,
- controlled initialization experiments,
- custom learned priors.

## Execution Order

Callbacks execute in registration order.

```python
manager.register("on_timestep_end", callback_a)
manager.register("on_timestep_end", callback_b)
```

This yields:

```text
callback_a -> callback_b
```

The model usually uses the collected list directly or applies "last one wins" logic depending on the hook.

## What Hooks Do Not Do

Hooks are not a full replacement for core architecture changes.

In the current runtime they do **not** automatically:

- rewrite the layer integration algorithm,
- mutate layer output on `on_layer_end`,
- replace layer plugin behavior,
- make `pre_forward` active in the base path.

## Practical Guidance

Use hooks when:

- you want model-level extensibility,
- you want timestep outputs such as logits or pooled summaries,
- you want to wrap the outer evolution loop,
- you want custom state initialization.

Use layer plugins when:

- the intervention belongs inside a single layer,
- you need to alter `dt`, post-integrator state, or layer finalization,
- the change is naturally expressed as `pre_integrate` / `post_integrate` / `finalize`.

## Runtime Cross-References

- `gfn/realizations/gssm/models/hooks.py`
- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/components/readout.py`
- `gfn/realizations/gssm/models/components/adjoint.py`
- `gfn/realizations/gssm/models/plugins/__init__.py`
- `gfn/realizations/gssm/models/manifold_layer.py`

# Hysteresis

This document describes the **current `HysteresisModule` runtime**.

The authoritative code is:

- `gfn/realizations/gssm/physics/components/hysteresis.py`
- `gfn/realizations/gssm/physics/engine.py`

## What It Does

The hysteresis module provides a stateful memory signal that can inject a ghost force into the physics engine.

In the current runtime:

- the engine owns the module if hysteresis is enabled,
- the module stores state across timesteps,
- the engine can reset that state between batches through `reset_hysteresis()`.

## Current Runtime Contract

The main forward path is:

```text
state <- update_state(state, x, v, topo_id)
ghost_force <- get_ghost_force(state)
return ghost_force
```

The physics engine then adds that ghost force into the net acceleration.

## State Representation

The module stores:

- `state`
- `last_x`
- `last_v`

as registered buffers.

Important current detail:

- this is true stateful runtime memory, not a purely stateless transformation of the current input.

## Feature Extraction

The update uses position-dependent features plus velocity.

For torus:

- features are `[sin(x), cos(x)]`

For Euclidean:

- features are `[x, 0]`

Then velocity is concatenated, so the update input is:

```text
[x_feat, v]
```

This is why the update weight shape is based on `dim * 3`.

## State Update

The current update path is:

```text
update = tanh(W_update * [x_feat, v] + b_update)
state_next = state_prev * decay + update
```

Important current caveat:

- the decay actually used inside `update_state(...)` defaults to `DEFAULT_HYSTERESIS_DECAY = 0.95`,
- the engine path does not currently pass `config.hyst_decay` into `update_state(...)`.

So although the schema still exposes `hyst_decay`, the present runtime path does not wire that config value into the actual update call.

## Ghost Force Readout

The current ghost-force path is:

```text
force = state @ readout_w^T + readout_b
ghost_force = force * GHOST_FORCE_SCALE
```

where:

- `GHOST_FORCE_SCALE` is currently `0.1`.

If `ghost_force_enabled` is false, the module returns zero ghost force.

## Configuration Reality

The schema currently exposes:

- `enabled`
- `ghost_force`
- `hyst_decay`
- `hyst_update_w`
- `hyst_update_b`
- `hyst_readout_w`
- `hyst_readout_b`

Important current caveat:

- in the runtime path validated here, only `enabled` and `ghost_force` are actually used by `HysteresisModule.from_config(...)`,
- the other hysteresis config knobs are still present in schema but are not wired into the current module construction path.

So the docs should not pretend those values are all actively controlling the module today.

## Topology Awareness

The module is topology-aware through:

- `topo_id = 1` for torus
- `topo_id = 0` for Euclidean

This affects feature extraction, not a completely different module implementation.

## Reset Behavior

The engine exposes:

- `reset_hysteresis()`

and `BaseModel.forward()` currently calls that reset logic across layers at batch start to avoid state bleed between unrelated batches.

That is an important runtime behavior and one of the reasons the hysteresis state should be understood as sequence memory, not as global persistent memory across all training batches.

## Practical Interpretation

The current hysteresis module is best understood as:

- stateful latent memory,
- updated from geometry-aware features plus velocity,
- read out as a small ghost-force correction,
- reset between unrelated batches by the model lifecycle.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- `hyst_decay` from the schema is definitely driving the update in the current engine path,
- the module updates its own weights online from `hyst_update_w` or `hyst_update_b`,
- the ghost force is the raw readout with no fixed scale factor.

Those claims do not match the current runtime.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/components/hysteresis.py`
- `gfn/realizations/gssm/physics/engine.py`
- `gfn/realizations/gssm/models/base.py`

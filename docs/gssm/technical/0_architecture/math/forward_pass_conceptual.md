# Forward Pass - Conceptual Explanation

This document explains the **current forward-pass shape and flow** at a conceptual level.

For the exact runtime behavior, the authoritative sources are:

- `gfn/realizations/gssm/models/base.py`
- `gfn/realizations/gssm/models/manifold_layer.py`
- [09_forward_pass.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/09_forward_pass.md)

## High-Level Picture

The current GSSM forward pass has three conceptual stages:

1. resolve a force sequence,
2. evolve `(x, v)` through manifold layers over time,
3. collect logits through timestep-end readout hooks.

That last point matters: in the current runtime, logits are not produced by a hardcoded direct call inside `BaseModel._evolve_sequence()`. They are usually emitted by hooks attached during model construction.

## Stage 1: Force Resolution

`BaseModel.forward()` currently accepts either:

- `input_ids`
- or `force_manual`

If `force_manual` is provided, it is used directly.

Otherwise:

```python
all_forces = self.embedding(input_ids)
```

Then the model builds:

- `batch_size`
- `seq_len`
- `mask`

and gives hooks a chance to modify the force path through:

- `on_resolve_forces`

So the most faithful conceptual statement is:

- the forward pass starts by creating or receiving a force sequence,
- then optionally lets hooks transform that force sequence before evolution starts.

## Stage 2: State Initialization

The runtime keeps a latent state:

- `x`
- `v`

If a prior state is passed in, that state is reused.

Otherwise the model:

- expands learned `x0` and `v0`,
- optionally adds noise to `x` using `initial_spread`,
- leaves `v` as the expanded learned initial velocity unless a hook overrides state initialization.

Important current detail:

- only `x` gets the default random perturbation in the fallback initialization path shown in `BaseModel.forward()`.

## Stage 3: Sequence Evolution

The main evolution logic lives in `BaseModel._evolve_sequence()`.

Conceptually, for each timestep:

1. extract the current force from the full force sequence,
2. apply timestep-start hooks,
3. pass the current state through every manifold layer,
4. trigger timestep-end hooks,
5. collect any tensor outputs from those hooks as logits,
6. optionally store full `x` and `v` trajectories.

So the core forward path is:

```text
forces -> timestep loop -> layer loop -> timestep-end readout hooks -> logits
```

## What A Manifold Layer Does

Inside `ManifoldLayer.forward()`, the current conceptual flow is:

1. reshape to `[B_eff, H, D_h]`,
2. let pre-integrate plugins adjust state or `dt`,
3. call the integrator,
4. let post-integrate plugins modify the stepped state,
5. mix heads,
6. apply dynamics routing,
7. wrap topology on the new position,
8. run finalize hooks such as fractal refinement,
9. reshape back to the original tensor layout.

This is why the more accurate conceptual layer formula is:

```text
Layer = reshape + pre_integrate_plugins + integrator + post_integrate_plugins
        + mixer + dynamics + topology_wrap + finalize_plugins
```

not just a generic "integrator then readout" story.

## Force Shape Handling

`ManifoldLayer` supports multiple force layouts depending on the input state shape and geometry scope.

Conceptually:

- 4D state input gets flattened to effective batch form,
- 2D or 3D force inputs may be broadcast or partitioned,
- `geometry_scope` affects whether force is shared globally or partitioned per head.

So the docs should not pretend there is only one universal force shape during the whole forward path.

## Readout And Logits

In the current runtime, logits are usually produced by hook callbacks triggered on:

- `on_timestep_end`

Any hook that returns a tensor there contributes to `l_logits`.

That means the conceptual story is:

- the forward pass evolves state continuously through layers,
- then readout is attached as a hook-driven observation mechanism at timestep boundaries.

This is more faithful than describing readout as an unconditional direct method call inside the evolution loop.

## Return Value

`BaseModel.forward()` currently returns:

```python
(res_logits, (x_final, v_final), state_info)
```

where `state_info` contains at least:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

Important current caveat:

- `x_seq` and `v_seq` only contain full trajectories when `store_full_sequence=True`,
- otherwise they are reduced to final-state-shaped placeholders for consistency.

## Hook-Wrapped Evolution

The evolution function itself can be replaced or wrapped through:

- `wrap_evolution`

This is how optional features such as checkpointing or adjoint-style evolution enter the forward path.

So conceptually, the forward pass is not strictly a single immutable loop; it is a hook-wrappable evolution skeleton.

## Practical Summary

The current forward pass is best understood as:

- force resolution,
- state initialization,
- hook-wrappable timestep evolution through manifold layers,
- hook-driven readout collection,
- final state plus trajectory metadata assembly.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- logits always come from a direct hardcoded readout call in the main loop,
- force shape is always a single fixed layout,
- the forward pass always stores the full trajectory,
- the evolution loop cannot be wrapped by plugins.

Those claims do not match the current runtime.

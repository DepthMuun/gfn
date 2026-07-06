# ISN Usage Guide

This guide covers the practical aspects of using the ISN implementation in the `gfn` framework.

## Available Components

The `gfn` framework provides several specialized components for building an ISN:

### Scanners (Boundary)
- `GFNScanner`: Default geometric projector.
- `SSMScanner`: State-space based initialization.
- `TransformerScanner`: Uses a shallow attention block for impulse preparation.
- `LinearScanner`: Minimal footprint projection.

### World Engines (Physics)
- `GFNPhysics`: Default persistent latent world with configurable `euler`, `leapfrog`, or `yoshida` integrators.
- `TopologicalPhysics`: Alternative topological world backend.
- `ParallelPhysics`: Alternative scan/parallel world backend.

### Emitters (Outcome)
- `GFNEmitter`: Standard projection to logit space.
- `ThresholdEmitter`: Uses energy-based gating for sparse emissions.
- `SSMEmitter`: High-resolution state materialization.

## High-Level Model Creation

```python
from gfn import isn

model = isn.create(
    vocab_size=50000,
    d_model=256,
    scanner="gfn",
    world="gfn",
    emitter="gfn",
)
```

`import gfn` and `gfn.create("isn", ...)` is also supported, but `from gfn import isn` is the shorter public shortcut.

## Advanced Assembly

Most users should stay with `isn.create(...)`. Manual component wiring through `gfn.realizations.isn...` is possible for advanced extension work, but it is a lower-level path and not the primary public usage pattern.

## Inference & Generation

ISN supports two primary forward modes: **Sequence Mode** and **Stateful Mode**.

### Sequence Mode (Parallel)
Used for training or processing a fixed prompt.
```python
results = model(input_ids)  # input_ids: [batch, seq_len]
logits = results["logits"]
final_state = results["final_state"]
```

### Stateful Mode (Autoregressive)
Used for generating text token-by-token. The `generate` method handles the state persistence for you.
```python
# Generate 50 new tokens
generated, info = model.generate(
    input_ids=prompt_ids,
    max_length=50,
    temperature=0.7,
    noise_std=0.01 # Add "Thermal Noise" for variety
)

# Access the final world state
final_state = info["final_state"]
final_scanner_state = info["final_scanner_state"]
```

## Handling Persistent State

One of the main advantages of ISN is the ability to maintain context over extremely long periods by simply saving the `world_state`.

```python
# Part 1: Initial context
res1 = model(prompt_part_1, return_world_state=True)
state_v1 = res1["final_state"]

# Part 2: Continue from previous state (O(1) cost)
res2 = model(prompt_part_2, world_state=state_v1)
```

If you need scanner continuity as well, pass `scanner_state=res1["final_scanner_state"]` into the next call.

## Configuration Parameters

- `noise_std`: Adds noise during world evolution.
- `max_burst`: Forward argument passed into the world step budget.
- `temperature`: Sampling temperature used by `generate()`.
- `world_state`: Optional latent state to continue a previous run.
- `scanner_state`: Optional scanner-side state to continue incremental processing.
- `world_kwargs={"integrator": ...}`: Selects `euler`, `leapfrog`, or `yoshida` when using the default `GFNPhysics` backend.

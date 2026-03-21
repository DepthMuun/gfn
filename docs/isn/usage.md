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
- `GFNWorld`: Standard second-order flow with Symplectic integration.
- `TopologicalWorld`: Optimized for tasks involving strict manifold constraints (e.g., Toroidal XOR).

### Emitters (Outcome)
- `GFNEmitter`: Standard projection to logit space.
- `ThresholdEmitter`: Uses energy-based gating for sparse emissions.
- `SSMEmitter`: High-resolution state materialization.

## Model Creation Example

```python
from gfn.realizations.isn import Model
from gfn.realizations.isn.components.scanners import GFNScanner
from gfn.realizations.isn.components.worlds import GFNWorld
from gfn.realizations.isn.components.emitters import GFNEmitter

# Orchestrate a specific architecture
model = Model(
    scanner=GFNScanner(vocab_size=50000, d_model=256),
    world=GFNWorld(d_model=256),
    emitter=GFNEmitter(d_model=256, vocab_size=50000)
)
```

## Inference & Generation

ISN supports two primary forward modes: **Sequence Mode** and **Stateful Mode**.

### Sequence Mode (Parallel)
Used for training or processing a fixed prompt.
```python
results = model(input_ids) # input_ids: [batch, seq_len]
logits = results['logits']
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
final_state = info['final_state']
```

## Handling Persistent State

One of the main advantages of ISN is the ability to maintain context over extremely long periods by simply saving the `world_state`.

```python
# Part 1: Initial context
res1 = model(prompt_part_1, return_state=True)
state_v1 = res1['final_state']

# Part 2: Continue from previous state (O(1) cost)
res2 = model(prompt_part_2, world_state=state_v1)
```

## Configuration Parameters

- `noise_std`: The amount of "stochastic force" applied during integration. Higher values increase variety but can break structural coherence.
- `max_burst`: Controls the integration step density. Higher values provide more precise physics at the cost of speed.
- `temperature`: Standard softmax temperature for the Emitter's output.

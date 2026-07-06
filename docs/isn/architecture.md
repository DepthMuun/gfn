# ISN Architecture: Persistent Latent World

The **Inertial State Network (ISN)** treats sequence processing as the evolution of a persistent latent world instead of storing context in an external attention cache. The current implementation is a **modular pipeline** assembled from registered components.

## Runtime Structure

At runtime, an ISN model is composed of three injected modules:

1. **Scanner**: maps token IDs to continuous impulses.
2. **World Engine**: updates the persistent latent state using those impulses.
3. **Emitter**: projects emitted world embeddings back to token logits when the world engine does not already provide logits directly.

Conceptually, the forward path is:

```text
token ids -> scanner -> impulses -> world engine -> emitted embeddings -> emitter -> logits
```

This matches the actual `Model.forward()` orchestration used by the implementation.

## Registries And Assembly

ISN is assembled through registries rather than a hard-coded monolith:

- `scanners`
- `physics`
- `emitters`
- `strategies`

The public factory resolves component names from those registries and instantiates the model:

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

This is the canonical public entry point for most users.

## World Engine

The default ISN world engine is `GFNPhysics`, which maintains a continuous latent state and updates it from scanner impulses. In `v2.7.3`, the integrator became a swappable component while preserving backward compatibility.

Supported integrators in the default world engine are:

- `euler`
- `leapfrog`
- `yoshida`

`euler` remains the default for backward compatibility. The symplectic variants (`leapfrog` and `yoshida`) additionally maintain velocity internally and expose `final_velocity` in the world output.

Example:

```python
from gfn import isn

model = isn.create(
    vocab_size=50000,
    d_model=256,
    world="gfn",
    world_kwargs={"integrator": "leapfrog"},
)
```

ISN also exposes alternative registered world engines such as `topological` and `parallel`, but they are different physics backends, not aliases for the default `GFNPhysics` integrator stack.

## Stateful Operation

Persistence comes from carrying the latent state across calls. `Model.forward()` accepts an optional `world_state`, and `Model.generate()` keeps both `world_state` and `scanner_state` internally while generating autoregressively.

That means ISN can process long contexts by advancing the latent state rather than re-materializing a full attention history.

## Output Contract

The model returns a dictionary centered on:

- `logits`
- `energy_trace`
- `world_coherence`
- `emitted_embeddings`
- `final_state`
- `final_scanner_state`

If a world engine already produces logits directly, the model uses them. Otherwise, logits are produced by the emitter from `emitted_embeddings`.

## Complexity Notes

ISN is designed around a persistent state whose size does not scale with sequence length in the same way as a quadratic attention matrix. In practice:

- autoregressive generation advances the model one token at a time while reusing latent state,
- the stored world state is constant-size with respect to the processed sequence length,
- training-time memory depends on the chosen backpropagation strategy.

When the `adjoint` training strategy is used, the world evolution can be differentiated with reduced memory usage, subject to the constraints of that strategy and its dependencies.

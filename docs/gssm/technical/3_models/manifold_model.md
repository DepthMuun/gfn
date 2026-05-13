# Manifold Model Architecture

The `gfn/models` module implements the neural architecture that lives on the manifold. Unlike traditional Transformers, GFN models are stateful evolution engines.

## 1. Manifold Model (`manifold.py`)
The top-level orchestrator. It manages the sequence of evolution steps and the persistence of the $(x, v)$ state.

### Autoregressive State Persistence
GFN models can persist their physical state across forward calls. This allows "infinite context" evolution where the last $(x, v)$ of Batch $N$ is the starting point for Batch $N+1$.

---

## 2. Manifold Layer (`manifold_layer.py`)
The fundamental unit of GFN. It represents a single "step" in the manifold's time $dt$.

### Components of a Layer:
1. **Mixer**: Exchanges information between particles (heads/features).
2. **Integrator**: Computes the motion step.
3. **Dynamics**: Routes the mixer proposal into the integrator.
4. **Gating**: Dynamically scales the layer's $dt$.

---

## 3. Mixers (`gfn/models/components/mixer.py`)
Mixers are the "Attention" of GFN. They define how particles interact.

- **FlowMixer**: A high-speed geometric mixing layer. Uses low-rank projections to exchange momentum between heads.
- **GeodesicAttentionMixer**: Combines standard Multi-Head Attention with manifold constraints. Attention scores are modulated by the geodesic distance between particles.

---

## 4. Readouts & Embeddings (`gfn/models/components/`)
Interface between the discrete vocab and the continuous manifold.

### Functional Embedding
Maps token IDs to "Impulses" (Forces).
- When a token occurs, it applies a force $F$ to the particle, changing its velocity $v$.
- **Holographic Mode**: Embeddings are interference patterns in the manifold.

### Categorical Readout
A projection layer that converts the final manifold state into logits for classification.
- **Toroidal Readout**: Projects angulares $x$ into $[\sin(x), \cos(x)]$ before the linear layer.

---

## 5. Plugin System & Hooks (`hooks.py`)
GFN uses a powerful Hook system to inject logic without modifying the core loop.
- **Pre-forward**: Setup inputs.
- **On-layer-end**: Extract latent states (e.g., for Jacobi tracking).
- **On-timestep-end**: Apply readouts.

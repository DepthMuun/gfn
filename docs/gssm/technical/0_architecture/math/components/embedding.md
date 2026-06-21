# Embeddings (FunctionalEmbedding)

## What is an Embedding?

In GSSM, the embedding component is better understood as a **force encoder**.

It maps model inputs into continuous vectors that are later interpreted as external forces acting on the manifold state.

Most common runtime path:

- input: token indices `[B, T]`
- output: force sequence `[B, T, D]`

Conceptually:

- each input contributes an impulse-like force,
- that force perturbs the latent dynamics through the physics engine.

---

## The Mapping Problem

For token-based paths, inputs are discrete:

$$t \in \{0, 1, 2, ..., V-1\}$$

but the physics engine consumes continuous force vectors:

$$F \in \mathbb{R}^{D}$$

So the embedding learns a mapping of the form:

$$F = f(t)$$

or, for continuous mode:

$$F = f(x_{continuous})$$

---

## Embedding Modes

### 1. Lookup Mode

**What it does**: Direct lookup table, implemented with `nn.Embedding`.

**Table**: $W_{embed} \in \mathbb{R}^{V \times D}$

**Operation**: $F = W_{embed}[t]$

Simply "look up" the row corresponding to token $t$.

**Runtime note**: This is a true discrete-token path and works naturally with the current `BaseModel.forward(input_ids=...)`.

---

### 2. Linear Mode (Bit Expansion)

**What it does**: Expand token IDs into binary coordinates and then apply a learned linear projection.

**Step 1: Bit extraction**
$$\text{bits}_i(t) = (t \, \& \, 2^i) > 0 \quad \text{for } i = 0, ..., D_c-1$$

Extract $D_c$ bits from the token ID.

**Step 2: Projection**
$$F = W_{proj} \cdot \text{bits}(t) \cdot s_{impulse}$$

Where:

- $W_{proj}$ is the learned output projection,
- $s_{impulse}$ is a learnable impulse scale.

**Why this mode exists**

- does not require a full lookup table,
- works naturally with integer IDs,
- is the current schema-backed default runtime mode.

**Current default**:

- `physics.embedding.mode = "linear"`

---

### 3. Binary Mode

This is similar to linear mode, but the bits are remapped to $\{-1, +1\}$ instead of $\{0, 1\}$.

$$\text{binary}_i(t) = 2 \cdot \text{bits}_i(t) - 1$$

**Benefit**:

- centered representation,
- often easier to optimize than pure nonnegative bit features.

---

### 4. SIREN / Implicit Mode

**What it does**: Use sinusoidal hidden layers to map token-derived coordinates into force space.

**SIREN layers**:
$$h_{k+1} = \sin(\omega_0 \cdot (W_k h_k + b_k))$$

**Process**:

1. convert token ID into coordinate features,
2. pass them through `SineLayer` blocks,
3. project to the final force dimension.

**Runtime note**:

- `omega_0` is now actually wired into the builder and the `SineLayer` path,
- this is no longer a dead configuration knob.

---

### 5. Continuous Mode

**What it does**: Bypass token lookup and directly project continuous input `[B, T, D_in]` into force space.

**Input**: $x_{continuous} \in \mathbb{R}^{B \times T \times D_{in}}$

Examples:
- Image patches: $D_{in}$ = pixels per patch
- Audio frames: $D_{in}$ = samples per frame
- Any vector sequence

**Operation**:
$$F = MLP(x_{continuous}) \cdot s_{impulse}$$

Where $MLP$: $\mathbb{R}^{D_{in}} \to \mathbb{R}^{D}$

**Why it exists**

- native multimodal support,
- no vocabulary lookup,
- direct force injection from vector-valued inputs.

### Important Runtime Caveat

The embedding module supports `mode="continuous"` at the component level, but the current main forward path in `BaseModel` still resolves forces like this:

```python
if force_manual is not None:
    all_forces = force_manual
elif input_ids is not None:
    all_forces = self.embedding(input_ids)
```

That means `continuous` is **not** automatically a drop-in replacement for every existing token-based script.

Today there are two safe ways to use continuous-style inputs:

1. build a compatible call path that explicitly passes `continuous_input` to `FunctionalEmbedding`,
2. precompute forces externally and pass them through `force_manual`.

So the mathematical description of continuous embedding is valid, but user-facing workflows must still be checked against the actual runtime path.

---

## The Impulse Scale

All modes multiply the final force by a learnable scale:

$$F_{final} = F \cdot s_{impulse}$$

**Purpose**:

- controls force magnitude,
- gives the model a trainable global force gain,
- belongs to the set of physics-sensitive parameters often tuned separately.

**Initialization**: $s_{impulse} \approx 1.0$

---

## Force Interpretation

The resulting force is used as an external contribution in the dynamics:

$$\frac{dv}{dt} = a_{net} + F_{embed}(t)$$

So each input effectively pushes the latent dynamics in a particular direction.

### Physical Analogy

Think of tokens as:
- **Wind**: $F$ is wind direction and strength
- **Gravity**: $F$ pulls toward token's meaning
- **Impulse**: $F$ is an instantaneous kick to velocity

---

## Comparison of Modes

| Mode | Input Type | Main Mechanism | Current Runtime Note |
|------|------------|----------------|----------------------|
| `lookup` | discrete ids | `nn.Embedding` lookup | naturally supported by `input_ids` path |
| `linear` | discrete ids | bit expansion + linear projection | current default |
| `binary` | discrete ids | symmetric bit expansion + linear projection | centered bit features |
| `siren` | discrete ids | sinusoidal implicit network | uses `omega_0` |
| `continuous` | continuous vectors | MLP projection | requires compatible nonstandard call path |

---

## Mathematical Formulation

For discrete-token modes:

$$F: \{0, ..., V-1\} \to \mathbb{R}^{D}$$

For continuous mode:

$$F: \mathbb{R}^{D_{in}} \to \mathbb{R}^{D}$$

Examples:

**Lookup**

$$F(t) = W[t] \cdot s$$

**Bit-based**

$$F(t) = W_{proj} \cdot \phi_{bits}(t) \cdot s$$

**SIREN**

$$F(t) = W_{out} \cdot \text{SIREN}(\phi_{coord}(t)) \cdot s$$

**Continuous**

$$F(x) = MLP(x) \cdot s$$

---

## Practical Selection

Use `lookup` when:

- you want a standard discrete embedding table.

Use `linear` when:

- you want the current default path,
- you want token IDs mapped through bit features rather than a lookup table.

Use `binary` when:

- you want a similar path to `linear` but centered around zero.

Use `siren` when:

- you want a more expressive implicit embedding field,
- oscillatory structure in the input representation matters.

Use `continuous` when:

- your data is already vector-valued,
- you are willing to use a compatible custom path instead of assuming every token-based training script will just work unchanged.

---

## Runtime Cross-References

- `gfn/realizations/gssm/models/components/embedding.py`
- `gfn/realizations/gssm/models/builders/embedding_builder.py`
- `gfn/realizations/gssm/models/base.py`
- `docs/gssm/technical/runtime/01-hyperparameters.md`

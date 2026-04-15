# Embeddings (FunctionalEmbedding)

## What is an Embedding?

An embedding transforms discrete tokens (like words or token IDs) into continuous force vectors that can drive the manifold dynamics.

**Input**: Token indices (integers)  
**Output**: Force vectors (continuous)  

Think of it as: "Each token pushes the manifold in a specific direction."

---

## The Mapping Problem

Tokens are discrete: $t \in \{0, 1, 2, ..., V-1\}$ where $V$ = vocabulary size.

Forces must be continuous: $F \in \mathbb{R}^D$ where $D$ = model dimension.

The embedding learns: $F = f(t)$

---

## Embedding Modes

### 1. Lookup Mode (Standard)

**What it does**: Direct lookup table - each token has a pre-learned vector.

**Table**: $W_{embed} \in \mathbb{R}^{V \times D}$

**Operation**: $F = W_{embed}[t]$

Simply "look up" the row corresponding to token $t$.

**Use case**: Standard NLP, when vocabulary is moderate.

---

### 2. Linear Mode (Bit Expansion)

**What it does**: Expand token ID into bits, then project.

**Step 1: Bit extraction**
$$\text{bits}_i(t) = (t \, \& \, 2^i) > 0 \quad \text{for } i = 0, ..., D_c-1$$

Extract $D_c$ bits from the token ID.

**Step 2: Projection**
$$F = W_{proj} \cdot \text{bits}(t) \cdot s_{impulse}$$

Where:
- $W_{proj} \in \mathbb{R}^{D \times D_c}$ learns to combine bits
- $s_{impulse}$ = learnable impulse scale

**Why bits?**
- Generalizes to any token ID (even unseen ones)
- No fixed vocabulary limit
- Smooth interpolation between nearby IDs

**Use case**: Large vocabularies, continuous token spaces.

---

### 3. Binary Mode

**Similar to Linear**, but bits are mapped to $\{-1, +1\}$ instead of $\{0, 1\}$.

$$\text{binary}_i(t) = 2 \cdot \text{bits}_i(t) - 1$$

**Benefit**: Symmetric around zero, better for gradient flow.

---

### 4. SIREN Mode (Neural Field)

**What it does**: Use a Sinusoidal Representation Network to encode coordinates.

**SIREN layers**:
$$h_{k+1} = \sin(\omega_0 \cdot (W_k h_k + b_k))$$

**Process**:
1. Convert token ID to coordinate space
2. Pass through SIREN network
3. Project to force dimension

**Why SIREN?**
- Excellent for representing high-frequency details
- Smooth derivatives (good for physics)
- Implicit neural representation

**Use case**: Complex patterns, when token structure matters.

---

### 5. Continuous Mode (Multimodal)

**What it does**: Bypass token lookup entirely - directly project continuous input.

**Input**: $x_{continuous} \in \mathbb{R}^{B \times T \times D_{in}}$

Examples:
- Image patches: $D_{in}$ = pixels per patch
- Audio frames: $D_{in}$ = samples per frame
- Any vector sequence

**Operation**:
$$F = MLP(x_{continuous}) \cdot s_{impulse}$$

Where $MLP$: $\mathbb{R}^{D_{in}} \to \mathbb{R}^{D}$

**Why continuous?**
- Native multimodal support
- No vocabulary limitation
- Direct force injection

**Use case**: Images, audio, video, any continuous data.

---

## The Impulse Scale

All modes apply a learnable scaling factor:

$$F_{final} = F \cdot s_{impulse}$$

**Purpose**:
- Controls force magnitude
- Learnable adaptation
- Prevents force explosion

**Initialization**: $s_{impulse} \approx 1.0$

---

## Force Interpretation

The output force drives the manifold:

$$\frac{dv}{dt} = a_{net} + F_{embed}(t)$$

Each token "pushes" the velocity in its embedding direction.

### Physical Analogy

Think of tokens as:
- **Wind**: $F$ is wind direction and strength
- **Gravity**: $F$ pulls toward token's meaning
- **Impulse**: $F$ is an instantaneous kick to velocity

---

## Comparison of Modes

| Mode | Lookup Table | Generalization | Use Case |
|------|--------------|--------------|----------|
| Lookup | Yes | Poor (fixed vocab) | Standard NLP |
| Linear | No | Good (any integer) | Large vocab |
| Binary | No | Good (symmetric) | Balanced features |
| SIREN | No | Excellent (smooth) | Complex patterns |
| Continuous | N/A | Native (any vector) | Multimodal |

---

## Mathematical Formulation

### General Form

$$F: \{0, ..., V-1\} \to \mathbb{R}^D$$

**Lookup**:
$$F(t) = W[t] \cdot s$$

**Bit-based**:
$$F(t) = W_{proj} \cdot \phi_{bits}(t) \cdot s$$

**SIREN**:
$$F(t) = W_{out} \cdot \text{SIREN}(\phi_{coord}(t)) \cdot s$$

**Continuous**:
$$F(x) = MLP(x) \cdot s$$

---

## When to Use Each Mode

**Lookup**: Default for text (vocab < 100k)

**Linear/Binary**: Very large vocabularies, byte-level modeling

**SIREN**: When tokens have structure/ordering that matters

**Continuous**: Images, audio, or any non-discrete input

---

*File: technical/0_architecture/math/components/embedding.md*
*Last Updated: 2026-04-02*

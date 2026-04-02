# Readout

## What is the Readout?

The Readout transforms the final manifold state back into predictions. It is the inverse of the embedding - while embedding maps tokens → forces, readout maps manifold state → logits.

**Input**: Manifold position $x \in \mathbb{R}^{B \times H \times D_h}$  
**Output**: Logits over vocabulary $logits \in \mathbb{R}^{B \times V}$

Think of it as: "Given where the manifold ended up, what token should come next?"

---

## The Mapping Challenge

The manifold state lives on a curved space:
- Torus: Periodic, bounded $[-\pi, \pi]$
- Euclidean: Unbounded

The readout must project this to a flat vocabulary space.

---

## Readout Types

### 1. Categorical Readout (Standard)

**What it does**: Projects manifold state to vocabulary logits for classification.

**For Euclidean topology**:
$$x_{flat} = \text{flatten}(x) \in \mathbb{R}^{B \times D}$$
$$\text{logits} = W_{readout} \cdot x_{flat} + b_{readout}$$

Where $W_{readout} \in \mathbb{R}^{V \times D}$

**For Torus topology**:
$$x_{enc} = [\sin(x); \cos(x)] \in \mathbb{R}^{B \times 2D}$$
$$\text{logits} = W_{readout} \cdot x_{enc} + b_{readout}$$

**Why $\sin/\cos$ encoding for Torus?**
- Direct linear projection fails on periodic space
- $\sin/\cos$ capture angular position uniquely
- Preserves periodic structure

---

### 2. Identity Readout (Holographic)

**What it does**: Returns manifold state directly without projection.

$$\text{output} = \text{flatten}(x) \in \mathbb{R}^{B \times D}$$

**Use case**: 
- Latent space models
- When output dimension equals manifold dimension
- Regression tasks (not classification)

**Warning**: Cannot use with cross-entropy loss unless $D = V$.

---

### 3. Implicit Readout (MLP)

**What it does**: Uses a neural network for flexible mapping.

**Architecture**:
$$h_1 = \text{GELU}(W_1 \cdot x_{enc} + b_1)$$
$$\text{output} = W_2 \cdot h_1 + b_2$$

**For Torus**: Uses $[\sin(x); \cos(x)]$ as input (like Categorical)

**Use case**:
- Complex readout functions
- Regression with non-linear mapping
- Latent space alignment

---

## The Projection Process

### Step 1: Flatten

Convert multi-head state to vector:
$$x \in \mathbb{R}^{B \times H \times D_h} \to x_{flat} \in \mathbb{R}^{B \times D}$$

### Step 2: Encode (for Torus)

If topology is torus, apply trigonometric encoding:
$$x_{enc} = \begin{bmatrix} \sin(x_{flat}) \\ \cos(x_{flat}) \end{bmatrix} \in \mathbb{R}^{B \times 2D}$$

This doubles the dimension but preserves angular information.

### Step 3: Linear Projection

Map to vocabulary space:
$$\text{logits} = W \cdot x_{(\text{flat}|\text{enc})} + b$$

Where $W \in \mathbb{R}^{V \times D_{in}}$:
- Euclidean: $D_{in} = D$
- Torus: $D_{in} = 2D$

### Step 4: Softmax (during inference)

$$P(token_i) = \frac{\exp(\text{logits}_i)}{\sum_j \exp(\text{logits}_j)}$$

---

## Topology-Aware Encoding

### Why Trigonometric Functions?

On a torus, position wraps around. Direct linear projection has discontinuities at the wrap boundary.

**Example**:
- Position A: $x = \pi - 0.1$
- Position B: $x = -\pi + 0.1$

These are actually close on the torus (wrapping around), but far in raw coordinates.

**Solution**: $\sin/\cos$ encoding
$$\sin(\pi - 0.1) \approx \sin(-\pi + 0.1)$$
$$\cos(\pi - 0.1) \approx \cos(-\pi + 0.1)$$

Nearby points on the manifold have similar $\sin/\cos$ encodings.

---

## Hook Integration

The Readout connects to the model through hooks:

**Registration**:
- Hook: `on_timestep_end`
- Triggered: After each layer completes
- Action: Compute readout(state) → logits

**Timeline**:
```
Token 1: Input → Embed → Layer 1 → ... → Layer L → Readout → Logits_1
Token 2: Input → Embed → Layer 1 → ... → Layer L → Readout → Logits_2
...
Token S: Input → Embed → Layer 1 → ... → Layer L → Readout → Logits_S
```

---

## Comparison of Readouts

| Type | Input | Output | Use Case |
|------|-------|--------|----------|
| Categorical | $x$ or $[\sin(x); \cos(x)]$ | Logits[V] | Classification |
| Identity | $x$ | State[D] | Latent models |
| Implicit | $[\sin(x); \cos(x)]$ | Arbitrary | Complex mapping |

---

## Mathematical Properties

### Information Flow

The readout extracts information from the manifold:
$$\text{logits} = f_{readout}(x_{final})$$

Gradients flow back:
$$\frac{\partial L}{\partial x} = W_{readout}^T \cdot \frac{\partial L}{\partial \text{logits}}$$

This trains the manifold to reach states that predict the correct tokens.

### Capacity

The readout matrix $W \in \mathbb{R}^{V \times D}$ can represent:
- Up to $D$ independent output directions
- Complex mappings through encoding
- With $V$ parameters per dimension

---

## Physical Interpretation

Think of the manifold as a "thought space":
- Each position represents a "concept"
- Trajectory represents "thinking process"
- Readout asks: "Given this mental state, what comes next?"

The $\sin/\cos$ encoding (for Torus) ensures:
- Similar mental states (nearby on torus) make similar predictions
- No discontinuities at concept boundaries
- Smooth transitions in meaning

---

*File: technical/0_architecture/math/components/readout.md*
*Last Updated: 2026-04-02*

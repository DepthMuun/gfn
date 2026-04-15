# Mixer (FlowMixer)

## What is the Mixer?

The Mixer is a component that combines information across multiple heads. After each head evolves independently through the integrator, the Mixer fuses their states to enable information exchange between heads.

Think of it as: "Each head has seen the input from its own perspective; now they share what they learned."

---

## The Problem

Each head in GSSM operates independently during integration:
- Head 1 evolves position $x_1$ and velocity $v_1$
- Head 2 evolves position $x_2$ and velocity $v_2$
- ...and so on

Without mixing, heads never share information. The Mixer solves this by combining head states.

---

## Two Mixing Modes

### 1. Partition Mode (low_rank, attention)

**What it does**: Collapses all heads into a single aggregated state.

**Input**: $x \in \mathbb{R}^{B \times H \times D_h}$ (B=batch, H=heads, D=head_dim)

**Output**: $x_{mixed} \in \mathbb{R}^{B \times D}$ where $D = H \times D_h$

**Process**:

For **Euclidean** topology:
$$x_{flat} = \text{flatten}(x) \in \mathbb{R}^{B \times (H \cdot D_h)}$$
$$x_{mixed} = W_{mix} \cdot x_{flat} + b_{mix}$$

For **Torus** topology:
$$\sin_x = \sin(x), \quad \cos_x = \cos(x)$$
$$v_{scaled} = \tanh(v / 10)$$
$$x_{cat} = [\sin_x; \cos_x; v_{scaled}] \in \mathbb{R}^{B \times 3H D_h}$$
$$x_{mixed} = W_{mix} \cdot x_{cat}$$
$$x_{mixed} = \arctan_2(\sin(x_{mixed}), \cos(x_{mixed}))$$

**Why trigonometric projection for Torus?**
- Direct averaging doesn't work on circular manifolds
- $\sin/\cos$ encoding preserves periodic structure
- Final $\arctan_2$ projects back to valid torus coordinates

---

### 2. Ensemble Mode

**What it does**: Preserves per-head structure but couples them softly.

**Input**: $x \in \mathbb{R}^{B \times H \times D_h}$

**Output**: $x_{coupled} \in \mathbb{R}^{B \times H \times D_h}$ (same shape)

**Process**:

**Step 1: Compute consensus center**

$$w = \text{softmax}(\text{ensemble\_attn}) \in \mathbb{R}^H$$

For Euclidean:
$$x_{center} = \sum_{h=1}^H w_h \cdot x_h \in \mathbb{R}^{B \times 1 \times D_h}$$

For Torus:
$$x_{center} = \arctan_2\left(\sum_h w_h \sin(x_h), \sum_h w_h \cos(x_h)\right)$$

**Step 2: Soft coupling**

Each head moves slightly toward the consensus:

$$\Delta x = x_{center} - x$$

For Torus:
$$\Delta x = \arctan_2(\sin(\Delta x), \cos(\Delta x))$$

$$x_{coupled} = x + 0.1 \cdot \tanh(W_{couple} \cdot \Delta x)$$

**Why 0.1?**
- Small step prevents disruption of individual head trajectories
- $\tanh$ limits maximum change
- Heads maintain identity while influencing each other

---

## Geodesic Attention Mixer

An alternative mixing mechanism using attention weights based on manifold distance.

### Distance-Based Attention

**Query-Key projection**:
$$Q = W_q \cdot x, \quad K = W_k \cdot x$$

**Geodesic distance** (for Torus):
$$d(q, k) = \arctan_2(\sin(q-k), \cos(q-k))^2$$

**Attention weights**:
$$A_{ij} = \text{softmax}\left(-\frac{d(Q_i, K_j)}{\tau}\right)$$

Closer heads (smaller geodesic distance) get higher attention weights.

**Mixing**:
$$x_{mixed} = \sum_j A_{ij} \cdot V_j$$

---

## Why Mixing Matters

### Without Mixer
- Each head is isolated
- No information sharing
- Like having multiple independent models

### With Mixer
- Heads share insights
- Emergent ensemble behavior
- Better representation capacity

### Analogy
Think of heads as experts in a meeting:
- **Partition mode**: Experts vote and produce single consensus decision
- **Ensemble mode**: Experts discuss and adjust their individual opinions

---

## Mathematical Properties

| Property | Partition | Ensemble |
|----------|-----------|----------|
| Output shape | [B, D] (collapsed) | [B, H, D] (preserved) |
| Information flow | Heads → Single state | Heads ↔ Heads |
| Parameters | $W_{mix} \in \mathbb{R}^{D \times D}$ | $W_{couple} \in \mathbb{R}^{D_h \times D_h}$, attn ∈ ℝ^H |
| Use case | Standard processing | Multi-trajectory preservation |

---

## When to Use Each Mode

**Use Partition (default)**:
- Standard sequence modeling
- Classification tasks
- Most language modeling

**Use Ensemble**:
- Need to preserve per-head trajectories
- Ensemble methods
- Uncertainty quantification per head

---

*File: technical/0_architecture/math/components/mixer.md*
*Last Updated: 2026-04-02*

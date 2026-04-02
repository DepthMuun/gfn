# Forward Pass - Conceptual Explanation

## What is the Forward Pass?

The forward pass is the process of transforming input tokens through the GSSM model to produce output logits. It represents the complete computational flow from input to prediction.

---

## Overview

The forward pass consists of three main phases:

1. **Force Generation**: Convert tokens to manifold forces
2. **State Evolution**: Evolve (position, velocity) through the manifold
3. **Output Production**: Project final state to vocabulary logits

---

## Phase 1: Force Generation

### Input
Token indices: $t \in \{0, 1, ..., V-1\}^{B \times S}$

Where:
- $B$ = batch size
- $S$ = sequence length  
- $V$ = vocabulary size

### Embedding
Each token is mapped to a force vector:

$$F_i = \text{Embedding}(t_i) \in \mathbb{R}^D$$

Where $D$ = model dimension (heads × head_dim).

### Force Tensor
Result: $F \in \mathbb{R}^{B \times S \times D}$

Each position in the sequence has an associated force that will drive the manifold dynamics.

---

## Phase 2: State Evolution

### Initial State
The manifold state consists of position $x$ and velocity $v$:

$$x_0 \in \mathbb{R}^{B \times H \times D_h}$$
$$v_0 \in \mathbb{R}^{B \times H \times D_h}$$

Where:
- $H$ = number of heads
- $D_h$ = head dimension
- $H \times D_h = D$ (total dimension)

Initialized with Gaussian noise scaled by `initial_spread`.

### Evolution Loop

For each timestep $s = 1, ..., S$:

#### Extract Current Force
$$f_s = F_{:,s} \in \mathbb{R}^{B \times D}$$

#### Layer Processing
For each layer $\ell = 1, ..., L$:

**Step 1: Pre-processing**
- Apply plugins (e.g., dynamic time adjustment)
- Adjust timestep $dt$ if needed

**Step 2: Integration**
Compute next state using symplectic integrator:

$$(x', v') = \text{Integrator}(x, v, f_s, dt)$$

The integrator solves:
$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = a(x, v, f_s)$$

Where acceleration $a$ comes from the PhysicsEngine.

**Step 3: Mixing**
Combine information across heads:

$$(x_{mixed}, v_{mixed}) = \text{Mixer}(x', v')$$

**Step 4: Dynamics Routing**
Apply state update rule:

$$x_{new} = \text{Dynamics}(x, x_{mixed})$$

Options:
- Direct: $x_{new} = x_{mixed}$
- Residual: $x_{new} = x + \sigma(s)(x_{mixed} - x)$
- Gated: $x_{new} = g \cdot x_{mixed} + (1-g) \cdot x$

**Step 5: Topology Resolution**
Wrap position to manifold bounds:

$$x_{wrapped} = \text{Wrap}(x_{new})$$

For torus: $x \to \arctan_2(\sin(x), \cos(x))$

**Step 6: Post-processing**
- Apply plugins (e.g., fractal refinement)
- Final adjustments

#### Readout
After all layers, compute logits:

$$\text{logits}_s = \text{Readout}(x_{final}) \in \mathbb{R}^{B \times V}$$

---

## Phase 3: Output Assembly

### Sequence of Logits
Collect logits from all timesteps:

$$\text{Logits} = [\text{logits}_1, \text{logits}_2, ..., \text{logits}_S] \in \mathbb{R}^{B \times S \times V}$$

### Final State
Return final manifold state for potential continuation:

$$\text{state}_{final} = (x_S, v_S) \in \mathbb{R}^{B \times H \times D_h} \times \mathbb{R}^{B \times H \times D_h}$$

### State Information
Additional information returned:
- Full trajectory: $x_{seq} \in \mathbb{R}^{B \times S \times H \times D_h}$
- Velocities: $v_{seq} \in \mathbb{R}^{B \times S \times H \times D_h}$
- Forces: $F \in \mathbb{R}^{B \times S \times D}$

---

## Key Operations

### 1. Reshape Operations

**Sequence Mode**: Flatten batch and sequence dimensions
$$[B, S, H, D] \to [B \cdot S, H, D]$$

**Head Mode**: Flatten all spatial dimensions
$$[B, H, D] \to [B, H \cdot D]$$

### 2. Force Broadcasting

Global force scope: Broadcast to all heads
$$[B, D] \to [B, H, D]$$

Local force scope: Partition across heads
$$[B, D] \to [B, H, D/H]$$

### 3. Topology Wrapping

**Torus** (periodic):
$$x \to \arctan_2(\sin(x), \cos(x)) \in [-\pi, \pi]$$

**Euclidean** (unbounded):
$$x \to x$$

---

## Mathematical Summary

### Complete Forward Pass

$$\text{Logits} = \text{Readout} \circ \underbrace{\text{Layer}_L \circ ... \circ \text{Layer}_1}_{\text{Depth}} \circ \text{Embedding}(t)$$

Where each layer:
$$\text{Layer} = \text{Dynamics} \circ \text{Mixer} \circ \text{Integrator} \circ \text{Plugins}$$

And the integrator solves Hamiltonian dynamics:
$$\dot{x} = v$$
$$\dot{v} = -\Gamma(x,v) + F_{ext} - \mu v + F_{aux}$$

---

## Output Format

The forward pass returns a tuple:

```
(logits, (x_final, v_final), {
    'x_seq': x_trajectory,
    'v_seq': v_trajectory,
    'forces': F,
    'x_final': x_final,
    'v_final': v_final,
    ...
})
```

- **logits**: $[B, S, V]$ - predictions for each position
- **state**: $(x, v)$ for autoregressive continuation
- **info**: Full trajectory for analysis/loss computation

---

*File: technical/0_architecture/math/forward_pass_conceptual.md*
*Last Updated: 2026-04-02*

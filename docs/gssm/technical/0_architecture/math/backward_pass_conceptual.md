# Backward Pass - Conceptual Explanation

## What is the Backward Pass?

The backward pass is the process of computing gradients of the loss with respect to all model parameters. It enables learning through gradient descent optimization.

---

## Overview

PyTorch automatically computes gradients using **automatic differentiation (autograd)**. The backward pass follows the reverse of the forward pass, applying the chain rule at each step.

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{state}} \cdot ... \cdot \frac{\partial \text{layer}_1}{\partial \theta}$$

---

## The Chain Rule

### Basic Principle

If $y = f(g(x))$, then:

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial g} \cdot \frac{\partial g}{\partial x}$$

### Extended to Neural Networks

For a sequence of operations $f_L \circ f_{L-1} \circ ... \circ f_1$:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdot ... \cdot \frac{\partial f_1}{\partial x_0}$$

Gradients flow backward from loss to input.

---

## Gradient Flow Through GSSM

### 1. Output Layer (Readout)

**Forward**: $\text{logits} = W_{readout} \cdot \text{encode}(x_{final}) + b_{readout}$

**Gradients**:
$$\frac{\partial L}{\partial W_{readout}} = \frac{\partial L}{\partial \text{logits}} \cdot x_{final}^T$$
$$\frac{\partial L}{\partial x_{final}} = W_{readout}^T \cdot \frac{\partial L}{\partial \text{logits}}$$

Gradients flow from logits back to final manifold state.

---

### 2. State Evolution (Backward Through Time)

For each layer traversed in reverse:

#### Dynamics Routing

**Residual Dynamics**:
$$x_{new} = x + \sigma(s) \cdot (x_{mixed} - x)$$

**Gradient**:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial x_{new}} \cdot (1 - \sigma(s) + \sigma'(s)(x_{mixed}-x))$$

The gradient splits between current state and mixed state.

#### Mixer

**Forward**: Linear transformation across heads
$$x_{mixed} = W_{mix} \cdot x_{stepped} + b_{mix}$$

**Gradients**:
$$\frac{\partial L}{\partial W_{mix}} = \frac{\partial L}{\partial x_{mixed}} \cdot x_{stepped}^T$$
$$\frac{\partial L}{\partial x_{stepped}} = W_{mix}^T \cdot \frac{\partial L}{\partial x_{mixed}}$$

#### Integrator

**Forward** (Leapfrog example):
$$v_{half} = v + \frac{\Delta t}{2} \cdot a(x, v)$$
$$x_{new} = x + \Delta t \cdot v_{half}$$

**Gradients** flow through:
1. Position updates (differentiable)
2. Velocity updates (differentiable)
3. Acceleration computation (through PhysicsEngine)

---

### 3. Physics Engine

**Forward**:
$$a = -\Gamma(x, v) + F - \mu v$$

**Gradients**:

#### Through Christoffel Symbols
$$\frac{\partial a}{\partial x} = -\frac{\partial \Gamma}{\partial x}$$

Geometry computes derivatives of connection coefficients.

#### Through Friction
$$\frac{\partial a}{\partial v} = -\mu$$

Constant gradient for velocity damping.

#### Through External Force
$$\frac{\partial a}{\partial F} = 1$$

Direct gradient flow to force (embedding).

---

### 4. Auxiliary Components

#### Hysteresis

**Forward**:
$$F_{ghost} = W \cdot \tanh(b + h)$$
$$h_{new} = (1-\alpha)h + \alpha v$$

**Gradients**:
$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial F_{ghost}} \cdot W \cdot \text{sech}^2(b+h) + \frac{\partial L}{\partial h_{new}} \cdot (1-\alpha)$$

Gradients flow through recurrent state update.

#### Stochasticity

**Forward**:
$$F_{stoch} = \sigma \cdot dt^{-1/2} \cdot \mathcal{N}(0,1)$$

**Gradients**:
- Through random sample: **blocked** (sampling is non-differentiable)
- Through $\sigma$: $\frac{\partial L}{\partial \sigma} = \frac{\partial L}{\partial F_{stoch}} \cdot dt^{-1/2} \cdot \mathcal{N}(0,1)$

Only parameters get gradients, not the random noise itself.

#### Curiosity

**Forward**:
$$F_{curiosity} = \lambda \cdot \frac{d}{\|d\|^2}$$

**Gradients** flow through:
- Strength parameter: $\frac{\partial L}{\partial \lambda}$
- Direction computation: $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial x}$

---

### 5. Initial State and Embedding

#### Initial State (x0, v0)

**Forward**:
$$x = x_0 + \epsilon \cdot \mathcal{N}(0, 1)$$

**Gradients**:
$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x}$$

Gradients flow directly to learnable initial state parameters.

#### Embedding

**Forward**:
$$F = W_{embed}[t]$$

**Gradients**:
$$\frac{\partial L}{\partial W_{embed}[t_i]} = \frac{\partial L}{\partial F_i}$$

Each token's embedding vector receives gradient from its force contribution.

---

## Gradient Accumulation

### Through Multiple Steps

For sequence of $S$ timesteps and $L$ layers:

$$\frac{\partial L}{\partial \theta} = \sum_{s=1}^S \sum_{\ell=1}^L \frac{\partial L_s}{\partial \text{layer}_\ell} \cdot \frac{\partial \text{layer}_\ell}{\partial \theta}$$

Gradients from all positions and layers accumulate.

### Through Time (BPTT)

For autoregressive state:

$$\frac{\partial L}{\partial x_t} = \frac{\partial L}{\partial x_{t+1}} \cdot \frac{\partial x_{t+1}}{\partial x_t}$$

Gradients flow backward through the sequence (similar to RNNs).

---

## Gradient Properties

### Vanishing Gradients

**Symptom**: Gradients become very small ($< 10^{-8}$)

**Causes**:
- Long sequences (many gradient multiplications)
- Small initial spread (near-zero initialization)
- High friction (damps gradients)

**Solutions**:
- Increase `initial_spread` to 0.1-0.5
- Use residual connections (better gradient flow)
- Gradient clipping (prevents extreme values)

### Exploding Gradients

**Symptom**: Gradients become very large ($> 1000$)

**Causes**:
- Large time steps without proper scaling
- Velocity saturation disabled
- Long unrolled sequences

**Solutions**:
- Reduce `dt` to 0.05
- Enable `velocity_saturation`
- Use gradient clipping (max_norm = 1.0)

---

## Trainable Parameters

| Parameter | Gradient Source | Update Rule |
|-----------|-----------------|-------------|
| $W_{embed}$ | Token forces | $\Delta W = -\eta \cdot \frac{\partial L}{\partial F}$ |
| $x_0, v_0$ | State evolution | $\Delta x_0 = -\eta \cdot \frac{\partial L}{\partial x}$ |
| $W_{mixer}$ | Head mixing | $\Delta W = -\eta \cdot \frac{\partial L}{\partial x_{mixed}}$ |
| $\sigma_{residual}$ | Residual dynamics | $\Delta \sigma = -\eta \cdot \frac{\partial L}{\partial x_{new}} \cdot \sigma'(s)(x_{mixed}-x)$ |
| $W_{gate}$ | Gated dynamics | $\Delta W = -\eta \cdot \frac{\partial L}{\partial x_{new}} \cdot g(1-g) \cdot [x; x_{mixed}]$ |
| $\mu$ (friction) | Physics engine | $\Delta \mu = -\eta \cdot \frac{\partial L}{\partial a} \cdot (-v)$ |
| $W_{hysteresis}$ | Ghost force | $\Delta W = -\eta \cdot \frac{\partial L}{\partial F_{ghost}} \cdot \tanh(b+h)$ |
| $\sigma$ (stochastic) | Noise magnitude | $\Delta \sigma = -\eta \cdot \frac{\partial L}{\partial F_{stoch}} \cdot dt^{-1/2} \cdot \mathcal{N}(0,1)$ |

---

## Mathematical Summary

### Complete Gradient Computation

$$\nabla_\theta L = \underbrace{\frac{\partial L}{\partial \text{logits}}}_{\text{Output}} \cdot \underbrace{\frac{\partial \text{logits}}{\partial x_{final}}}_{\text{Readout}} \cdot \prod_{\ell=L}^1 \underbrace{\frac{\partial x_\ell}{\partial x_{\ell-1}}}_{\text{Layer } \ell} \cdot \underbrace{\frac{\partial x_0}{\partial \theta}}_{\text{Initialization}}$$

Where each layer gradient includes:
- Dynamics routing
- Head mixing
- Integration step
- Physics engine computation

---

*File: technical/0_architecture/math/backward_pass_conceptual.md*
*Last Updated: 2026-04-02*

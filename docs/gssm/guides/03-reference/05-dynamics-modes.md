# Dynamics Modes Reference

The Dynamics subsystem in MANIFOLD regulates how state updates (proposals) from the physics engine are integrated into the persistent manifold state.

## Overview

Each `ManifoldLayer` processes head states through a `ManifoldMixer` to produce an **Absolute State Proposal**. The chosen **Dynamics Mode** then determines the specific mathematical operation used to transition from the `current_state` to the next state, utilizing this proposal.

## Available Modes

### 1. Direct Dynamics (`direct`)
The simplest and often fastest mode for logic-intensive tasks.
* **Operation**: `state_next = proposal` 
* **Use Case**: Tasks requiring high-frequency state "flips" (e.g., XOR, Binary Parity).
* **Pros**: Zero "momentum" barrier; state can change instantly.
* **Cons**: Can be unstable if the physics signal is noisy.

### 2. Residual Dynamics (`residual`)
Provides smoother gradients and stable flow.
* **Operation**: `state_next = state_curr + (proposal - state_curr)`
* **Use Case**: Continuous flow optimization, multi-step geometric navigation.
* **Pros**: Mathematically consistent with ODE/SDE solvers; preserves state continuity.
* **Cons**: Slower to react to drastic signal changes than Direct mode.

### 3. Mix Dynamics (`mix`)
A configurable blend between retention and update.
* **Operation**: `state_next = alpha * state_curr + (1 - alpha) * proposal`
* **Use Case**: Deep manifolds where gradient vanishing is a concern.
* **Parameters**: `alpha` (Retention factor, 0.0 to 1.0).

### 4. Gated Dynamics (`gated`)
A learnable, GRU-style gating mechanism.
* **Operation**: `z = sigmoid(Linear([state_curr, proposal])); state_next = (1 - z) * state_curr + z * proposal`
* **Use Case**: Complex sequential dependencies where the model must learn what to remember.
* **Pros**: Maximum flexibility; content-aware state updates.

### 5. Stochastic Dynamics (`stochastic`)
Introduces controlled noise for exploration.
* **Operation**: `state_next = proposal + sigma * epsilon`
* **Use Case**: Global optimization, avoiding local minima.
* **Parameters**: `sigma` (Noise intensity).

## Selection Guide

| Task Type | Recommended Mode | Rationale |
|-----------|------------------|-----------|
| **Logical/Discrete** | `direct` | Immediate state transitions required for parity gates. |
| **Physical/Motion** | `residual` | Smoother trajectories and physically consistent flow. |
| **Long-Sequence** | `gated` | Selective state retention prevents forgetting. |
| **Exploratory** | `stochastic` | Noise facilitates escaping geometric bottlenecks. |

## Technical Implementation

Dynamics are implemented as modular classes inheriting from `BaseDynamics`. They are instantiated via the `get_dynamics` factory and used inside `ManifoldLayer.forward`:

```python
# ManifoldLayer.forward snippet (simplified)
x_proposal, v_proposal = self.mixer(x_heads, v_heads)
x_next = self.dynamics_x(x_input, x_proposal)
v_next = self.dynamics_v(v_input, v_proposal)
```

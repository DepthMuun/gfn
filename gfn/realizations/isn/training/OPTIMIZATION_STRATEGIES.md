# ISN Training Optimization Strategies

## The Problem

ISN (Inertial State Network) has a sequential loop in its forward pass:

```python
for t in range(l):
    v_drift = torch.tanh(self.drift(state))
    f_ext = self.diffusion(impulses[:, t, :])
    state = state + v_drift + f_ext
```

This creates an **O(L)** time complexity in the forward pass, but more critically, **O(L)** memory and compute in the **backward pass** (backpropagation) because PyTorch must store all intermediate states to compute gradients.

With `seq_len=1024`, this becomes prohibitively slow (~85 seconds per batch).

---

## Strategy Overview

| Strategy | Forward | Backward | Memory | Implementation |
|----------|---------|----------|--------|----------------|
| **Full BPTT** (current) | O(L) | O(L) | O(L) | Default |
| **TBPTT** (k1/k2) | O(L) | O(k2) | O(k2) | `TruncatedBPTT` |
| **Final Loss Only** | O(L) | O(1) | O(1) | `FinalLossOnly` |
| **STE** | O(L) | O(1) | O(1) | `StraightThroughEstimator` |

---

## Strategy 1: Truncated Backpropagation Through Time (TBPTT)

### Concept
Instead of backpropagating through the entire sequence of length L, we only backprop through the last `k2` steps. The forward pass remains unchanged (still O(L)), but the backward pass is truncated to `k2 << L`.

### Parameters
- `k1`: How often to reset the hidden state (and start a new truncation window)
- `k2`: How many steps to backpropagate through

### Variants
- **TBPTT (k1, k2)**: k1 steps forward, backprop through k2
- **TBPTT (1, k2)**: Backprop through every step (most common)
- **RTRL** (Real-Time Recurrent Learning): No truncation, O(1) forward, O(L) backward (not recommended)

### Implementation
```python
class TruncatedBPTT:
    def __init__(self, k1=1, k2=64):
        self.k1 = k1
        self.k2 = k2

    def backward(self, loss, model, optimizer):
        loss.backward()
        # Detach hidden states that are too old
        if hasattr(model, 'truncated_state'):
            model.truncated_state.detach_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
```

### Trade-offs
- ✅ Reduces backward pass to O(k2) instead of O(L)
- ✅ Memory becomes O(k2) instead of O(L)
- ❌ Loses long-range gradient information
- ❌ May destabilize training if k2 is too small

---

## Strategy 2: Final Loss Only

### Concept
Instead of computing a loss at every timestep and summing/averaging them, we compute the loss **only at the final timestep**. The forward pass is still O(L) (we need to process all tokens to get the final state), but the backward pass is O(1) because there is only ONE loss value to differentiate.

### Why O(1) Backward?
- We only call `loss.backward()` once (on the final token)
- No gradient accumulation over intermediate steps
- PyTorch only needs to compute gradients for the computation graph leading to that single loss

### Implementation
```python
class FinalLossOnly:
    def compute_loss(self, final_hidden_state, targets):
        # Loss only on the last token prediction
        logits = self.head(final_hidden_state[:, -1, :])
        loss = F.cross_entropy(logits, targets[:, -1])
        return loss
```

### Comparison

**Before (Full BPTT):**
```python
losses = []
for t in range(l):
    loss_t = compute_loss(state_t, target_t)
    losses.append(loss_t)
total_loss = sum(losses) / l
total_loss.backward()  # Backprops through ALL timesteps
```

**After (Final Loss Only):**
```python
# Forward through all timesteps (O(L) time, but keeps all states)
for t in range(l):
    state = step(state, input_t)

# Loss only on final state
loss = compute_loss(state, target_final)
loss.backward()  # Backprops only through the computation graph of final state
```

### Trade-offs
- ✅ O(1) backward pass — massive speedup
- ✅ O(1) memory for gradients
- ✅ Simplifies training loop
- ❌ Loses all intermediate supervision signals
- ❌ Model may struggle to learn long-range dependencies (but ISN has O(1) memory, so this might be acceptable)

---

## Strategy 3: Parallel Scan (Mamba/S4 Approach)

### Concept
Linear State-Space Models can be reformulated as linear recurrences that can be computed in **O(log L)** time using parallel prefix sum algorithms (parallel scan).

The key insight: `state_t = A * state_{t-1} + B * input_t` can be computed for all t in parallel if we use an associative operator.

### Mathematical Reformulation
Standard recurrence:
```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t
```

Parallel scan requires:
1. Defining an associative binary operator
2. Using a parallel scan algorithm (Blelloch's algorithm, etc.)

### Status
**Not yet implemented.** Requires custom CUDA kernels or `torch.compile` with specific flags.

### Trade-offs
- ✅ O(log L) forward pass (theoretically optimal)
- ✅ Fully parallelizable
- ❌ Requires careful mathematical reformulation
- ❌ Complex implementation
- ❌ May not work well with non-linearities (tanh, etc.)

---

## Strategy 4: Straight-Through Estimator (STE)

### Concept
During the backward pass, we approximate the gradient of the sequential operation by bypassing the sequential dependency. The gradient of the state update is approximated as the identity, effectively treating the forward pass as a "pass-through" for gradient computation.

### Why O(1)?
The backward pass ignores the sequential chain rule and directly propagates gradients from the loss to all parameters in a single step.

### Implementation
```python
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, input, drift_fn, diffusion_fn):
        # Normal forward pass
        new_state = state + drift_fn(state) + diffusion_fn(input)
        ctx.save_for_backward(state, input)
        return new_state

    @staticmethod
    def backward(ctx, grad_output):
        state, input = ctx.saved_tensors
        # STE: gradient passes through unchanged
        # Treats forward as identity function for gradient purposes
        grad_state = grad_output  # No chain rule through the recurrence
        grad_input = grad_output
        return grad_state, grad_input, None, None
```

### Trade-offs
- ✅ O(1) backward pass
- ✅ Very fast
- ❌ Gradient is imprecise — model may not converge well
- ❌ Information from intermediate states is lost
- ❌ Experimental — needs validation

---

## Strategy 5: DirectProjection (ULTRA-FAST, O(1) Training)

### Concept
Train a **proxy network** that approximates the ISN behavior in a single feedforward pass.

Instead of the sequential loop:
```python
for t in range(L):
    state = state + f(state) + g(impulse_t)
```

We use:
```python
state = MLP(sum(impulses))  # Single feedforward pass!
```

### Why O(1)?
- No sequential dependency to track for gradients
- Single forward/backward pass through MLP
- No need to store intermediate states
- **Truly O(1) memory and time for training**

### Implementation
```python
from gfn.realizations.isn.training.direct_projection_trainer import train_direct_projection

trainer = train_direct_projection(
    train_loader=train_loader,
    vocab_size=86,
    d_embedding=256,
    d_model=256,
    num_epochs=50
)
```

### Trade-offs
- ✅ **Truly O(1) training** - Massive speedup
- ✅ Can process very long sequences
- ✅ Simple implementation
- ❌ Proxy may not perfectly approximate ISN
- ❌ Need to transfer learning to real model after

---

## Strategy 6: Lazy Gradient (Memory-Optimized)

### Concept
During backward pass, **don't save intermediate states**. Instead, recompute them on-the-fly going backwards.

Memory: O(1) instead of O(L)
Compute: ~2x forward cost (because we recompute during backward)

### Implementation
See `LazyGradientFunction` in `direct_projection.py`

---

## Recommended Implementation Order

1. **DirectProjection (Strategy 5)** — Fastest, try first
2. **Final Loss Only (Strategy 2)** — If Strategy 5 doesn't converge
3. **TBPTT (Strategy 1)** — Good middle ground
4. **STE (Strategy 4)** — Experimental

---

## Configuration

Strategies should be swappable via config:

```json
{
  "training": {
    "backprop_strategy": "final_loss_only",
    "tbptt_k2": 64,
    "ste_enabled": false
  }
}
```

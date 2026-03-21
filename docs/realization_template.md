# Realization Template: [Your Architecture Name]

*Instructions: Copy this file to `docs/[your_architecture]/README.md` and complete all sections.*

## 1. Executive Summary

Brief description of what problem this architecture solves and how it differs from other GFN realizations. Explain the specific mathematical formalism and domain of application.

### Key Characteristics

| Characteristic | Value |
|---------------|-------|
| **Type** | [Differential/Simulative/Hybrid] |
| **State Dimension** | [Fixed/Variable] |
| **Inference Memory** | [O(1)/O(d)/O(N)/other] |
| **Forward Pass** | [O(N) sequential/O(N) parallel] |
| **Training Memory** | [O(1)/O(N)] |

### Paradigm Alignment

Explain how this realization satisfies the GFN core principles:
- **State Persistence**: How does the system maintain state across sequences?
- **Structural Invariants**: What invariants does it preserve?
- **Deterministic Evolution**: What structured dynamics govern state transitions?

## 2. Mathematical Foundations

### State Representation

Describe how the internal state is represented mathematically. Include relevant equations and their interpretation.

### Evolution Dynamics

Explain the state evolution equations. For differential realizations, present the differential equations. For simulative realizations, explain the transition rules.

```
Mathematical formalism:
[Your equations here]

Parameters:
- parameter_1: description
- parameter_2: description
```

### Invariant Preservation

Describe how the system preserves its stated invariants. If invariants are enforced through topology, explain the topological structure. If enforced through regularization, explain the training objectives.

## 3. Implementation Structure

Explain the key components in `gfn/realizations/[your_architecture]/`:

### Core Components

- **Model**: Main neural network class and forward pass logic
- **State**: State representation and initialization
- **Dynamics**: How state forces are computed
- **Integration**: How continuous dynamics are discretized (if applicable)

### Optional Components

- **Constraints**: Module for enforcing structural invariants
- **Monitoring**: Tools for observing internal state evolution
- **Serialization**: Model save/load functionality

## 4. Quick Start Guide

```python
import gfn

# Create the model
model = gfn.create("your_architecture", config=your_config)

# Initialize state
state = model.init_state(batch_size=32)

# Process sequence
for token in sequence:
    logits, state = model(token, state)

# Generate predictions
output = model.readout(state)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_1` | int | 64 | Description |
| `param_2` | float | 1e-4 | Description |

## 5. Complexity Analysis

### Memory Complexity

Detailed analysis of memory usage:
- **Inference**: [Analysis]
- **Training**: [Analysis]
- **State Storage**: [Analysis]

### Time Complexity

Detailed analysis of computational complexity:
- **Forward Pass**: [Analysis]
- **Backward Pass**: [Analysis]
- **State Update**: [Analysis]

### Practical Considerations

Any caveats or conditions affecting complexity in practice.

## 6. Benchmark Results

Results on standard tasks to validate convergence and correctness.

### Synthetic Tasks

| Task | Metric | Result | Conditions |
|------|--------|--------|------------|
| Parity | Accuracy | X% | L=20 → L=1000 |
| Copy | Accuracy | X% | sequence_length=100 |
| Induction | Accuracy | X% | gap=50 |

### Domain-Specific Tasks

| Task | Metric | Result | Notes |
|------|--------|--------|-------|
| task_1 | metric | value | notes |

## 7. Limitations

- **Limitation 1**: Description and impact
- **Limitation 2**: Description and impact
- **Known Issues**: Any known problems and workarounds

## 8. References

- [Reference 1]: Citation or link
- [Reference 2]: Citation or link

---

## Template Checklist

Before publishing, ensure:

- [ ] All sections completed
- [ ] Mathematical equations properly formatted
- [ ] Code examples are runnable
- [ ] Complexity claims are verified
- [ ] Benchmark results are reproducible
- [ ] Limitations are honestly reported
- [ ] Tests exist in `tests/[your_architecture]/`

## Additional Resources

- [Theory Document](theory.md): Detailed mathematical foundations
- [API Reference](api.md): Complete API documentation
- [Examples](examples.md): Usage examples and tutorials

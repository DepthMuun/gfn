# GFN Architectures Registry

## Available Realizations

This document lists all officially available GFN realizations. Each realization implements the Geometric Flow Networks paradigm with specific mathematical formalisms and complexity characteristics.

---

## Official Realizations

### 1. G-SSM (Geodesic State Space Model)

**Type**: Differential Flow  
**Classification**: Continuous dynamics on Riemannian manifolds

**Description**:  
Formalizes representation as a flow on a learned Riemannian manifold. Evolves phase-space variables through symplectic integration of second-order dynamics defined by learned Christoffel symbols.

**Core Mechanism**:
- Symplectic integration (Leapfrog, Yoshida, Verlet)
- Learned Christoffel symbols $\Gamma^i_{jk}$ defining domain curvature
- Second-order differential equations on tangent spaces

**Complexity Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| **State Memory** | O(1) | Constant relative to sequence length |
| **Forward Pass** | O(N) sequential | Must process tokens sequentially |
| **Training Memory** | O(1) with adjoint | Adjoint sensitivity method |
| **State Dimension** | Fixed (d) | Determined at initialization |

**Invariants Supported**:
- Hamiltonian conservation (energy-preserving integration)
- Toroidal topology (configurable)
- Phase-space volume preservation (Liouville's theorem)

**Verified Results**:
- 100% accuracy on XOR parity with 50,000× extrapolation
- 100% accuracy on MNIAH with 32,000 token separation
- Hallucination score < 0.045

**Resources**:
- [Documentation](docs/gssm/README.md)
- [Theory](docs/gssm/theory.md)
- [API Reference](docs/gssm/api.md)

---

### 2. ISN (Inertial State Network)

**Type**: Simulative Flow  
**Classification**: Discrete entity-based dynamics

**Description**:  
Maintains a persistent latent world populated by interacting entities. Implements physics-validated state transitions that preserve world coherence metrics.

**Core Mechanism**:
- Entity-based state representation
- Conservation-gated dynamics
- World coherence monitoring

**Complexity Characteristics**:

| **Inference Memory** | O(1) State | 3 KB persistent latent world state |
| **Forward Pass** | O(N) sequential | Token processing |
| **Training Memory** | O(N) or O(1) | Configuration-dependent |
| **State Dimension** | Fixed (d) | Latent world vector |

**Invariants Supported**:
- World coherence preservation
- Entity identity conservation
- Type compatibility constraints

**Verified Results**:
- 0.980 World Coherence on TinyShakespeare
- 80,000x VRAM reduction (State) @ 32k tokens
- PPL 2.48 on Shakespeare (360k params)
- Stable long-range generation

**Resources**:
- [Documentation](docs/isn/README.md)
- [Theory](docs/isn/theory.md)
- [API Reference](docs/isn/api.md)

---

## Complexity Comparison

| Realization | State Memory | Forward Pass | Training Memory | Best For |
|-------------|-----------------|--------------|----------------|----------|
| **G-SSM** | O(1) | O(N) sequential | O(1)$^\dagger$ | Long-context, logical tasks |
| **ISN** | O(1) | O(N) sequential | O(N)$^\ddagger$ | Natural language, world-states |

---

## Adding New Realizations

To add a new GFN realization to this registry:

1. Implement the realization following [CONTRIBUTING.md](CONTRIBUTING.md) guidelines
2. Add entry to this registry with:
   - Realization name and type
   - Complexity characteristics
   - Supported invariants
   - Verified results (if any)
   - Links to documentation

3. Register in `gfn/realizations/registry.py`

**Note**: All realizations must satisfy the GFN paradigm's core principles: state persistence, structural invariants, and deterministic/stochastic structured evolution.

---

## Realization Selection Guide

Choose a realization based on your requirements:

### When to Use G-SSM
- Tasks requiring perfect extrapolation (arithmetic, logic)
- Long-context retrieval without degradation
- Applications where O(1) inference memory is critical
- Tasks where Hamiltonian conservation is beneficial

### When to Use ISN
- Language modeling with structured world-state
- Tasks benefiting from explicit entity tracking
- Applications requiring interpretable internal states
- Scenarios where world coherence matters

### When to Develop a New Realization
- Existing realizations don't match your task's structure
- You have a novel mathematical formalism for GFN principles
- You need different complexity tradeoffs
- Domain-specific invariants are required

---

## Community Realizations

*This section will be populated with community-contributed realizations as they are added.*

---

## Maintenance

This registry is maintained by the DepthMuun team. To update information or add new realizations, submit a pull request following the contribution guidelines.

**Last Updated**: March 2026  
**Framework Version**: 2.6.6

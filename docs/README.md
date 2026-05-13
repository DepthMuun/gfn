# GFN Framework Documentation

## Welcome to Geometric Flow Networks

This is the unified documentation for the GFN (Geometric Flow Networks) framework. GFN is a physics-informed paradigm for sequential intelligence that reformulates computation as the evolution of persistent state according to structural invariants.

## Documentation Map

### For Users

| Topic | Document | Description |
|-------|----------|-------------|
| **Paradigm Overview** | [README.md](../README.md) | High-level introduction to GFN |
| **Theoretical Foundations** | [THEORY.md](THEORY.md) | Complete mathematical theory |
| **Available Architectures** | [ARCHITECTURES.md](ARCHITECTURES.md) | Realization registry and comparison |
| **Getting Started** | [docs/gssm/guides/01-introduction/02-installation.md](gssm/guides/01-introduction/02-installation.md) | Setup instructions |

### For Developers

| Topic | Document | Description |
|-------|----------|-------------|
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) | Standards for new realizations |
| **Realization Template** | [realization_template.md](realization_template.md) | Template for new architectures |
| **API Reference** | [docs/gssm/guides/03-reference/02-api-classes.md](gssm/guides/03-reference/02-api-classes.md) | Complete API documentation |

### For Researchers

| Topic | Document | Description |
|-------|----------|-------------|
| **Paradigm Paper** | [GEOMETRY_IS_ALL_YOU_NEED.tex](../GEOMETRY_IS_ALL_YOU_NEED.tex) | Primary reference |
| **Research Papers** | [docs/gssm/00_papers/](gssm/00_papers/) | Theoretical extensions |

## Realization Documentation

### G-SSM (Geodesic State Space Model)

A differential realization using continuous dynamics on Riemannian manifolds.

- **Documentation**: [docs/gssm/README.md](gssm/README.md)
- **Type**: Differential Flow
- **Complexity**: O(1) inference memory

### ISN (Inertial State Network)

A simulative realization using discrete entity-based dynamics.

- **Documentation**: [docs/ISN/README.md](ISN/README.md)
- **Type**: Simulative Flow
- **Complexity**: O(1) or O(world_size)

## Key Concepts

### The GFN Paradigm

GFN replaces statistical correlation with geometric flow. Key principles:

1. **State Persistence**: Computation operates on persistent state vectors
2. **Structural Invariants**: Domain constraints limit valid transitions
3. **Deterministic Evolution**: Structured dynamics govern state changes

### Complexity Characteristics

GFN is a **paradigm**, not a specific architecture. Complexity varies:

| Realization | Inference Memory | Forward Pass |
|-------------|-----------------|--------------|
| G-SSM | O(1) or O(d) | O(N) sequential |
| ISN | O(1) or O(world_size) | O(N) sequential |

*Not all GFN realizations achieve O(1) memory complexity.*

## Navigation

```
GFN Framework
├── README.md                    # Paradigm overview
├── THEORY.md                    # Theoretical foundations
├── ARCHITECTURES.md             # Realization registry
├── CONTRIBUTING.md              # Developer guidelines
└── docs/
    ├── THEORY.md               # (duplicate for navigation)
    ├── ARCHITECTURES.md        # (duplicate for navigation)
    ├── realization_template.md # Template for new realizations
    ├── gssm/                   # G-SSM documentation
    └── ISN/                    # ISN documentation
```

## Resources

- [GitHub Repository](https://github.com/DepthMuun/gfn)
- [Issue Tracker](https://github.com/DepthMuun/gfn/issues)
- [Citation](CITATION.cff)

---

**GFN Framework**  
*Version 2.6.6 | March 2026*  
*DepthMuun Research*

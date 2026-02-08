# Manifold Project Documentation

## Documentation Structure

This documentation is organized into thematic modules to make navigation and understanding of the project easier. Each section addresses a specific aspect of the system.

---

## Module 1: Introduction

**Objective:** Provide general context about the project, its purpose, and basic structure.

Included files:
- [01-introduction/01-about-the-project.md](01-introduction/01-about-the-project.md): Project overview, motivation, and main goals.
- [01-introduction/02-installation.md](01-introduction/02-installation.md): System requirements, dependencies, and step-by-step installation guide.
- [01-introduction/03-archive-structure.md](01-introduction/03-archive-structure.md): Description of the directory structure and source code organization.

---

## Module 2: Core Concepts

**Objective:** Explain the theoretical principles and mathematical concepts that underpin the system.

Included files:
- [02-concepts-core/01-physical-model.md](02-concepts-core/01-physical-model.md): Foundations of Hamiltonian mechanics, symplectic systems, and variational formulation.
- [02-concepts-core/02-Riemannian-geometry.md](02-concepts-core/02-Riemannian-geometry.md): Metrics, Christoffel symbols, curvature, and Riemannian manifolds.
- [02-concepts-core/03-geodetic-flow.md](02-concepts-core/03-geodetic-flow.md): The concept of geodesic flow, trajectory integration, and optimality properties.
- [02-concepts-core/04-dynamic-systems.md](02-concepts-core/04-dynamic-systems.md): Symplectic integrators, energy preservation, and numerical stability.

---

## Module 3: Technical Reference

**Objective:** Document the API, configuration parameters, and implementation details.

Included files:
- [03-reference/01-constants.md](03-reference/01-constants.md): Complete catalog of all system constants with values, valid ranges, and behavior effects.
- [03-reference/02-api-classes.md](03-reference/02-api-classes.md): Documentation of the main classes, their methods, and interface contracts.
- [03-reference/03-integrators.md](03-reference/03-integrators.md): Description of available integrators, implemented algorithms, and selection criteria.
- [03-reference/04-geometries.md](03-reference/04-geometries.md): Catalog of implemented geometries, specific parameters, and recommended use cases.

---

## Module 4: Practical Guides

**Objective:** Provide detailed instructions for common tasks and specific use cases.

Included files:
- [04-guides/01-quick-start-guide.md](04-guides/01-quick-start-guide.md): Tutorial to run the first experiment with default configuration.
- [04-guides/02-advanced-configuration.md](04-guides/02-advanced-configuration.md): Guide to adjust parameters, create custom configurations, and optimize performance.
- [04-guides/03-problem-solving.md](04-guides/03-problem-solving.md): Common problems, frequent error messages, and debugging strategies.
- [04-guides/04-numeric-validation.md](04-guides/04-numeric-validation.md): Consistency tests, CPU/GPU parity verification, and results validation.

---

## Module 5: Technical Analyses

**Objective:** Document technical studies, comparisons, and design decisions.

Included files:
- [05-analysis/01-comparison-versions.md](05-analysis/01-comparison-versions.md): Comparative analysis between v2.6.5 and the current development version.
- [05-analysis/02-viability-reversal-constants.md](05-analysis/02-viability-reversal-constants.md): Detailed study on the possibility of reverting constants to earlier version values.
- [05-analysis/03-logic-audit.md](05-analysis/03-logic-audit.md): Thorough audit of the system's logical and mathematical core, verifying Christoffel correctness, backward pass, energy conservation, and variational formulation.

---

## Module 6: History

**Objective:** Document project evolution, significant changes, and technical decisions.

Included files:
- [06-historical/01-changelog.md](06-historical/01-changelog.md): Change log between versions, from v2.6.5 to the current development version.
- [06-historical/02-technical-decisions.md](06-historical/02-technical-decisions.md): Justification of architectural and technical decisions, including Leapfrog integrator, implicit friction, hysteresis, and fused CUDA kernels.

---

## Documentation Conventions

### Writing Style

This documentation follows a pragmatic and direct style:

- We avoid redundant introductions like "In this document we will..."
- We explicitly recognize failure modes
- We provide concrete values and valid ranges
- We explain the reasoning behind technical decisions

### Typographic Conventions

- `code`: Refers to variable names, functions, and configuration parameters
- **bold**: Used for important technical terms the first time they appear
- *italics*: Used for mathematical concepts and file names

### Documentation Updates

If you find errors or outdated documentation, please:
1. Verify the source code to confirm current behavior
2. Update the documentation to reflect actual behavior
3. Include the update date and the verified code version

---

## Documentation Statistics

The full documentation includes:

| Module | Files | Approximate Words |
|--------|----------|---------------------|
| Introduction | 3 | 1,200 |
| Core Concepts | 4 | 2,500 |
| Technical Reference | 4 | 6,500 |
| Practical Guides | 4 | 5,500 |
| Technical Analyses | 3 | 5,000 |
| History | 2 | 2,500 |
| **Total** | **20** | **~23,200** |

---

## Shortcuts

### For New Users
1. Read [01-about-the-project.md](01-introduction/01-about-the-project.md) for context
2. Follow [02-installation.md](01-introduction/02-installation.md) to set up the environment
3. Run [01-quick-start-guide.md](04-guides/01-quick-start-guide.md) for your first experiment

### For Advanced Users
1. Consult [02-advanced-configuration.md](04-guides/02-advanced-configuration.md) for parameter tuning
2. Review [01-constants.md](03-reference/01-constants.md) for parameter reference
3. Explore [04-geometries.md](03-reference/04-geometries.md) for geometry options

### For Debugging
1. Consult [03-problem-solving.md](04-guides/03-problem-solving.md) for common issues
2. Use [04-numeric-validation.md](04-guides/04-numeric-validation.md) for verification
3. Review [03-logic-audit.md](05-analysis/03-logic-audit.md) for deep validation

### For Migration from v2.6.5
1. Read [01-comparison-versions.md](05-analysis/01-comparison-versions.md) to understand differences
2. Consult [02-viability-reversal-constants.md](05-analysis/02-viability-reversal-constants.md) for reversion guidance
3. Review [01-changelog.md](06-historical/01-changelog.md) for full history

---

## Document Version

Last update: February 2026
Documented code version: Current development (post-v2.6.5)
Number of documentation files: 20
Organization: Modular by functionality

---

**Manifold Labs (Joaquín Stürtz)**

# ISN Architecture: The Physics of Persistent State

The **Inertial State Network (ISN)** is designed around the principle of **Inertial Persistence**. In this realization, the sequence context is not an external buffer (like a KV-cache) but the state of a dynamical system itself.

## The Entity-Centric World ($W$)

Unlike standard GFN realizations that use monolithic state vectors, ISN implements an **Entity-Based Simulation**. The World State $W$ is a collection of discrete **Entities**.

### The Anatomy of an Entity ($E$)
Each entity $E$ is defined by the tuple $E = (id, \tau, p, e, R, s)$:
- **$\tau$ (Type)**: Categorization (NUMBER, CONCEPT, OBJECT, OPERATION).
- **$p$ (Properties)**: Intrinsic attribute vector (e.g., magnitude, parity).
- **$e$ (Embedding)**: Semantic location in the manifold.
- **$R$ (Relations)**: Connections to other entities in the world.
- **$s$ (Dynamic State)**: Temporal variables like momentum or decay.

### Persistence & Interaction
- **Persistence**: Entities survive across time steps unless they undergo **Decay** or **Transformation**.
- **Interactions**: Sequence tokens act as catalysts for interactions between entities:
    - **TRANSFORMATION**: $E_1 + E_2 \to E_3$ (e.g., addition).
    - **RELATION**: Establishing links between concepts.
    - **INFLUENCE**: One entity perturbing the properties of another.
    - **EMERGENCE**: A collection of entities forming a higher-order concept.

## Efficiency Characteristics

| Metric | ISN | Transformer |
|--------|-----|-------------|
| **Memory** | $O(1)$ | $O(L^2)$ or $O(L)$ (with cache) |
| **Inference Time** | $O(L)$ total, $O(1)$ per token | $O(L^2)$ or $O(L)$ per token |
| **Backprop Memory** | $O(1)$ (with Adjoint Method) | $O(L^2)$ |

This architecture allows ISN models like the Shakespeare realization (363k params) to maintain stable throughput of over **2000 TPS** on standard hardware.

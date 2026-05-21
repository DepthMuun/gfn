# Factory and Builder Pattern

## What is the Factory?

The Factory is a component that constructs complete GSSM models from configuration. It translates high-level parameters (like `dim=64, heads=4`) into fully initialized model components.

Think of it as: "The architect that builds the model from blueprints."

---

## The Building Process

### Step 1: Configuration Resolution

The factory receives configuration through multiple pathways:

**Option A: Direct Config**
```
config = ManifoldConfig(vocab_size=1000, dim=64, ...)
model = ModelFactory.create(config=config)
```

**Option B: Keyword Arguments**
```
model = ModelFactory.create(vocab_size=1000, dim=64, heads=4, ...)
```

**Option C: Physics Overrides**
```
model = ModelFactory.create(
    vocab_size=1000,
    physics={'stability': {'base_dt': 0.05}}
)
```

The factory merges these into a complete `ManifoldConfig`.

---

## Specialized Builders

The factory delegates to specialized builders for each component:

### 1. EmbeddingBuilder

**Purpose**: Creates the token embedding layer.

**Input**: `vocab_size`, `dim`, `embedding_mode`

**Output**: `FunctionalEmbedding`

**Decision**:
- If `mode='lookup'` → Standard nn.Embedding
- If `mode='linear'` → Bit projection
- If `mode='continuous'` → Direct projection

### 2. LayerBuilder

**Purpose**: Creates all ManifoldLayer instances.

**Input**: `depth`, `dim`, `heads`, `physics_config`

**Output**: `ModuleList` of `ManifoldLayer` × depth

**Per-Layer Construction**:
```
For each layer ℓ in 1..depth:
    1. Create Geometry (from topology config)
    2. Create PhysicsEngine (with Geometry)
    3. Create Integrator (with PhysicsEngine)
    4. Create Mixer (FlowMixer)
    5. Create Plugins (if enabled)
    6. Assemble ManifoldLayer
```

### 3. ReadoutBuilder

**Purpose**: Creates the output projection layer.

**Input**: `dim`, `vocab_size`, `topology_type`

**Output**: `CategoricalReadout` (or other readout type)

**Decision**:
- If `readout_type='categorical'` → CategoricalReadout
- If `readout_type='implicit'` → ImplicitReadout

---

## Component Dependency Chain

The factory respects dependencies between components:

```
Embedding
    ↓
ManifoldLayer[0] → ManifoldLayer[1] → ... → ManifoldLayer[depth-1]
    ↓
Readout

Where each ManifoldLayer contains:
    Geometry
        ↓
    PhysicsEngine (uses Geometry)
        ↓
    Integrator (uses PhysicsEngine)
        ↓
    Mixer (independent)
        ↓
    Plugins (optional, use layer state)
```

---

## Configuration Hierarchy

### Default Values

The factory uses a hierarchy of defaults:

1. **Code defaults**: Hardcoded in schema.py
2. **Config defaults**: Centralized in defaults.py
3. **User config**: Provided by user
4. **Runtime overrides**: **kwargs in create()

**Precedence** (higher number wins):
```
Runtime overrides > User config > Config defaults > Code defaults
```

### Example

```python
# Code default: base_dt = 0.1
# User sets: base_dt = 0.05
# Runtime: base_dt = 0.02

Result: base_dt = 0.02 (runtime wins)
```

---

## Model Assembly

### Final Assembly Process

```
ModelFactory.create():
    1. Resolve configuration
    2. embedding = EmbeddingBuilder.build(config)
    3. layers = LayerBuilder.build(config)  # List of layers
    4. readout = ReadoutBuilder.build(config)
    5. x0, v0 = initialize_state(config)  # Learnable parameters
    6. model = ManifoldModel(layers, embedding, x0, v0, readout)
    7. Return model
```

---

## Why Use a Factory?

### 1. Single Point of Configuration

All model creation goes through one interface:
```python
model = gfn.create('gssm', vocab_size=1000, dim=64)
```

### 2. Validation

The factory validates configuration before building:
```python
if dim % heads != 0:
    raise ConfigurationError("dim must be divisible by heads")
```

### 3. Consistency

Ensures all components are compatible:
- Geometry matches topology setting
- Integrator matches stability config
- Readout matches topology type

### 4. Extensibility

New components can be added without changing user API:
```python
# Add new plugin
ModelFactory.create(..., new_plugin='enabled')
```

---

## Builder Pattern Benefits

### Separation of Concerns

Each builder handles one component type:
- EmbeddingBuilder: Only embeddings
- LayerBuilder: Only layers
- ReadoutBuilder: Only readouts

### Reusability

Builders can be used independently:
```python
embedding = EmbeddingBuilder.build(config)
# Use embedding in custom model
```

### Testing

Each builder can be tested in isolation:
```python
def test_embedding_builder():
    emb = EmbeddingBuilder.build(test_config)
    assert emb.vocab_size == 1000
```

---

## Configuration Resolution

### From Simple to Complex

**Simple**:
```python
gfn.create('gssm', vocab_size=1000)
# Uses all defaults
```

**Moderate**:
```python
gfn.create('gssm', 
    vocab_size=1000,
    dim=128,
    depth=6,
    physics={'topology': {'type': 'torus'}}
)
# Overrides specific values
```

**Complex**:
```python
config = ManifoldConfig(
    vocab_size=1000,
    dim=128,
    physics=PhysicsConfig(
        topology=TopologyConfig(type='torus', R=3.0),
        stability=StabilityConfig(base_dt=0.05, friction=0.01)
    )
)
model = gfn.create(config)
# Full explicit configuration
```

---

## When to Use Factory

**Always use factory for:**
- Creating production models
- Ensuring configuration consistency
- Validating parameters

**Don't use factory for:**
- Unit tests of individual components
- Custom model architectures
- Research experiments with non-standard structures

---

*File: technical/0_architecture/math/system/factory.md*
*Last Updated: 2026-04-02*

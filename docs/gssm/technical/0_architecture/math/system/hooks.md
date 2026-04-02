# Hooks System

## What are Hooks?

Hooks are injection points in the model's forward pass where external code can execute. They allow extending the model's behavior without modifying its core code.

Think of hooks as: "Event listeners for the model's lifecycle."

---

## The Hook Lifecycle

### Available Hooks

| Hook Name | When Triggered | Purpose |
|-----------|----------------|---------|
| `pre_forward` | Before forward() starts | Setup, preprocessing |
| `state_init` | Initializing (x, v) | Custom state initialization |
| `on_batch_start` | New batch begins | Batch-level setup |
| `on_timestep_start` | Before each sequence position | Modify forces, adjust dt |
| `on_layer_start` | Before each layer | Inject layer-specific logic |
| `on_layer_end` | After each layer | Post-processing, logging |
| `on_timestep_end` | After each position | Readout, collect outputs |
| `on_batch_end` | Batch complete | Cleanup, metrics |
| `wrap_evolution` | Wrap _evolve_sequence | Custom integration schemes |

---

## How Hooks Work

### Registration

Plugins register callbacks at specific hook points:

```
Plugin.register_hooks(manager):
    manager.register("on_timestep_end", self.readout)
```

### Triggering

During forward pass, the model triggers hooks:

```
model.forward():
    self.hooks.trigger("pre_forward")
    for t in sequence:
        self.hooks.trigger("on_timestep_start", x, v, force)
        for layer in layers:
            self.hooks.trigger("on_layer_start", layer, x, v)
            x, v = layer(x, v, force)
            self.hooks.trigger("on_layer_end", layer, x, v)
        logits = self.hooks.trigger("on_timestep_end", x, v)
```

### Execution

When triggered:
1. All registered callbacks execute
2. Results are collected in a list
3. Return values can modify behavior

---

## Hook Types

### 1. Informational Hooks

**Purpose**: Notify that something happened.

**Examples**: `on_batch_start`, `on_batch_end`

**Return**: Ignored (used for side effects like logging)

### 2. Transformation Hooks

**Purpose**: Modify inputs/outputs.

**Examples**: `on_timestep_start` (modifies force), `on_layer_end` (modifies state)

**Pattern**:
```
hook_result = trigger("on_timestep_start", x, v, force)
if hook_result:
    force = force + hook_result[0]
```

### 3. Production Hooks

**Purpose**: Generate outputs.

**Examples**: `on_timestep_end` (readout produces logits)

**Pattern**:
```
logits_list = trigger("on_timestep_end", x, v)
for logits in logits_list:
    outputs.append(logits)
```

### 4. Wrapping Hooks

**Purpose**: Replace entire functions.

**Examples**: `wrap_evolution` (custom integration loop)

**Pattern**:
```
wrapped = trigger("wrap_evolution", evolution_fn)
if wrapped:
    evolution_fn = wrapped[-1]
result = evolution_fn(x, v, ...)
```

---

## Common Use Cases

### Readout via Hooks

**Plugin**: `CategoricalReadout`

**Hook**: `on_timestep_end`

**Behavior**:
- After each timestep, projects state to vocabulary
- Collects logits for output

### Dynamic Time via Hooks

**Plugin**: `DynamicTimePlugin`

**Hook**: `on_timestep_start` (via pre_integrate)

**Behavior**:
- Adjusts dt per head based on state
- Called before integrator step

### Fractal via Hooks

**Plugin**: `FractalPlugin`

**Hook**: `on_layer_end` (via finalize)

**Behavior**:
- Adds micro-manifold refinement
- Called after layer completes

---

## Hook Priority

### Execution Order

Hooks execute in registration order:

```
manager.register("hook", callback_a)  # First
manager.register("hook", callback_b)  # Second

trigger("hook"):
    # Executes: callback_a, then callback_b
```

### Multiple Results

When multiple hooks return values:

```
results = trigger("on_timestep_end")
# results = [logits_from_plugin_a, logits_from_plugin_b]
```

---

## Benefits of Hooks

### 1. Extensibility

Add functionality without modifying core code:
```
# New feature? Just write a plugin!
class MyPlugin(Plugin):
    def register_hooks(self, manager):
        manager.register("on_timestep_end", my_logic)
```

### 2. Composability

Multiple plugins work together:
- DynamicTimePlugin adjusts dt
- FractalPlugin adds refinement
- ReadoutPlugin produces outputs

### 3. Testing

Mock specific behaviors:
```
manager.register("on_timestep_start", mock_force)
```

### 4. Debugging

Inject logging at any point:
```
manager.register("on_layer_end", print_state)
```

---

## When to Use Hooks

**Use hooks when:**
- Adding orthogonal features (don't modify core)
- Need to observe model behavior
- Want to modify behavior conditionally
- Building modular extensions

**Don't use hooks when:**
- Feature requires core changes anyway
- Overhead is unacceptable
- Debugging is needed (use direct code)

---

## Comparison with Direct Code

| Aspect | Direct Code | Hooks |
|--------|-------------|-------|
| Performance | Faster | Slight overhead |
| Modularity | Tight coupling | Loose coupling |
| Extensibility | Hard | Easy |
| Debugging | Straightforward | Indirection |

---

*File: technical/0_architecture/math/system/hooks.md*
*Last Updated: 2026-04-02*

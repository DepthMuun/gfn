# Forward Pass - Complete Data Flow

**Source Files Analyzed:**
- `models/base.py` lines 36-114
- `models/manifold_layer.py` lines 119-241
- `physics/integrators/symplectic/leapfrog.py` lines 38-117

---

## 1. Entry Point: BaseModel.forward()

**Location**: `models/base.py:36`

### Signature
```python
def forward(self, input_ids=None, attention_mask=None, state=None, force_manual=None, **kwargs)
    -> Tuple[Tensor, Tuple[Tensor, Tensor], Dict]
```

### Step 1: Resolve Forces (lines 42-58)

```python
if force_manual is not None:
    all_forces = force_manual
elif input_ids is not None:
    all_forces = self.embedding(input_ids)  # [B, S, D]
```

**Flow**:
- `input_ids` [Batch, Sequence] integer tokens
- `self.embedding` is `FunctionalEmbedding`
- Output: `all_forces` [B, S, D] where D = heads × head_dim

**Mask Creation**:
```python
mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
# or ones if no mask provided
```

---

## 2. State Initialization (lines 73-86)

**Location**: `models/base.py:73`

### From External State
```python
if state is not None:
    x, v = state  # Directly use provided state
```

### Default Initialization
```python
x = self.x0.expand(batch_size, self.x0.shape[1], self.x0.shape[2])
v = self.v0.expand(batch_size, self.v0.shape[1], self.v0.shape[2])
if self.initial_spread > 0:
    x = x + torch.randn_like(x) * self.initial_spread
```

**Shapes**:
- `self.x0` initialized in factory: [1, heads, head_dim]
- After expand: [B, heads, head_dim]
- `initial_spread` adds Gaussian noise

---

## 3. Sequence Evolution Loop (lines 116-168)

**Location**: `models/base.py:_evolve_sequence()`

### Outer Loop: Sequence Positions
```python
for i in range(l_seq_len):  # For each token position
    force = fs[:, i] * ms[:, i]  # [B, D]
```

### Inner Loop: Layers
```python
for layer in self.layers:  # For each manifold layer
    res = layer(local_x, local_v, force)
    local_x, local_v = res[0], res[1]
```

---

## 4. ManifoldLayer Forward (Detailed)

**Location**: `models/manifold_layer.py:119`

### Input Shapes
- `x`: [B, S, H, D] or [B, H, D]
- `v`: Same shape as x
- `force`: [B, S, H×D] or [B, H×D]

### Step 4.1: Reshape (lines 152-185)

**4D Input [B, S, H, D]**:
```python
B, S = x.shape[:2]
x_3d = x.reshape(B * S, self.heads, self.head_dim)  # [B×S, H, D]
v_3d = v.reshape(B * S, self.heads, self.head_dim)
```

**Force Reshaping**:
```python
if force.dim() == 3:  # [B, S, D]
    f_3d = force.reshape(B * S, 1, -1).expand(-1, self.heads, -1)
```

### Step 4.2: Pre-Integrate Plugins (lines 190-193)

```python
dt_base = getattr(self.config.stability, 'base_dt', 0.1)
dt_eff = dt_base
for plugin in self.plugins.values():
    x_3d, v_3d, dt_eff = plugin.pre_integrate(x_3d, v_3d, dt_eff, f_3d)
```

**Plugins**: dynamic_time (adjusts dt per head)

### Step 4.3: Integration (lines 196-198)

```python
res = self.integrator.step(x_3d, v_3d, force=f_3d, dt=dt_eff)
x_stepped, v_stepped = res["x"], res["v"]
```

**Integrator**: Leapfrog, Yoshida, etc. Returns dict with 'x', 'v' keys.

### Step 4.4: Post-Integrate Plugins (lines 200-204)

```python
for plugin in self.plugins.values():
    x_stepped, v_stepped = plugin.post_integrate(
        x_stepped, v_stepped, x_prev, v_prev
    )
```

### Step 4.5: Head Mixing (lines 206-207)

```python
x_mix, v_mix = self.mixer(x_stepped, v_stepped)
```

**FlowMixer**:
- Flattens heads: [B, H, D] → [B, H×D]
- Low-rank mixing: projects through learnable matrices
- Unflattens: [B, H×D] → [B, H, D]

### Step 4.6: Dynamics Routing (lines 209-226)

**Partition Mode** (x_mix is 2D [B, D]):
```python
x_ref_h = x_3d.reshape(B_eff, -1)  # Flatten heads
x_next_flat = self.dynamics_x(x_ref_h, x_mix, context_x=x_ref_h)
x_next = x_next_flat.view(B_eff, self.heads, self.head_dim)
```

**Dynamics Types**:
- `direct`: x_next = x_mix
- `residual`: x_next = x + σ(s) × (x_mix - x)
- `gated`: x_next = g × x_mix + (1-g) × x

### Step 4.7: Topology Wrapping (line 229)

```python
x_next = self.integrator._resolve_topology(x_next)
```

**For Torus**:
```python
def _resolve_topology(self, x):
    return torch.atan2(torch.sin(x), torch.cos(x))  # Wrap to [-π, π]
```

### Step 4.8: Finalize Plugins (lines 231-233)

```python
for plugin in self.plugins.values():
    x_next, v_next = plugin.finalize(x_next, v_next)
```

**Fractal Plugin**: Adds sub-manifold refinement steps

### Step 4.9: Restore Shape (lines 236-240)

```python
if len(original_shape) == 4:
    x_next = x_next.view(B, S, self.heads, self.head_dim)
    v_next = v_next.view(B, S, self.heads, self.head_dim)
```

---

## 5. Integrator Step (Detailed)

**Location**: `physics/integrators/symplectic/leapfrog.py:38`

### Leapfrog Algorithm (lines 90-115)

**Input**: x [B, H, D], v [B, H, D], force [B, H, D], dt

```python
for i in range(steps):
    # 1. Resolve friction coefficient μ
    mu1 = self._resolve_friction_mu(curr_x, curr_v, force=force)
    
    # 2. Compute acceleration
    a1 = self._get_acceleration(curr_x, curr_v, force, dt=eff_dt)
    a1_nf = a1 + mu1 * curr_v
    
    # 3. Half-step velocity (Kick)
    v_half = (curr_v + 0.5 * eff_dt * a1_nf) / (1.0 + 0.5 * eff_dt * mu1 + EPS)
    v_half = self._clamp_velocity(v_half)
    
    # 4. Full-step position (Drift)
    curr_x = self._resolve_topology(curr_x + eff_dt * v_half)
    
    # 5. Re-evaluate acceleration at new position
    mu2 = self._resolve_friction_mu(curr_x, v_half, force=force)
    a2 = self._get_acceleration(curr_x, v_half, force, dt=eff_dt)
    a2_nf = a2 + mu2 * v_half
    
    # 6. Final half-step velocity (Kick)
    a_avg = (a1_nf + a2_nf) / 2
    mu_avg = (mu1 + mu2) / 2
    curr_v = (curr_v + eff_dt * a_avg) / (1.0 + eff_dt * mu_avg + EPS)
    curr_v = self._clamp_velocity(curr_v)
```

**Return**: `{'x': curr_x, 'v': curr_v}`

---

## 6. Physics Engine Integration

**Location**: `physics/integrators/base.py:72`

### Acceleration Computation

```python
def _get_acceleration(self, x, v, force, dt, **kwargs):
    res = self.physics_engine.compute_acceleration(x, v, force=force, dt=dt, **kwargs)
    if isinstance(res, tuple):
        return res[0]  # (accel, friction) - return accel only
    return res
```

**In PhysicsEngine.compute_acceleration()**:
```python
# 1. Geometry: Christoffel symbols
geo_out = self.geometry(x, v, force=force)
if isinstance(geo_out, tuple):
    christoffel, mu_geo = geo_out
else:
    christoffel = geo_out
    mu_geo = 0.0

# 2. Friction
mu_total = self.get_friction_coefficient(x, v, mu_geo=mu_geo)
friction_term = mu_total * v

# 3. Net acceleration
net_accel = -christoffel - friction_term + force

# 4. Add auxiliary forces (hysteresis, stochastic, curiosity)
if self.hysteresis is not None:
    net_accel = net_accel + self.hysteresis(x, v)
if self.stochasticity_module is not None:
    net_accel = net_accel + self.stochasticity_module(x, v, dt)
if self.curiosity_module is not None:
    net_accel = net_accel + self.curiosity_module(x, v)

return net_accel
```

---

## 7. Readout Generation

**Location**: `models/base.py:151-154`

### Hook-Based Readout

```python
step_res = self.hooks.trigger("on_timestep_end", x=local_x, v=local_v)
for r in step_res:
    if isinstance(r, torch.Tensor):
        l_logits.append(r)  # Readout plugin produces logits
```

**CategoricalReadout**:
```python
def forward(self, x):
    # x: [B, H, D]
    if self.topology == 'torus':
        x_enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    else:
        x_enc = x
    logits = self.proj(x_enc.flatten(-2))  # [B, vocab_size]
    return logits
```

---

## 8. Complete Forward Pass Flow Summary

```
Input: token_ids [B, S]

1. Embedding
   token_ids [B, S] → forces [B, S, D]

2. State Init
   x0 [1, H, D] → expand → x [B, H, D]
   v0 [1, H, D] → expand → v [B, H, D]

3. For each timestep t in [0, S):
   
   a. Extract force: f = forces[:, t]  # [B, D]
   
   b. For each layer in depth:
      
      i. Reshape: [B, H, D] → [B, H, D] (no change if 3D)
      
      ii. Pre-integrate plugins (adjust dt)
      
      iii. Integrator.step(x, v, f, dt):
           - Get acceleration from PhysicsEngine
           - Leapfrog: Kick-Drift-Kick
           - Return x_stepped, v_stepped
      
      iv. Post-integrate plugins
      
      v. Mixer(x_stepped, v_stepped) → x_mix, v_mix
      
      vi. Dynamics routing:
          - Flatten: [B, H, D] → [B, H×D]
          - Apply: x_next = dynamics(x, x_mix)
          - Unflatten: [B, H×D] → [B, H, D]
      
      vii. Topology wrap (atan2 for torus)
      
      viii. Finalize plugins (fractal)
   
   c. Readout(x) → logits [B, vocab_size]
   
   d. Store: logits[t], x_seq[t], v_seq[t]

4. Stack outputs
   logits: [S, B, V] → [B, S, V]
   x_seq: [S, B, H, D] → [B, S, H, D]
   v_seq: [S, B, H, D] → [B, S, H, D]

5. Return: (logits, (x_final, v_final), state_info)
```

---

## 9. Tensor Shape Transformations

| Stage | Shape | Notes |
|-------|-------|-------|
| Input | [B, S] | token_ids |
| Embedding | [B, S, D] | D = heads × head_dim |
| State Init | [B, H, D] | heads × head_dim = D |
| Layer Input | [B×S, H, D] | If sequence mode |
| After Mixer | [B×S, H, D] | Mixed across heads |
| Dynamics Flat | [B×S, H×D] | For routing |
| Output | [B, S, V] | logits |

---

*File: technical/0_architecture/math/09_forward_pass.md*
*Last Updated: 2026-04-02*

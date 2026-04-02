# Plugins - Dynamic Time

## What is Dynamic Time?

Dynamic Time is a plugin that allows each head to have its own adaptive time step. Instead of using a fixed $dt$ for all heads, each head learns its own optimal integration speed based on its current state.

Think of it as: "Different experts (heads) think at different speeds."

---

## The Problem with Fixed Time Steps

### Standard Approach
All heads use the same $dt$:
$$x_{n+1} = x_n + dt \cdot v_n$$

**Issues**:
- Some regions need small steps (high curvature)
- Other regions can use large steps (flat areas)
- Fixed $dt$ is suboptimal for all heads

### Dynamic Solution
Each head $h$ has its own $dt_h$:
$$x_{n+1}^{(h)} = x_n^{(h)} + dt_h \cdot v_n^{(h)}$$

---

## How It Works

### Step 1: Learnable Base Time Steps

Each head has a learnable parameter $\theta_h$:
$$dt_{h,base} = \text{softplus}(\theta_h)$$

Where softplus ensures positivity:
$$\text{softplus}(x) = \log(1 + e^x)$$

**Initialization**:
$$\theta_h \approx \log(\exp(dt_{target}) - 1)$$

This gives $dt_{h,base} \approx dt_{target}$ at initialization.

### Step 2: State-Dependent Gating

The base $dt$ is modulated by a gating function based on current state:

$$gate_h = \sigma(W_{gate} \cdot x_h + b_{gate})$$

Where:
- $\sigma$ = sigmoid (outputs 0 to 1)
- $W_{gate}$ = learnable weights
- $x_h$ = position of head $h$

### Step 3: Effective Time Step

$$dt_{h,eff} = dt_{h,base} \cdot gate_h$$

Clamped to safe range:
$$dt_{h,eff} = \text{clamp}(dt_{h,eff}, dt_{min}, dt_{max})$$

**Typical values**:
- $dt_{min} = 0.0001$ (very small)
- $dt_{max} = 0.5$ (conservative maximum)

---

## Gating Types

### Standard Gating

**Input**: Position $x_h$ only

$$gate_h = \sigma(W \cdot x_h + b)$$

**Use case**: Position-dependent adaptation

### Thermodynamic Gating

**Input**: Both position $x_h$ and velocity $v_h$

$$gate_h = \sigma(W_x \cdot x_h + W_v \cdot v_h + b)$$

**Use case**: Full state-dependent adaptation (more expressive)

---

## Physical Interpretation

### Analogy: Variable Speed Processing

Think of each head as a processor:
- **Fast regions** (low curvature): Large $dt$ = quick processing
- **Complex regions** (high curvature): Small $dt$ = careful processing

### Energy Landscape

$$dt_{eff} \propto \frac{1}{\|\nabla \text{Energy}\|}$$

In steep regions (high gradient), use small steps.
In flat regions, use large steps.

---

## Benefits

### 1. Stability
Heads in difficult regions automatically slow down.

### 2. Efficiency  
Heads in easy regions automatically speed up.

### 3. Specialization
Each head learns its optimal processing speed.

### 4. Adaptation
Time steps adjust during training as the model learns.

---

## Mathematical Formulation

### Complete Dynamic Time Formula

For head $h$ at layer $\ell$, timestep $t$:

$$dt_{h,\ell}(t) = \text{clamp}\left(\text{softplus}(\theta_{h,\ell}) \cdot \sigma(W_{h,\ell} \cdot x_{h,\ell}(t) + b_{h,\ell}), dt_{min}, dt_{max}\right)$$

Where:
- $\theta_{h,\ell}$ = learnable base parameter
- $W_{h,\ell}, b_{h,\ell}$ = gating parameters
- $x_{h,\ell}(t)$ = position state

### Integration with Dynamic Time

$$v_{n+1/2} = v_n + \frac{dt_{h,eff}}{2} \cdot a(x_n, v_n)$$
$$x_{n+1} = x_n + dt_{h,eff} \cdot v_{n+1/2}$$

Each head uses its own $dt_{h,eff}$ in the integrator.

---

## When to Use

**Use Dynamic Time when:**
- Different heads process different complexity patterns
- You want adaptive stability per head
- Training shows heads need different speeds

**Don't use when:**
- All heads process similar complexity (overkill)
- You need deterministic behavior (dynamic makes it state-dependent)
- Computational budget is tight (adds overhead)

---

## Configuration

```python
physics = {
    'active_inference': {
        'dynamic_time': {
            'enabled': True,
            'type': 'standard',  # or 'thermo'
            'base_dt': 0.1,
            'dt_min': 0.0001,
            'dt_max': 0.5
        }
    }
}
```

---

*File: technical/0_architecture/math/plugins/dynamic_time.md*
*Last Updated: 2026-04-02*

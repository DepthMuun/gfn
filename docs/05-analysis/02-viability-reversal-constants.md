# Viability of Reverting to v2.6.5 Constants

## Executive Summary

We analyzed the possibility of reverting numeric constants to the v2.6.5 values. The main conclusion is that a full reversion is **not viable** due to fundamental architectural changes between versions. However, a **partial and gradual reversion** is possible and recommended to recover convergence.

Differences between versions are not merely numerical: they represent changes in system physics, integrator stability, and model assumptions. Reverting constants without understanding these implications can produce erratic behavior or divergence.

## Detailed Constants Analysis

### Comparison Table

| Constant | v2.6.5 | Current | Ratio | Risk |
|-----------|--------|--------|-------|--------|
| FRICTION_SCALE | 5.0 | 0.02 | 250x lower | HIGH |
| CURVATURE_CLAMP | 20.0 | 3.0 | 6.7x lower | HIGH |
| LAMBDA_H_DEFAULT | 0.01 | 0.0 | Disabled | MEDIUM |
| LAMBDA_G_DEFAULT | 0.001 | 0.00005 | 20x lower | LOW |
| DEFAULT_LR | 1e-3 | 1e-4 | 10x lower | MEDIUM |
| ADAM_BETA2 | 0.999 | 0.99 | More stable | LOW |
| DEFAULT_DT | 0.1 | 0.05 | 2x lower | MEDIUM |
| EPSILON_STRONG | 1e-4 | 1e-7 | 1400x lower | HIGH |
| EPSILON_STANDARD | 1e-6 | 1e-7 | 10x lower | HIGH |
| READOUT_GAIN | 10.0 | 2.0 | 5x lower | LOW |
| SIREN_OMEGA_0 | 30.0 | 10.0 | 3x lower | MEDIUM |
| IMPULSE_SCALE | 1.0 | 0.5 | 2x lower | MEDIUM |
| DEFAULT_FRICTION | 0.05 | 0.002 | 25x lower | HIGH |
| MAX_WEIGHT_NORM | 10.0 | 5.0 | 2x lower | LOW |
| INIT_STD | 0.02 | 0.01 | 2x lower | LOW |
| INIT_X0_SCALE | 0.02 | 0.01 | 2x lower | LOW |
| INIT_V0_SCALE | 0.01 | 0.005 | 2x lower | LOW |
| GATE_BIAS_OPEN | 2.0 | 1.0 | 2x lower | LOW |

## Group Analysis

### Group A: Physics and Dynamics (HIGH Risk)

This group contains constants that directly affect system physics: friction, curvature, and forces. The changes are an order of magnitude or more.

**FRICTION_SCALE (5.0 → 0.02)**

This is the most dramatic difference. The current value is 250 times lower than v2.6.5. A FRICTION_SCALE of 5.0 applies extreme friction that practically freezes the system. The current value of 0.02 allows significant exploration.

If we revert to 5.0:
- The system will slow down almost instantly
- Geodesic dynamics will be severely inhibited
- Trajectories will not have time to evolve
- The model will essentially behave like static attention

We do not recommend a full reversion. We suggest an intermediate value like 0.5 or 1.0 for experimentation.

**CURVATURE_CLAMP (20.0 → 3.0)**

The lower curvature clamp in the current version prevents the metric from becoming singular. A clamp of 20.0 would allow extreme curvature that can cause numerical instability.

If we revert to 20.0:
- The metric can become ill-conditioned
- Christoffel symbols can blow up
- Loss can become NaN in high-curvature regions

We suggest keeping the current clamp (3.0) or gradually increasing it to 5.0 maximum.

**DEFAULT_FRICTION (0.05 → 0.002)**

25 times lower than v2.6.5. It affects base friction in the system. A value of 0.05 provides significant damping.

If we revert:
- More damping but less exploration
- Faster convergence but to local minima
- Possible underfitting

We suggest 0.005 as an intermediate point.

**EPSILON_STRONG and EPSILON_STANDARD**

Both were reduced significantly. Small epsilon prevents division by near-zero numbers but can amplify numerical noise.

If we revert to larger epsilon (1e-4, 1e-6):
- More stability for small values
- Less precision in calculations
- Possible excessive clipping

The current value (1e-7) is a balance. Revert only if we observe instability from numerical noise.

### Group B: Optimization (MEDIUM Risk)

This group contains optimizer constants and timestep. They affect convergence but are more tolerant to changes.

**DEFAULT_LR (1e-3 → 1e-4)**

Learning rate is 10 times smaller. This makes training slower but more stable.

If we revert to 1e-3:
- Faster training
- Possible divergence on difficult problems
- More sensitive to initialization

We suggest 5e-4 as an intermediate point if more speed is desired.

**ADAM_BETA2 (0.999 → 0.99)**

Lower beta2 makes the optimizer more sensitive to recent gradients. It is more stable under noise but can oscillate more.

The current value (0.99) is more robust. Reverting to 0.999 can help if the gradient has long-range structure.

**DEFAULT_DT (0.1 → 0.05)**

The timestep is 2 times smaller. More precise but requires more steps per token.

If we revert to 0.1:
- Each integration step covers more distance
- Less precise in high-curvature regions
- More prone to "jumping" optimal geodesics

We suggest 0.08 as a compromise.

**Adam EPSILON (1e-8 → 1e-7)**

Larger epsilon prevents division by zero in the optimizer. More robust but potentially less precise.

The current value (1e-7) is standard in modern implementations. Reverting is safe.

### Group C: Initialization (LOW Risk)

This group contains initialization parameters for weights and activations. Small changes have limited effects.

**INIT_STD, INIT_X0_SCALE, INIT_V0_SCALE, GATE_BIAS_OPEN**

All reduced by a factor of 2. More conservative initializations produce smoother gradients.

If we revert:
- Larger initial gradients
- Can help problems that require more initial "energy"
- Risk of early saturation or divergence

We suggest reverting them gradually if more initial capacity is needed.

**SIREN_OMEGA_0 (30.0 → 10.0)**

Lower SIREN frequency produces smoother activations. A value of 30.0 produces rapid oscillations that require more steps to converge.

If we revert:
- More oscillatory activations
- Can capture high frequencies better
- Requires more integration steps

It depends on the problem. For high-frequency signals, a high omega_0 can be beneficial.

**READOUT_GAIN (10.0 → 2.0)**

Reduced output gain. A gain of 10.0 concentrates probability in fewer options.

If we revert:
- Sharper output distribution
- Faster training if there are few output classes
- Higher risk of miscalibrated confidence

Safe to revert if the problem has few output classes.

## Recommendations

### Scenario 1: Recover Fast Convergence

If the goal is simply for the model to converge like v2.6.5:

1. Keep FRICTION_SCALE low (0.02-0.1)
2. Reduce DEFAULT_LR gradually (5e-4)
3. Increase DEFAULT_DT to 0.08
4. Reduce LEAPFROG_SUBSTEPS to 2
5. Do not revert epsilon

### Scenario 2: Physics Closer to v2.6.5

If the goal is to reproduce v2.6.5 physics:

1. Set FRICTION_SCALE to 0.5 (not 5.0 directly)
2. Set DEFAULT_FRICTION to 0.01
3. Set CURVATURE_CLAMP to 5.0 (not 20.0)
4. Reduce LAMBDA_G_DEFAULT to 0.0002
5. Set DEFAULT_LR to 5e-4

### Scenario 3: Full Reversion (Not Recommended)

If full reversion is insisted on:

1. Test first with a small data subset
2. Monitor loss closely
3. Be prepared to reduce timestep
4. You will likely need to reduce learning rate to 1e-5

## Justification for the Changes

The constant changes between v2.6.5 and the current version were not arbitrary. Each change solved a specific problem.

FRICTION_SCALE reduced from 5.0 to 0.02:
- Problem: excessive friction prevented geodesic exploration
- Solution: minimal friction to allow the system to evolve
- Effect: the system can find shorter geodesics

DEFAULT_LR reduced from 1e-3 to 1e-4:
- Problem: high learning rate caused oscillations and divergence
- Solution: more conservative learning rate
- Effect: slower but more stable convergence

CURVATURE_CLAMP reduced from 20.0 to 3.0:
- Problem: extreme curvature caused singular metrics
- Solution: strict clamp prevents singularities
- Effect: prevents NaNs in loss

EPSILON reduced:
- Problem: large epsilon caused excessive clipping
- Solution: smaller epsilon for better precision
- Effect: more precise gradients

## Conclusions

A full reversion to v2.6.5 **is not viable** because:

1. The integrator architecture changed (it is now implicit)
2. The Python-CUDA parity policy requires small epsilon values
3. The hysteresis system is new and has its own constants
4. Trace normalization is a new feature

A partial reversion is possible following the gradual approach described. We recommend starting with Group C changes, then Group B, and only if necessary Group A with intermediate values.

The current version was designed to be more stable and robust than v2.6.5. The loss of convergence in some experiments is likely due to hyperparameter imbalance, not fundamental bugs.

---

**Manifold Labs (Joaquín Stürtz)**

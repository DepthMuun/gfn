# G-SSM Physics Notes: Momentum Drift and The Overdamped Limit

**Date:** March 18, 2026
**Topic:** Continuous Momentum vs. Discrete Automata

## The Problem: "Momentum Drift"
The Geometric State Space Model (G-SSM) uses continuous physical integrators (Yoshida, Leapfrog) to evolve the state vector over a Riemann manifold (e.g., Torus or Sphere). 
The fundamental equation of motion is:
$v_{t+1} = \frac{v_t + \Delta t \cdot F_{ext}}{1 + \Delta t \cdot \mu_f}$

Where $F_{ext}$ is the external force from the token embedding, and $\mu_f$ is the internal friction.

### Why it fails in strictly discrete sequences length>1000:
In tasks like **Logic XOR** or boolean parity, the model must act as an exact discrete state machine (a Flip-Flop). 
- A token `1` applies a positive force `+F`.
- A token `0` applies exactly `0` force (do nothing).

If $\mu_f$ (friction) is low (e.g., $0.05$), the system operates in an **Underdamped Regime**.
When a long sequence of `1`s is fed, the velocity $v$ saturates to the upper limit (e.g., $10.0$).
When a subsequent long sequence of `0`s (force=0) is fed, the G-SSM should stop changing because $0$ does not alter Parity. However, due to the **momentum** (low friction), it takes $>50$ timesteps for the velocity to decay to zero. During those 50 steps, the state continues to spin and drift wildly, utterly destroying the recorded parity phase.

## The Solution: The Overdamped Limit
To force a continuous mechanical system to behave exactly like a memoryless discrete automaton, the internal friction MUST be set extremely high (e.g., `friction = 2.0` to `5.0`), moving the system into the **Overdamped Regime**.

In the Overdamped Regime, $1 + \Delta t \cdot \mu_f \gg 1$. 
Any residual velocity from the previous step is immediately dissipated in a single timestep.
$v_{t} \approx \frac{F_{ext}}{\mu_f}$

- When token `1` arrives, it moves exactly $\frac{F_{ext}}{\mu_f} \cdot \Delta t$ distance, and effectively stops.
- When token `0` arrives ($F_{ext}=0$), the velocity immediately drops to 0. There is no coasting. There is no momentum drift.

### Implementation Checklist for Logic Gates / Counting Automata
If using G-SSM for perfectly discrete sequence resolution (XOR parity, Counting, Exact Pattern Matching), ALWAYS use:
1. `friction: 2.0` (or higher) to kill momentum.
2. `velocity_saturation: 15.0` (allow high instant velocity to traverse the manifold in one step).
3. `holographic=True` (or Trace Normalization) to prevent long-term scaling collapse.
4. `integrator: yoshida` (to properly handle the massive $\Delta v$ kicks).

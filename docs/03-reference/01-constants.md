# Constants Reference

## Complete Catalog

This section documents all system constants, organized by functional category. For each constant, the current value, valid range, and effect on system behavior are provided.

## Embedding Constants

EMBEDDING_SCALE controls the output scale of implicit embeddings. Value: 1.5. Typical range: 0.5 to 3.0. High values amplify the output signal but can cause saturation. Low values attenuate the signal but can make the model less expressive.

IMPULSE_SCALE determines the magnitude of functional impulse in embeddings. Value: 0.5. Range: 0.1 to 2.0. This parameter scales the initial "push" of the embedding. High values produce more energetic but less stable initializations. The current value is half the original (1.0) to improve stability.

SIREN_OMEGA_0 is the base frequency for SIREN initialization. Value: 10.0. Range: 1.0 to 50.0. High values produce faster oscillations but require more integration steps to converge. Reduced from 30.0 to 10.0 for a better stability-expressiveness balance.

## Readout Constants

READOUT_GAIN controls the logit gain in the readout layer. Value: 2.0. Range: 0.5 to 10.0. This parameter scales logits before softmax. High values concentrate probability on a few options. Low values produce flatter distributions. Reduced from 10.0 to 2.0 for smoother gradients.

## Geometry Constants

CURVATURE_CLAMP sets the maximum effective curvature limit. Value: 3.0. Range: 1.0 to 20.0. This clamp prevents the metric from becoming singular. High values allow more curvature but risk instability. Low values constrain geometry to be nearly flat. The current value is 6.7 times lower than v2.6.5 (20.0).

TOROIDAL_CURVATURE_SCALE scales curvature in toroidal geometries. Value: 0.01. Range: 0.001 to 0.1. High values amplify torsion effects. Low values produce locally near-flat geometry.

TOROIDAL_MAJOR_RADIUS and TOROIDAL_MINOR_RADIUS define the base toroidal geometry. Values: 2.0 and 1.0 respectively. Only relevant for toroidal geometry.

## Friction Constants (Critical)

FRICTION_SCALE is the main friction coefficient. Value: 0.02. Range: 0.0 to 0.5. This is the most sensitive parameter in the system. The current value is 250 times lower than v2.6.5 (5.0). High values damp dynamics but can prevent convergence. Low values allow more exploration but can cause oscillations.

FRICTION_SCALE_CUDA must match FRICTION_SCALE for parity. Value: 0.02.

VELOCITY_FRICTION_SCALE introduces velocity-dependent friction. Value: 0.02. Range: 0.0 to 0.1. High-velocity regions experience additional friction.

DEFAULT_FRICTION is the base friction coefficient. Value: 0.002. Range: 0.0 to 0.05. This is the fallback value when explicit friction is not specified. 25 times lower than v2.6.5 (0.05).

## Numerical Stability Constants

EPSILON_STRONG is the epsilon for division protection (strong protection). Value: 1e-7. Range: 1e-9 to 1e-5. Used in critical operations where stability is prioritized.

EPSILON_STANDARD is the epsilon for division protection (standard protection). Value: 1e-7. Range: 1e-9 to 1e-5. Used in routine operations. Must match the value in CUDA kernels.

EPSILON_SMOOTH is the epsilon for gradient smoothing. Value: 1e-7. Useful to prevent spurious gradients.

CLAMP_MIN_STRONG and CLAMP_MIN_STANDARD set minimums for denominator clamping. Value: 1e-7 for both.

## Loss Constants

LAMBDA_H_DEFAULT is the weight of the Hamiltonian loss term. Value: 0.0. Range: 0.0 to 0.01. This term reinforces energy conservation. Disabled in the current version (was 0.01 in v2.6.5). A zero value means the term does not contribute to total loss.

LAMBDA_G_DEFAULT is the weight of the geodesic loss term. Value: 0.00005. Range: 0.0 to 0.001. This term reinforces geodesic trajectories. 20 times lower than v2.6.5 (0.001). High values enforce strict geometry.

LAMBDA_N_DEFAULT is the weight of the Noether symmetry term. Value: 0.0. Range: 0.0 to 0.01.

LAMBDA_K_DEFAULT is the weight of the kinetic energy term. Value: 0.0001. Range: 0.0 to 0.001. 10 times lower than v2.6.5 (0.001).

GEODESIC_FUSED_SCALE scales fused geodesic regularization. Value: 100.0. Multiplication factor for the fused term.

## Optimizer Constants

DEFAULT_LR is the default learning rate. Value: 1e-4. Range: 1e-6 to 1e-2. 10 times lower than v2.6.5 (1e-3). Smaller values require more steps but are more stable.

ADAM_BETA1 is Adam's beta1 parameter. Value: 0.9. Standard in the literature.

ADAM_BETA2 is Adam's beta2 parameter. Value: 0.99. v2.6.5 used 0.999. The current value is more conservative and stable under noise.

ADAM_EPSILON is the epsilon for the Adam optimizer. Value: 1e-7. v2.6.5 used 1e-8. A larger epsilon prevents division by very small numbers.

DEFAULT_WEIGHT_DECAY is the default weight decay. Value: 0.001. 10 times lower than v2.6.5 (0.01).

MAX_WEIGHT_NORM is the maximum weight norm for retraction. Value: 5.0. v2.6.5 used 10.0. Limits weight growth.

## Initialization Constants

INIT_STD is the standard deviation for normal initialization. Value: 0.01. 2 times lower than v2.6.5 (0.02). More conservative initializations produce smoother gradients.

INIT_X0_SCALE is the scale for position initialization. Value: 0.01. 2 times lower than v2.6.5 (0.02).

INIT_V0_SCALE is the scale for velocity initialization. Value: 0.005. 2 times lower than v2.6.5 (0.01).

GATE_BIAS_OPEN is the bias for open gates. Value: 1.0. sigmoid(1.0) ≈ 0.73. v2.6.5 used 2.0. More conservative gates.

GATE_BIAS_CLOSED is the bias for closed gates. Value: -3.0. sigmoid(-3.0) ≈ 0.05.

## Integration Constants

DEFAULT_DT is the default timestep. Value: 0.05. v2.6.5 used 0.1. 2 times lower. Smaller timesteps require more steps but are more precise.

LEAPFROG_SUBSTEPS is the number of Leapfrog substeps. Value: 3. v2.6.5 used 5. Fewer substeps produce cleaner gradients.

HEUN_SAFETY_FACTOR is the safety factor for Heun. Value: 0.9.

PARALLEL_SCAN_THRESHOLD is the threshold for parallel scan. Value: 32.

## Active Inference Constants

DEFAULT_PLASTICITY is the plasticity coefficient for reactive curvature. Value: 0.02. 2 times higher than v2.6.5 (0.01).

SINGULARITY_THRESHOLD is the threshold for singularity activation. Value: 0.5. v2.6.5 used 0.8. Earlier activation prevents instability.

BLACK_HOLE_STRENGTH is the "black hole" intensity. Value: 1.5. v2.6.5 used 2.0.

REACTIVE_CURVATURE_LR is the learning rate for reactive curvature. Value: 0.01.

MAX_CURVATURE_ADJUSTMENT is the maximum curvature adjustment per step. Value: 0.1.

SINGULARITY_GATE_SLOPE is the slope of the singularity gate. Value: 0.5. v2.6.5 used 1.0.

## Hysteresis Constants

HYSTERESIS_FORGET_GATE_INIT is the initial value of the forget gate. Value: 0.9. sigmoid(2.0) ≈ 0.88.

HYSTERESIS_STATE_MOMENTUM is the momentum for state updates. Value: 0.95.

HYSTERESIS_MIN_STRENGTH and HYSTERESIS_MAX_STRENGTH are hysteresis strength limits. Values: 0.01 and 0.5.

HYSTERESIS_GHOST_FORCE_SCALE is the ghost force coefficient. Value: 0.1.

## Toroidal Geometry Constants

TOROIDAL_PERIOD is the period of toroidal coordinates. Value: 2π ≈ 6.28.

TOROIDAL_ANGLE_WEIGHT is the weight for angular components. Value: 1.0.

TOROIDAL_RADIUS_WEIGHT is the weight for radial components. Value: 0.5.

TOROIDAL_MIN_RADIUS and TOROIDAL_MAX_RADIUS are radial limits. Values: 0.1 and 3.0.

## Aggregation Constants

AGGREGATION_MOMENTUM_DEFAULT is the momentum for aggregation. Value: 0.9.

AGGREGATION_MIN_SAMPLES is the minimum samples for valid aggregation. Value: 1.

AGGREGATION_MAX_TRAJECTORY_LEN is the maximum trajectory length. Value: 1000.

STATE_BUFFER_SIZE is the state buffer size. Value: 100.

## Training Stability Constants

GRAD_CLIP_NORM is the gradient norm clip. Value: 1.0.

LR_WARMUP_STEPS and LR_WARMUP_RATIO are warmup parameters. Values: 100 and 0.1.

VELOCITY_SATURATION is the velocity saturation. Value: 100.0. v2.6.5 used 50.0.

GRADIENT_CLIP_VALUE is the per-parameter gradient clip. Value: 1.0.

ADAPTIVE_FRICTION_SCALE is the adaptive friction scale. Value: 0.1.

HUBER_DELTA is the delta for Huber loss. Value: 1.0.

GATE_SLOPE_LOWERED is the reduced gate slope. Value: 2.0.

## CUDA Constants

CUDA_MAX_THREADS_PER_BLOCK is the maximum threads per block. Value: 256.

CUDA_SHARED_MEMORY_SIZE is the shared memory size. Value: 16384 bytes.

CUDA_REDUCTION_BLOCK_SIZE is the block size for reduction. Value: 32.

CUDA_WARP_SIZE is the warp size. Value: 32.

CUDA_OPTIMIZE_MEMORY enables memory optimization. Value: True.

CUDA_USE_FUSED_KERNELS enables fused kernels. Value: True.

---

**Manifold Labs (Joaquín Stürtz)**

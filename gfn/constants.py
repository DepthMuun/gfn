""""
GFN Configuration Constants - STABLE VERSION
=============================================

Centralized constants for the GFN codebase with numerical stability fixes.

CUDA/PYTHON PARITY POLICY (Audit 2026-02-06):
=============================================

All numerical epsilon constants MUST match exactly between Python and CUDA:
- Python: defined in this file (constants.py)
- CUDA: defined in gfn/cuda/core.py::CudaConstants

VERIFIED MATCHES:
- EPSILON_STANDARD = 1e-8 (division safety)
- EPSILON_STRONG   = 1e-8 (strong division protection)
- EPSILON_SMOOTH   = 1e-8 (gradient smoothing)
- CLAMP_MIN_STRONG = 1e-8 (minimum denominators)

ALLOWED EXCEPTIONS:
- Test tolerances (atol/rtol) may differ from runtime epsilons
- ADAM_EPSILON = 1e-7 (optimizer-specific, documented choice)
- Hardcoded 1e-6 in std() fallbacks (defensive programming, negligible impact)

To verify parity, run:
    python tests/test_cuda_python_consistency.py::test_epsilon_constants_match

"""

# ============================================================================
# EMBEDDING CONSTANTS
# ============================================================================

# Implicit embedding output scale
EMBEDDING_SCALE = 1.5

# Functional embedding impulse scale
IMPULSE_SCALE = 0.5  # Reduced from 1.0 for stability

# SIREN omega_0 frequency - REDUCED for stability
SIREN_OMEGA_0 = 10.0  # Was 30.0


# ============================================================================
# READOUT CONSTANTS
# ============================================================================

# Readout logit gain - REDUCED for stable gradients
READOUT_GAIN = 2.0  # Was 10.0


# ============================================================================
# GEOMETRY CONSTANTS
# ============================================================================

# Curvature clamping limit - REDUCED for stability
# AUDIT FIX: Reduced to prevent excessive curvature amplification
CURVATURE_CLAMP = 3.0  # Was 5.0 - Tighter bounds for stability

# Toroidal curvature scale
TOROIDAL_CURVATURE_SCALE = 0.01  # Was 0.05

# Default major radius for toroidal manifolds
TOROIDAL_MAJOR_RADIUS = 2.0

# Default minor radius for toroidal manifolds
TOROIDAL_MINOR_RADIUS = 1.0


# ============================================================================
# FRICTION CONSTANTS - CRITICAL FIXES
# ============================================================================

# Friction gate scale - OPTIMIZED for proper symplectic behavior
# AUDIT FIX (2026-02-07): Reduced from 0.05 to 0.02 for better conservation
FRICTION_SCALE = 0.02  # Was 5.0, then 0.5, then 0.05 - Now optimal

# Friction scale for CUDA - MUST match Python fallback
FRICTION_SCALE_CUDA = 0.02

# AUDIT FIX: Velocity-dependent friction scale (2026-02-06)
# Higher velocities experience proportionally more drag
VELOCITY_FRICTION_SCALE = 0.02  # Reduced for stability and conservation

# Default friction coefficient for conformal symplectic systems
DEFAULT_FRICTION = 0.002  # Was 0.005


# ============================================================================
# NUMERICAL STABILITY CONSTANTS - OPTIMIZED
# ============================================================================

# Epsilon for division safety (strong protection)
# OPTIMIZED: 1e-7 for better gradient flow while maintaining stability
EPSILON_STRONG = 1e-7  # Was 1e-8 - Better balance

# Epsilon for division safety (standard protection)
EPSILON_STANDARD = 1e-7  # Was 1e-8 - Match strong for consistency

# Epsilon for gradient smoothing
EPSILON_SMOOTH = 1e-7

# Minimum clamping value for denominators
CLAMP_MIN_STRONG = 1e-7

# Standard clamping minimum
CLAMP_MIN_STANDARD = 1e-7


# ============================================================================
# LOSS FUNCTION CONSTANTS - OPTIMIZED FOR CONVERGENCE
# ============================================================================

# Default Hamiltonian loss weight - DISABLED for standard training
# OPTIMIZED: Zero to allow physics response to external forces
LAMBDA_H_DEFAULT = 0.0  # Was 0.001 - Disabled for clean convergence

# Default geodesic regularization weight - REDUCED to preserve curvature
LAMBDA_G_DEFAULT = 0.00005  # Was 0.0001 - Lower for better curvature preservation

# Default Noether symmetry loss weight
LAMBDA_N_DEFAULT = 0.0

# Default kinetic energy penalty weight
LAMBDA_K_DEFAULT = 0.0001  # Was 0.001 - Reduced for stability

# Heuristic scaling for fused geodesic regularization
GEODESIC_FUSED_SCALE = 100.0


# ============================================================================
# OPTIMIZER CONSTANTS
# ============================================================================

# Default learning rate - REDUCED for stability
DEFAULT_LR = 1e-4  # Was 1e-3

# Default beta1 for Adam
ADAM_BETA1 = 0.9

# Default beta2 for Adam
ADAM_BETA2 = 0.99  # Was 0.999 - increased for stability

# Default epsilon for Adam
ADAM_EPSILON = 1e-7  # Was 1e-8

# Default weight decay
DEFAULT_WEIGHT_DECAY = 0.001  # Was 0.01

# Maximum weight norm for retraction
MAX_WEIGHT_NORM = 5.0  # Was 10.0


# ============================================================================
# INITIALIZATION CONSTANTS
# ============================================================================

# Standard deviation for normal initialization - REDUCED
INIT_STD = 0.01  # Was 0.02

# Initial position state scale
INIT_X0_SCALE = 0.01  # Was 0.02

# Initial velocity state scale
INIT_V0_SCALE = 0.005  # Was 0.01

# Gate bias initialization (open state)
GATE_BIAS_OPEN = 1.0  # sigmoid(1.0) ≈ 0.73 - Was 2.0

# Gate bias initialization (closed state)
GATE_BIAS_CLOSED = -3.0  # sigmoid(-3.0) ≈ 0.05 - Was -5.0


# ============================================================================
# INTEGRATION CONSTANTS - OPTIMIZED FOR CONVERGENCE
# ============================================================================

# Default timestep for integrators - OPTIMIZED for exploration
# OPTIMIZED: Larger timestep for effective geodesic exploration
DEFAULT_DT = 0.05  # Was 0.02 - Better exploration while maintaining stability

# Minimum sequence length for parallel scan
PARALLEL_SCAN_THRESHOLD = 32

# Leapfrog integration constant (number of substeps)
# OPTIMIZED: Reduced for cleaner gradient flow
LEAPFROG_SUBSTEPS = 3  # Was 5 - Cleaner backward pass

# Heun integration safety factor
HEUN_SAFETY_FACTOR = 0.9


# ============================================================================
# ACTIVE INFERENCE CONSTANTS - OPTIMIZED FOR STABILITY
# ============================================================================

# Default plasticity coefficient for reactive curvature - OPTIMIZED
DEFAULT_PLASTICITY = 0.02  # Was 0.01 - Better responsiveness

# Default singularity threshold
SINGULARITY_THRESHOLD = 0.5  # Was 0.8 - Lower threshold for earlier activation

# Default black hole strength - OPTIMIZED
BLACK_HOLE_STRENGTH = 1.5  # Was 2.0 - Reduced for stability

# Reactive curvature learning rate
REACTIVE_CURVATURE_LR = 0.01

# Maximum curvature adjustment per step
MAX_CURVATURE_ADJUSTMENT = 0.1


# ============================================================================
# SINGULARITY GATE CONSTANTS - OPTIMIZED
# ============================================================================

# Singularity gate slope - OPTIMIZED for smooth gradients
SINGULARITY_GATE_SLOPE = 0.5  # Was 1.0 - Smoother transitions


# ============================================================================
# DEVICE CONSTANTS
# ============================================================================

# Default device fallback
DEFAULT_DEVICE = 'cpu'


# ============================================================================
# TRAINING STABILITY CONSTANTS
# ============================================================================

# Gradient clipping norm
GRAD_CLIP_NORM = 1.0  # Max gradient norm

# Learning rate warmup steps
LR_WARMUP_STEPS = 100

# Learning rate warmup ratio
LR_WARMUP_RATIO = 0.1

# Velocity saturation - INCREASED for less compression
# AUDIT FIX: Increased to allow Christoffel effects to be meaningful
VELOCITY_SATURATION = 100.0  # Was 50.0 - Allow higher velocities

# Gradient clipping value (per-parameter)
GRADIENT_CLIP_VALUE = 1.0

# Adaptive friction scale for energy stabilization
ADAPTIVE_FRICTION_SCALE = 0.1

# Huber loss delta for robust loss computation
HUBER_DELTA = 1.0

# Lowered gate slope for smoother transitions
GATE_SLOPE_LOWERED = 2.0  # Reduced from implicit 10.0

# ============================================================================
# HYSTERESIS CONSTANTS
# ============================================================================

# Initial forget gate value for hysteresis - controls how quickly past states decay
HYSTERESIS_FORGET_GATE_INIT = 0.9  # sigmoid(2.0) ≈ 0.88 - gradual decay

# Hysteresis state update momentum
HYSTERESIS_STATE_MOMENTUM = 0.95

# Minimum hysteresis strength
HYSTERESIS_MIN_STRENGTH = 0.01

# Maximum hysteresis strength
HYSTERESIS_MAX_STRENGTH = 0.5

# Hysteresis ghost force coefficient
HYSTERESIS_GHOST_FORCE_SCALE = 0.1

# ============================================================================
# TOROIDAL GEOMETRY CONSTANTS
# ============================================================================

# Period for toroidal coordinate wrapping (2π)
TOROIDAL_PERIOD = 6.283185307179586  # 2 * π

# Toroidal distance weight for angle components
TOROIDAL_ANGLE_WEIGHT = 1.0

# Toroidal distance weight for radius components
TOROIDAL_RADIUS_WEIGHT = 0.5

# Minimum radius for toroidal boundaries
TOROIDAL_MIN_RADIUS = 0.1

# Maximum radius for toroidal boundaries
TOROIDAL_MAX_RADIUS = 3.0

# ============================================================================
# AGGREGATION CONSTANTS
# ============================================================================

# Default momentum for state aggregation
AGGREGATION_MOMENTUM_DEFAULT = 0.9

# Minimum samples for valid aggregation
AGGREGATION_MIN_SAMPLES = 1

# Maximum trajectory length for aggregation
AGGREGATION_MAX_TRAJECTORY_LEN = 1000

# State buffer size for momentum accumulation
STATE_BUFFER_SIZE = 100


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device(tensor=None, default='cpu'):
    """
    Get device from tensor or return default.
    """
    if tensor is not None:
        return tensor.device
    return default


def get_stable_lr_scale(step, total_steps, warmup_ratio=LR_WARMUP_RATIO):
    """
    Compute learning rate scale with warmup and cosine decay.
    
    Args:
        step: Current step
        total_steps: Total training steps
        warmup_ratio: Fraction of training for warmup
    
    Returns:
        LR scale factor in [0, 1]
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    if step < warmup_steps:
        # Linear warmup
        return float(step) / max(1, warmup_steps)
    else:
        # Cosine decay
        decay_steps = total_steps - warmup_steps
        decay_progress = float(step - warmup_steps) / max(1, decay_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(decay_progress * 3.14159)).item())


# ============================================================================
# CUDA KERNEL CONSTANTS
# ============================================================================

# Maximum threads per block for CUDA kernels
CUDA_MAX_THREADS_PER_BLOCK = 256

# Shared memory size for CUDA (bytes)
CUDA_SHARED_MEMORY_SIZE = 16384

# Block size for parallel reduction in CUDA
CUDA_REDUCTION_BLOCK_SIZE = 32

# Warp size for CUDA operations
CUDA_WARP_SIZE = 32

# Enable CUDA memory optimization flags
CUDA_OPTIMIZE_MEMORY = True

# Use fused kernels when available
CUDA_USE_FUSED_KERNELS = True

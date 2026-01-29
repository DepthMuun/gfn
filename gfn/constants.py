"""
GFN Configuration Constants
============================

Centralized constants for the GFN codebase to improve maintainability
and make hyperparameter tuning easier.
"""

# ============================================================================
# EMBEDDING CONSTANTS
# ============================================================================

# Implicit embedding output scale
# Boosts implicit embeddings for better gradient flow
EMBEDDING_SCALE = 1.5

# Functional embedding impulse scale
# Controls the strength of token forces in functional embeddings
IMPULSE_SCALE = 1.0

# SIREN omega_0 frequency
# Higher values allow fitting higher-frequency signals
SIREN_OMEGA_0 = 30.0


# ============================================================================
# READOUT CONSTANTS
# ============================================================================

# Readout logit gain
# Sharpens logits for better BCE loss gradients
READOUT_GAIN = 10.0


# ============================================================================
# GEOMETRY CONSTANTS
# ============================================================================

# Curvature clamping limit
# Prevents extreme curvature values that can cause instability
CURVATURE_CLAMP = 20.0

# Toroidal curvature scale
# Controls the strength of toroidal geometry effects
TOROIDAL_CURVATURE_SCALE = 0.05

# Default major radius for toroidal manifolds
TOROIDAL_MAJOR_RADIUS = 2.0

# Default minor radius for toroidal manifolds
TOROIDAL_MINOR_RADIUS = 1.0


# ============================================================================
# FRICTION CONSTANTS
# ============================================================================

# Friction gate scale
# Maximum friction coefficient (multiplied by sigmoid output)
FRICTION_SCALE = 5.0

# Default friction coefficient for conformal symplectic systems
DEFAULT_FRICTION = 0.05


# ============================================================================
# NUMERICAL STABILITY CONSTANTS
# ============================================================================

# Epsilon for division safety (strong protection)
EPSILON_STRONG = 1e-4

# Epsilon for division safety (standard protection)
EPSILON_STANDARD = 1e-6

# Epsilon for gradient smoothing
EPSILON_SMOOTH = 1e-8

# Minimum clamping value for denominators
CLAMP_MIN_STRONG = 1e-4

# Standard clamping minimum
CLAMP_MIN_STANDARD = 1e-6


# ============================================================================
# LOSS FUNCTION CONSTANTS
# ============================================================================

# Default Hamiltonian loss weight
LAMBDA_H_DEFAULT = 0.01

# Default geodesic regularization weight
LAMBDA_G_DEFAULT = 0.001

# Default Noether symmetry loss weight
LAMBDA_N_DEFAULT = 0.0

# Default kinetic energy penalty weight
LAMBDA_K_DEFAULT = 0.001

# Heuristic scaling for fused geodesic regularization
GEODESIC_FUSED_SCALE = 1000.0


# ============================================================================
# OPTIMIZER CONSTANTS
# ============================================================================

# Default learning rate for RiemannianAdam
DEFAULT_LR = 1e-3

# Default beta1 for Adam
ADAM_BETA1 = 0.9

# Default beta2 for Adam
ADAM_BETA2 = 0.999

# Default epsilon for Adam
ADAM_EPSILON = 1e-8

# Default weight decay
DEFAULT_WEIGHT_DECAY = 0.01

# Maximum weight norm for retraction
MAX_WEIGHT_NORM = 10.0


# ============================================================================
# INITIALIZATION CONSTANTS
# ============================================================================

# Standard deviation for normal initialization
INIT_STD = 0.02

# Initial position state scale
INIT_X0_SCALE = 0.02

# Initial velocity state scale
INIT_V0_SCALE = 0.01

# Gate bias initialization (open state)
GATE_BIAS_OPEN = 2.0  # sigmoid(2.0) ≈ 0.88

# Gate bias initialization (closed state)
GATE_BIAS_CLOSED = -5.0  # sigmoid(-5.0) ≈ 0.007


# ============================================================================
# INTEGRATION CONSTANTS
# ============================================================================

# Default timestep for integrators
DEFAULT_DT = 0.1

# Minimum sequence length for parallel scan
PARALLEL_SCAN_THRESHOLD = 32


# ============================================================================
# ACTIVE INFERENCE CONSTANTS
# ============================================================================

# Default plasticity coefficient for reactive curvature
DEFAULT_PLASTICITY = 0.1

# Default singularity threshold
SINGULARITY_THRESHOLD = 0.8

# Default black hole strength
BLACK_HOLE_STRENGTH = 10.0


# ============================================================================
# DEVICE CONSTANTS
# ============================================================================

# Default device fallback
DEFAULT_DEVICE = 'cpu'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device(tensor=None, default='cpu'):
    """
    Get device from tensor or return default.
    
    Args:
        tensor: Optional tensor to get device from
        default: Default device if tensor is None
        
    Returns:
        torch.device
    """
    if tensor is not None:
        return tensor.device
    return default

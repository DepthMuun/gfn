# constants.py — GFN V5
# Constantes físicas y matemáticas universales.
# NO contiene hiperparámetros de entrenamiento ni valores configurables.

import torch

# ─── Constantes Matemáticas ─────────────────────────────────────────────────
PI = 3.14159265358979
E = 2.718281828459045
SQRT_2 = 1.4142135623730951
LOG_2 = 0.6931471805599453

# ─── Estabilidad Numérica ─────────────────────────────────────────────────
EPS = 1e-8
INF = 1e12
EPSILON_STANDARD = 1e-7
EPSILON_SMOOTH = 1e-9
EPSILON_STRONG = 1e-6
CLAMP_MIN_STRONG = 1e-4

# ─── Límites Físicos ───────────────────────────────────────────────────────
MIN_DT = 0.001
MAX_DT = 1.0

# ─── Geometría / Curvatura ─────────────────────────────────────────────────
CURVATURE_CLAMP = 5.0  # Maximum absolute value of Christoffel output
FRICTION_SCALE = 0.1   # Global friction scaling factor
VELOCITY_FRICTION_SCALE = 0.01

# ─── Gate initialization constants ─────────────────────────────────────────
GATE_BIAS_OPEN   = 2.0   # sigmoid(2.0)  ≈ 0.88
GATE_BIAS_CLOSED = -2.0  # sigmoid(-2.0) ≈ 0.12

# ─── Singularity / Active Inference ───────────────────────────────────────
SINGULARITY_THRESHOLD = 0.5
BLACK_HOLE_STRENGTH = 3.0
SINGULARITY_GATE_SLOPE = 10.0

# ─── Torus geometry ───────────────────────────────────────────────────────
TOROIDAL_MAJOR_RADIUS   = 1.0
TOROIDAL_MINOR_RADIUS   = 0.3
TOROIDAL_PERIOD         = 2.0 * PI
TOROIDAL_CURVATURE_SCALE = 0.1

# ─── Tipo de dato por defecto ─────────────────────────────────────────────
DTYPE = torch.float32

# ─── Topology Names ───────────────────────────────────────────────────────
TOPOLOGY_TORUS      = "torus"
TOPOLOGY_SPHERE     = "spherical"
TOPOLOGY_HYPERBOLIC = "hyperbolic"
TOPOLOGY_EUCLIDEAN  = "euclidean"

# ─── Dynamics Modes ───────────────────────────────────────────────────────
DYNAMICS_DIRECT     = "direct"
DYNAMICS_RESIDUAL   = "residual"
DYNAMICS_MIX        = "mix"
DYNAMICS_GATED      = "gated"
DYNAMICS_STOCHASTIC = "stochastic"

# ─── Alias de compatibilidad (valores por defecto que moved a defaults.py) ─
# NOTA: Estos valores se mantienen aquí por compatibilidad pero deberían
# imports desde config/defaults.py en código nuevo
DEFAULT_FRICTION = 0.01
DEFAULT_DT = 0.1
DEFAULT_PLASTICITY = 0.05
MAX_VELOCITY = 10.0

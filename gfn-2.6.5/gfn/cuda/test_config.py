"""
Test Configuration for CUDA Test Suite
Centralized configuration for all CUDA tests.
"""

import torch

# Device and precision
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64  # Double precision for exact numerical agreement

# Tolerances for numerical comparison
RTOL = 1e-12  # Relative tolerance
ATOL = 1e-13  # Absolute tolerance

# Gradient check tolerances
GRAD_EPS = 1e-6    # Step size for numerical differentiation
GRAD_ATOL = 1e-5   # Absolute tolerance for gradient check
GRAD_RTOL = 1e-4   # Relative tolerance for gradient check

# Test dimensions
BATCH_SIZES = [1, 4, 16, 64]
DIMENSIONS = [8, 16, 32, 64]
RANKS = [2, 4, 8, 16]

# Integration parameters
DT_VALUES = [0.1, 0.05, 0.025, 0.0125]  # For convergence tests
INTEGRATION_STEPS = [1, 5, 10, 50]

# Topology types
TOPOLOGY_EUCLIDEAN = 0
TOPOLOGY_TORUS = 1

# Physics parameters
DEFAULT_PLASTICITY = 0.1
DEFAULT_SING_THRESH = 0.8
DEFAULT_SING_STRENGTH = 10.0
DEFAULT_R = 2.0  # Toroidal major radius
DEFAULT_r = 1.0  # Toroidal minor radius

# Random seed for reproducibility
RANDOM_SEED = 42

# Performance benchmark settings
BENCHMARK_WARMUP = 10
BENCHMARK_ITERATIONS = 100

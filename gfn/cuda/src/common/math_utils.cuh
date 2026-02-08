#ifndef GFN_CUDA_MATH_UTILS_CUH
#define GFN_CUDA_MATH_UTILS_CUH

#include "types.cuh"
#include "device_utils.cuh"
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Matrix-Vector Operations
// ============================================================================

/**
 * Matrix-vector multiplication: y = A * x
 * A is [m x n], x is [n], y is [m]
 */
GFN_DEVICE void matvec(
    scalar_t* y,
    const scalar_t* A,
    const scalar_t* x,
    int m,
    int n
) {
    for (int i = 0; i < m; ++i) {
        scalar_t sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Transposed matrix-vector multiplication: y = A^T * x
 * A is [m x n], x is [m], y is [n]
 */
GFN_DEVICE void matvec_transpose(
    scalar_t* y,
    const scalar_t* A,
    const scalar_t* x,
    int m,
    int n
) {
    for (int j = 0; j < n; ++j) {
        scalar_t sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + j] * x[i];
        }
        y[j] = sum;
    }
}

/**
 * Outer product: C = a ⊗ b (element-wise)
 * Result: C[i] = a[i] * b[i]
 */
GFN_DEVICE void outer_product_elementwise(
    scalar_t* C,
    const scalar_t* a,
    const scalar_t* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        C[i] = a[i] * b[i];
    }
}

// ============================================================================
// Trigonometric Helpers
// ============================================================================

/**
 * Compute sin and cos simultaneously (more efficient).
 */
GFN_DEVICE void sincos_scalar(scalar_t x, scalar_t* s, scalar_t* c) {
    #ifdef __CUDA_ARCH__
      sincos(x, s, c);  // Double precision version
    #else
      *s = sin(x);
      *c = cos(x);
    #endif
}

/**
 * Compute Fourier features for toroidal geometry: [sin(x), cos(x)]
 */
GFN_DEVICE void compute_fourier_features(
    scalar_t* features,
    const scalar_t* x,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        scalar_t s, c;
        sincos_scalar(x[i], &s, &c);
        features[i] = s;
        features[dim + i] = c;
    }
}

// ============================================================================
// Numerical Stability Helpers
// ============================================================================

/**
 * Stable computation of 1 / (1 + sqrt(x))
 */
GFN_DEVICE scalar_t stable_inv_sqrt_plus_one(scalar_t x) {
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    return 1.0 / (1.0 + sqrt(x + EPSILON_STANDARD));
}

/**
 * Stable computation of exp(-x) for large x
 */
GFN_DEVICE scalar_t stable_exp_neg(scalar_t x) {
    return exp(-fmin(x, 50.0)); // Clamp to prevent underflow
}

/**
 * Log-sum-exp trick for numerical stability
 */
GFN_DEVICE scalar_t log_sum_exp(const scalar_t* values, int n) {
    scalar_t max_val = values[0];
    for (int i = 1; i < n; ++i) {
        max_val = fmax(max_val, values[i]);
    }
    
    scalar_t sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += exp(values[i] - max_val);
    }
    
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    return max_val + log(sum + EPSILON_STANDARD);
}

// ============================================================================
// Energy and Hamiltonian Helpers
// ============================================================================

/**
 * Compute kinetic energy: E_k = 0.5 * ||v||^2
 */
GFN_DEVICE scalar_t kinetic_energy(const scalar_t* v, int dim) {
    scalar_t energy = 0.0f;
    for (int i = 0; i < dim; ++i) {
        energy += v[i] * v[i];
    }
    return 0.5f * energy;
}

/**
 * Compute potential energy for toroidal geometry
 */
GFN_DEVICE scalar_t toroidal_potential(
    const scalar_t* x,
    int dim,
    scalar_t R,
    scalar_t r
) {
    scalar_t potential = 0.0f;
    for (int i = 0; i < dim - 1; i += 2) {
        scalar_t theta = x[i];
        scalar_t cos_theta = cos(theta);
        // Potential from metric variation
        potential += (R + r * cos_theta) * (R + r * cos_theta);
    }
    return potential;
}

// ============================================================================
// Interpolation and Smoothing
// ============================================================================

/**
 * Linear interpolation
 */
GFN_DEVICE scalar_t lerp(scalar_t a, scalar_t b, scalar_t t) {
    return a + t * (b - a);
}

/**
 * Smooth step function (cubic Hermite)
 */
GFN_DEVICE scalar_t smoothstep(scalar_t edge0, scalar_t edge1, scalar_t x) {
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    scalar_t t = clamp_value((x - edge0) / (edge1 - edge0 + EPSILON_STANDARD), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/**
 * Smoother step function (quintic)
 */
GFN_DEVICE scalar_t smootherstep(scalar_t edge0, scalar_t edge1, scalar_t x) {
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    scalar_t t = clamp_value((x - edge0) / (edge1 - edge0 + EPSILON_STANDARD), 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// ============================================================================
// Random Number Generation (for testing)
// ============================================================================

/**
 * Simple LCG random number generator (for testing only)
 */
GFN_DEVICE scalar_t random_uniform(unsigned int* seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return static_cast<scalar_t>(*seed) / static_cast<scalar_t>(0x7fffffff);
}

/**
 * Box-Muller transform for Gaussian random numbers
 */
GFN_DEVICE scalar_t random_gaussian(unsigned int* seed) {
    scalar_t u1 = random_uniform(seed);
    scalar_t u2 = random_uniform(seed);
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    return sqrt(-2.0 * log(u1 + EPSILON_STANDARD)) * cos(TWO_PI * u2);
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_MATH_UTILS_CUH

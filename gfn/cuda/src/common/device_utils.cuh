#ifndef GFN_CUDA_DEVICE_UTILS_CUH
#define GFN_CUDA_DEVICE_UTILS_CUH

#include "types.cuh"
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Boundary Conditions
// ============================================================================

/**
 * Apply periodic boundary conditions for toroidal topology.
 * Maps coordinates to [-π, π] range.
 */
GFN_DEVICE scalar_t apply_boundary_device(scalar_t x, Topology topology) {
    if (topology == Topology::TORUS) {
        // Map to [0, 2π]
        scalar_t wrapped = fmodf(x, TWO_PI);
        if (wrapped < 0.0f) wrapped += TWO_PI;
        return wrapped;
    }
    return x;
}

/**
 * Apply boundary conditions to entire vector.
 */
GFN_DEVICE void apply_boundary_vector(
    scalar_t* x,
    int dim,
    Topology topology
) {
    if (topology == Topology::TORUS) {
        for (int i = 0; i < dim; ++i) {
            x[i] = apply_boundary_device(x[i], topology);
        }
    }
}

// ============================================================================
// Safe Mathematical Operations
// ============================================================================

/**
 * Safe division with epsilon protection.
 */
GFN_DEVICE scalar_t safe_divide(
    scalar_t numerator,
    scalar_t denominator,
    scalar_t epsilon = EPSILON_STRONG
) {
    return numerator / (denominator + epsilon);
}

/**
 * Clamp value to range with configurable limits.
 */
GFN_DEVICE scalar_t clamp_value(
    scalar_t value,
    scalar_t min_val = CURVATURE_CLAMP_MIN,
    scalar_t max_val = CURVATURE_CLAMP
) {
    return fmin(fmax(value, min_val), max_val);
}

/**
 * Soft clamp using tanh.
 */
GFN_DEVICE scalar_t soft_clamp(
    scalar_t value,
    scalar_t scale = CURVATURE_CLAMP
) {
    return scale * tanh(value / scale);
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Dot product of two vectors.
 */
GFN_DEVICE scalar_t dot_product(
    const scalar_t* a,
    const scalar_t* b,
    int dim
) {
    scalar_t result = 0.0f;
    for (int i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Compute L2 norm of a vector with numerical stability
 * @param v Input vector
 * @param dim Dimension of vector
 * @return ||v||_2
 */
GFN_DEVICE scalar_t norm(const scalar_t* v, int dim) {
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    return sqrt(dot_product(v, v, dim) + EPSILON_STANDARD);
}

/**
 * Vector addition: c = a + b
 */
GFN_DEVICE void vector_add(
    scalar_t* c,
    const scalar_t* a,
    const scalar_t* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Scaled vector addition: c = a + scale * b
 */
GFN_DEVICE void vector_add_scaled(
    scalar_t* c,
    const scalar_t* a,
    scalar_t scale,
    const scalar_t* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        c[i] = a[i] + scale * b[i];
    }
}

/**
 * Vector scaling: b = scale * a
 */
GFN_DEVICE void vector_scale(
    scalar_t* b,
    scalar_t scale,
    const scalar_t* a,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        b[i] = scale * a[i];
    }
}

/**
 * Copy vector: dst = src
 */
GFN_DEVICE void vector_copy(
    scalar_t* dst,
    const scalar_t* src,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        dst[i] = src[i];
    }
}

/**
 * Zero vector: v = 0
 */
GFN_DEVICE void vector_zero(
    scalar_t* v,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        v[i] = 0.0f;
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Sigmoid activation.
 */
GFN_DEVICE scalar_t sigmoid(scalar_t x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Tanh activation (already in CUDA, but for consistency).
 */
GFN_DEVICE scalar_t tanh_activation(scalar_t x) {
    return tanh(x);
}

// ============================================================================
// Warp-Level Primitives
// ============================================================================

/**
 * Warp-level reduction sum.
 */
GFN_DEVICE scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level reduction sum.
 */
GFN_DEVICE scalar_t block_reduce_sum(scalar_t val) {
    __shared__ scalar_t shared[WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_DEVICE_UTILS_CUH

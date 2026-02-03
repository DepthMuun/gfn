#ifndef GFN_CUDA_TYPES_CUH
#define GFN_CUDA_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace gfn {
namespace cuda {

// ============================================================================
// Scalar Types
// ============================================================================

using scalar_t = float;  // Use float precision to match PyTorch float32 and for better performance on consumer GPUs

// ============================================================================
// Topology Types
// ============================================================================

enum class Topology : int32_t {
    EUCLIDEAN = 0,
    TORUS = 1
};

// ============================================================================
// Physics Constants
// ============================================================================

// Curvature clamping
constexpr scalar_t CURVATURE_CLAMP = 20.0f;
constexpr scalar_t CURVATURE_CLAMP_MIN = -20.0f;

// Friction scaling
constexpr scalar_t FRICTION_SCALE = 5.0f;
constexpr scalar_t DEFAULT_FRICTION = 0.1f;

// Numerical stability epsilons
constexpr scalar_t EPSILON_WEAK = 1e-6f;
constexpr scalar_t EPSILON_STRONG = 1e-4f;
constexpr scalar_t EPSILON_SMOOTH = 1e-5f;

// Clamping for division safety
constexpr scalar_t CLAMP_MIN_WEAK = 1e-6f;
constexpr scalar_t CLAMP_MIN_STRONG = 0.1f;

// Gate biases
constexpr scalar_t GATE_BIAS_OPEN = 2.0f;   // sigmoid(2) ≈ 0.88 (mostly open)
constexpr scalar_t GATE_BIAS_CLOSED = -2.0f; // sigmoid(-2) ≈ 0.12 (mostly closed)

// Toroidal geometry
constexpr scalar_t TOROIDAL_MAJOR_RADIUS = 2.0f;  // R
constexpr scalar_t TOROIDAL_MINOR_RADIUS = 1.0f;  // r
constexpr scalar_t TOROIDAL_CURVATURE_SCALE = 1.0f;

// Active inference
constexpr scalar_t DEFAULT_PLASTICITY = 0.1f;
constexpr scalar_t SINGULARITY_THRESHOLD = 0.8f;
constexpr scalar_t BLACK_HOLE_STRENGTH = 10.0f;

// Mathematical constants
constexpr scalar_t PI = 3.14159265358979323846f;
constexpr scalar_t TWO_PI = 6.28318530717958647692f;

// ============================================================================
// Device Function Attributes
// ============================================================================

#define GFN_DEVICE __device__ __forceinline__
#define GFN_HOST_DEVICE __host__ __device__ __forceinline__
#define GFN_GLOBAL __global__

// ============================================================================
// Grid/Block Configuration
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int DEFAULT_BLOCK_SIZE = 256;

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

GFN_HOST_DEVICE int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

inline dim3 get_grid_size(int n, int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(div_ceil(n, block_size));
}

inline dim3 get_block_size(int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(block_size);
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_TYPES_CUH

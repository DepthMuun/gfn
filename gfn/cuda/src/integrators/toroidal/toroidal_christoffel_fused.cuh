/**
 * Toroidal Christof

fel Fused Kernel - Header
 * ========================================
 * 
 * Header file for toroidal-specific CUDA kernels.
 * Provides interface for Python bindings.
 */

#ifndef GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH
#define GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH

#include "../../common/types.cuh"
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

/**
 * @brief Launch toroidal leapfrog fused kernel
 * 
 * Performs full sequence integration using metric-derived
 * Christoffel symbols for toroidal topology.
 * 
 * @param x Initial positions [batch, dim]
 * @param v Initial velocities [batch, dim]
 * @param f Force sequence [batch, seq_len, dim]
 * @param R Major radius of torus
 * @param r Minor radius of torus
 * @param dt Time step
 * @param batch Batch size
 * @param seq_len Sequence length  
 * @param dim Dimension (should be even for angle pairs)
 * @param x_out Output positions [batch, seq_len, dim]
 * @param v_out Output velocities [batch, seq_len, dim]
 * @param stream CUDA stream (optional, default=0)
 */
void launch_toroidal_leapfrog_fused(
    const scalar_t* x,
    const scalar_t* v,
    const scalar_t* f,
    scalar_t R,
    scalar_t r,
    scalar_t dt,
    int batch,
    int seq_len,
    int dim,
    scalar_t* x_out,
    scalar_t* v_out,
    cudaStream_t stream = 0
);

/**
 * @brief Device function for toroidal Christoffel computation
 * 
 * Can be called from other kernels for composability.
 * 
 * @param x Position vector [dim]
 * @param v Velocity vector [dim]
 * @param dim Dimension
 * @param R Major radius
 * @param r Minor radius
 * @param gamma Output Christoffel force [dim]
 */
GFN_DEVICE void toroidal_christoffel_full(
    const scalar_t* x,
    const scalar_t* v,
    int dim,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma
);

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH

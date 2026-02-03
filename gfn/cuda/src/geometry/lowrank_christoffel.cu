#include "christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Low-Rank Christoffel Kernel
// ============================================================================

__global__ void lowrank_christoffel_kernel(
    const scalar_t* __restrict__ v,      // [batch, dim]
    const scalar_t* __restrict__ U,      // [dim, rank]
    const scalar_t* __restrict__ W,      // [dim, rank]
    const scalar_t* __restrict__ x,      // [batch, dim] or nullptr
    const scalar_t* __restrict__ V_w,    // [dim] or nullptr
    scalar_t* __restrict__ gamma,        // [batch, dim]
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    int topology_id,
    scalar_t R,
    scalar_t r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Get pointers for this batch element
    const scalar_t* v_ptr = v + idx * dim;
    const scalar_t* x_ptr = (x != nullptr) ? (x + idx * dim) : nullptr;
    scalar_t* gamma_ptr = gamma + idx * dim;
    
    Topology topology = static_cast<Topology>(topology_id);
    
    // Compute Christoffel force
    christoffel_device(
        v_ptr, U, W, x_ptr, V_w,
        dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r,
        gamma_ptr
    );
}

// ============================================================================
// Low-Rank Christoffel with Friction Kernel
// ============================================================================

__global__ void lowrank_christoffel_friction_kernel(
    const scalar_t* __restrict__ v,          // [batch, dim]
    const scalar_t* __restrict__ U,          // [dim, rank]
    const scalar_t* __restrict__ W,          // [dim, rank]
    const scalar_t* __restrict__ x,          // [batch, dim]
    const scalar_t* __restrict__ V_w,        // [dim] or nullptr
    const scalar_t* __restrict__ force,      // [batch, dim] or nullptr
    const scalar_t* __restrict__ W_forget,   // [dim, feature_dim]
    const scalar_t* __restrict__ b_forget,   // [dim]
    const scalar_t* __restrict__ W_input,    // [dim, dim] or nullptr
    scalar_t* __restrict__ output,           // [batch, dim]
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    int topology_id,
    scalar_t R,
    scalar_t r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Get pointers for this batch element
    const scalar_t* v_ptr = v + idx * dim;
    const scalar_t* x_ptr = x + idx * dim;
    const scalar_t* force_ptr = (force != nullptr) ? (force + idx * dim) : nullptr;
    scalar_t* output_ptr = output + idx * dim;
    
    Topology topology = static_cast<Topology>(topology_id);
    
    // Compute Christoffel + Friction
    christoffel_with_friction(
        v_ptr, U, W, x_ptr, V_w, force_ptr,
        W_forget, b_forget, W_input,
        dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r,
        output_ptr
    );
}

} // namespace cuda
} // namespace gfn

// ============================================================================
// PyTorch C++ Interface
// ============================================================================

using namespace gfn::cuda;

torch::Tensor lowrank_christoffel_fused(
    torch::Tensor v,           // [batch, dim]
    torch::Tensor U,           // [dim, rank]
    torch::Tensor W,           // [dim, rank]
    torch::Tensor x,           // [batch, dim] or empty
    torch::Tensor V_w,         // [dim] or empty
    float plasticity,
    float sing_thresh,
    float sing_strength,
    int topology,
    float R,
    float r
) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(U.is_cuda(), "U must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(v.dim() == 2, "v must be 2D [batch, dim]");
    TORCH_CHECK(U.dim() == 2, "U must be 2D [dim, rank]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [dim, rank]");
    
    int batch_size = v.size(0);
    int dim = v.size(1);
    int rank = U.size(1);
    
    TORCH_CHECK(U.size(0) == dim, "U dimension mismatch");
    TORCH_CHECK(W.size(0) == dim, "W dimension mismatch");
    TORCH_CHECK(W.size(1) == rank, "W rank mismatch");
    
    // Create output tensor
    auto gamma = torch::empty_like(v);
    
    // Prepare pointers
    const scalar_t* x_ptr = (x.numel() > 0) ? x.data_ptr<scalar_t>() : nullptr;
    const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
    
    // Launch kernel
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = div_ceil(batch_size, threads);
    
    lowrank_christoffel_kernel<<<blocks, threads>>>(
        v.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        x_ptr,
        V_w_ptr,
        gamma.data_ptr<scalar_t>(),
        batch_size,
        dim,
        rank,
        plasticity,
        sing_thresh,
        sing_strength,
        topology,
        R,
        r
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return gamma;
}

torch::Tensor lowrank_christoffel_with_friction(
    torch::Tensor v,           // [batch, dim]
    torch::Tensor U,           // [dim, rank]
    torch::Tensor W,           // [dim, rank]
    torch::Tensor x,           // [batch, dim]
    torch::Tensor V_w,         // [dim] or empty
    torch::Tensor force,       // [batch, dim] or empty
    torch::Tensor W_forget,    // [dim, feature_dim]
    torch::Tensor b_forget,    // [dim]
    torch::Tensor W_input,     // [dim, dim] or empty
    float plasticity,
    float sing_thresh,
    float sing_strength,
    int topology,
    float R,
    float r
) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    
    int batch_size = v.size(0);
    int dim = v.size(1);
    int rank = U.size(1);
    
    // Create output tensor
    auto output = torch::empty_like(v);
    
    // Prepare pointers
    const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
    const scalar_t* force_ptr = (force.numel() > 0) ? force.data_ptr<scalar_t>() : nullptr;
    const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
    
    // Launch kernel
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = div_ceil(batch_size, threads);
    
    lowrank_christoffel_friction_kernel<<<blocks, threads>>>(
        v.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        V_w_ptr,
        force_ptr,
        W_forget.data_ptr<scalar_t>(),
        b_forget.data_ptr<scalar_t>(),
        W_input_ptr,
        output.data_ptr<scalar_t>(),
        batch_size,
        dim,
        rank,
        plasticity,
        sing_thresh,
        sing_strength,
        topology,
        R,
        r
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return output;
}

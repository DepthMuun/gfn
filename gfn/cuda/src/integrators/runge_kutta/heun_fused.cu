#include "../geometry/christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Heun Integrator Kernel (RK2 - Predictor-Corrector)
// ============================================================================

__global__ void heun_fused_kernel(
    const scalar_t* __restrict__ x_in,       // [batch, dim]
    const scalar_t* __restrict__ v_in,       // [batch, dim]
    const scalar_t* __restrict__ force,      // [batch, dim]
    const scalar_t* __restrict__ U,          // [dim, rank]
    const scalar_t* __restrict__ W,          // [dim, rank]
    scalar_t* __restrict__ x_out,            // [batch, dim]
    scalar_t* __restrict__ v_out,            // [batch, dim]
    int batch_size,
    int dim,
    int rank,
    scalar_t dt,
    scalar_t dt_scale,
    int steps,
    int topology_id,
    scalar_t R,
    scalar_t r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Thread-local storage
    scalar_t curr_x[64];
    scalar_t curr_v[64];
    scalar_t x_pred[64];
    scalar_t v_pred[64];
    scalar_t acc1[64];
    scalar_t acc2[64];
    scalar_t gamma[64];
    
    // Load initial state
    const scalar_t* x_ptr = x_in + idx * dim;
    const scalar_t* v_ptr = v_in + idx * dim;
    const scalar_t* f_ptr = force + idx * dim;
    
    for (int i = 0; i < dim; ++i) {
        curr_x[i] = x_ptr[i];
        curr_v[i] = v_ptr[i];
    }
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    
    // Integration loop
    for (int step = 0; step < steps; ++step) {
        // Stage 1: Compute derivatives at current state
        // dx1/dt = v
        // dv1/dt = F - Γ(v, x)
        
        christoffel_device(
            curr_v, U, W, curr_x, nullptr,
            dim, rank, 0.0f, 1.0f, 1.0f,
            topology, R, r, gamma
        );
        
        for (int i = 0; i < dim; ++i) {
            acc1[i] = f_ptr[i] - gamma[i];
        }
        
        // Predictor step (Euler)
        for (int i = 0; i < dim; ++i) {
            x_pred[i] = curr_x[i] + effective_dt * curr_v[i];
            v_pred[i] = curr_v[i] + effective_dt * acc1[i];
        }
        
        // Apply boundary to predicted position
        apply_boundary_vector(x_pred, dim, topology);
        
        // Stage 2: Compute derivatives at predicted state
        christoffel_device(
            v_pred, U, W, x_pred, nullptr,
            dim, rank, 0.0f, 1.0f, 1.0f,
            topology, R, r, gamma
        );
        
        for (int i = 0; i < dim; ++i) {
            acc2[i] = f_ptr[i] - gamma[i];
        }
        
        // Corrector step (average of two slopes)
        for (int i = 0; i < dim; ++i) {
            curr_x[i] += (effective_dt / 2.0f) * (curr_v[i] + v_pred[i]);
            curr_v[i] += (effective_dt / 2.0f) * (acc1[i] + acc2[i]);
        }
        
        // Apply boundary conditions
        apply_boundary_vector(curr_x, dim, topology);
    }
    
    // Store final state
    scalar_t* x_out_ptr = x_out + idx * dim;
    scalar_t* v_out_ptr = v_out + idx * dim;
    
    for (int i = 0; i < dim; ++i) {
        x_out_ptr[i] = curr_x[i];
        v_out_ptr[i] = curr_v[i];
    }
}

} // namespace cuda
} // namespace gfn

// ============================================================================
// PyTorch C++ Interface
// ============================================================================

using namespace gfn::cuda;

std::vector<torch::Tensor> heun_fused(
    torch::Tensor x,           // [batch, dim]
    torch::Tensor v,           // [batch, dim]
    torch::Tensor force,       // [batch, dim]
    torch::Tensor U,           // [dim, rank]
    torch::Tensor W,           // [dim, rank]
    float dt,
    float dt_scale,
    int steps,
    int topology,
    float R,
    float r
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(force.is_cuda(), "force must be a CUDA tensor");
    
    int batch_size = x.size(0);
    int dim = x.size(1);
    int rank = U.size(1);
    
    // Create output tensors
    auto x_out = torch::empty_like(x);
    auto v_out = torch::empty_like(v);
    
    // Launch kernel
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = div_ceil(batch_size, threads);
    
    heun_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        force.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        x_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        batch_size,
        dim,
        rank,
        dt,
        dt_scale,
        steps,
        topology,
        R,
        r
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return {x_out, v_out};
}

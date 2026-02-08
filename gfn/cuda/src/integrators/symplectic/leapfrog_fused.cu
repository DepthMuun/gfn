#include "../geometry/christoffel_impl.cuh"
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Leapfrog Integrator Kernel (Kick-Drift-Kick with Implicit Friction)
// ============================================================================

__global__ void leapfrog_fused_kernel(
    const scalar_t* __restrict__ x_in,       // [batch, dim]
    const scalar_t* __restrict__ v_in,       // [batch, dim]
    const scalar_t* __restrict__ force,      // [batch, dim]
    const scalar_t* __restrict__ U,          // [dim, rank]
    const scalar_t* __restrict__ W,          // [dim, rank]
    const scalar_t* __restrict__ W_forget,   // [dim, feature_dim] or nullptr
    const scalar_t* __restrict__ b_forget,   // [dim] or nullptr
    scalar_t* __restrict__ x_out,            // [batch, dim]
    scalar_t* __restrict__ v_out,            // [batch, dim]
    int batch_size,
    int dim,
    int rank,
    scalar_t dt,
    scalar_t dt_scale,
    int steps,
    int topology_id,
    scalar_t plasticity,
    scalar_t R,
    scalar_t r,
    
    // Hysteresis parameters
    scalar_t* __restrict__ hysteresis_state,     // [batch, dim] (RW)
    const scalar_t* __restrict__ hyst_update_w,  // [dim, in_dim]
    const scalar_t* __restrict__ hyst_update_b,  // [dim]
    const scalar_t* __restrict__ hyst_readout_w, // [dim, dim]
    const scalar_t* __restrict__ hyst_readout_b, // [dim]
    scalar_t hyst_decay,
    bool hyst_enabled,
    int hyst_in_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Thread-local storage (increased to 256 for dim=128 support)
    scalar_t curr_x[256];
    scalar_t curr_v[256];
    scalar_t gamma[256];
    scalar_t friction[256];
    scalar_t f_ghost[256];
    scalar_t hyst_local[256];
    
    // Load initial state
    const scalar_t* x_ptr = x_in + idx * dim;
    const scalar_t* v_ptr = v_in + idx * dim;
    const scalar_t* f_ptr = force + idx * dim;
    
    // Load hysteresis state if enabled
    scalar_t* h_ptr = nullptr;
    if (hyst_enabled && hysteresis_state != nullptr) {
        h_ptr = hysteresis_state + idx * dim;
        for (int i = 0; i < dim; ++i) {
            hyst_local[i] = h_ptr[i];
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            hyst_local[i] = 0.0f;
        }
    }
    
    for (int i = 0; i < dim; ++i) {
        curr_x[i] = x_ptr[i];
        curr_v[i] = v_ptr[i];
    }
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    scalar_t h = 0.5f * effective_dt;
    
    // Integration loop
    for (int step = 0; step < steps; ++step) {
        // 0. Compute Ghost Force from Hysteresis
        if (hyst_enabled && hyst_readout_w != nullptr) {
            for (int i = 0; i < dim; ++i) {
                scalar_t sum = hyst_readout_b ? hyst_readout_b[i] : 0.0f;
                for (int j = 0; j < dim; ++j) {
                    sum += hyst_local[j] * hyst_readout_w[i * dim + j];
                }
                f_ghost[i] = sum;
            }
        } else {
            for (int i = 0; i < dim; ++i) f_ghost[i] = 0.0f;
        }
        
        // 1. Compute friction at current position
        if (W_forget != nullptr && b_forget != nullptr) {
            compute_friction(
                curr_x, f_ptr, W_forget, b_forget, nullptr,
                dim, topology, friction
            );
        } else {
            vector_zero(friction, dim);
        }
        
        // 2. Compute Christoffel force at current state
        christoffel_device(
            curr_v, U, W, curr_x, nullptr,
            dim, rank, plasticity, 1.0f, 1.0f,
            topology, R, r, gamma
        );
        
        // 3. KICK 1 (Half step velocity with implicit friction)
        // v_half = (v + h*(F + F_ghost - gamma)) / (1 + h*mu)
        for (int i = 0; i < dim; ++i) {
            scalar_t total_force = f_ptr[i] + f_ghost[i];
            scalar_t numerator = curr_v[i] + h * (total_force - gamma[i]);
            scalar_t denominator = 1.0f + h * friction[i];
            curr_v[i] = safe_divide(numerator, denominator, EPSILON_STANDARD);
        }
        
        // 4. DRIFT (Full step position)
        for (int i = 0; i < dim; ++i) {
            curr_x[i] += effective_dt * curr_v[i];
        }
        
        // 5. Apply boundary conditions
        apply_boundary_vector(curr_x, dim, topology);
        
        // 6. Compute friction at new position
        if (W_forget != nullptr && b_forget != nullptr) {
            compute_friction(
                curr_x, f_ptr, W_forget, b_forget, nullptr,
                dim, topology, friction
            );
        }
        
        // 7. Compute Christoffel force at new state
        christoffel_device(
            curr_v, U, W, curr_x, nullptr,
            dim, rank, plasticity, 1.0f, 1.0f,
            topology, R, r, gamma
        );
        
        // 8. KICK 2 (Half step velocity with implicit friction)
        for (int i = 0; i < dim; ++i) {
            scalar_t total_force = f_ptr[i] + f_ghost[i];
            scalar_t numerator = curr_v[i] + h * (total_force - gamma[i]);
            scalar_t denominator = 1.0f + h * friction[i];
            curr_v[i] = safe_divide(numerator, denominator, EPSILON_STANDARD);
        }
        
        // 9. Update Hysteresis State
        if (hyst_enabled && hyst_update_w != nullptr) {
            for (int i = 0; i < dim; ++i) {
                scalar_t sum = hyst_update_b ? hyst_update_b[i] : 0.0f;
                
                if (topology == Topology::TORUS) {
                    // Input: [sin(x), cos(x), v]
                    for (int j = 0; j < dim; ++j) {
                        sum += sinf(curr_x[j]) * hyst_update_w[i * hyst_in_dim + j];
                    }
                    for (int j = 0; j < dim; ++j) {
                        sum += cosf(curr_x[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];
                    }
                    for (int j = 0; j < dim; ++j) {
                        sum += curr_v[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];
                    }
                } else {
                    // Input: [x, v]
                    for (int j = 0; j < dim; ++j) {
                        sum += curr_x[j] * hyst_update_w[i * hyst_in_dim + j];
                    }
                    for (int j = 0; j < dim; ++j) {
                        sum += curr_v[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];
                    }
                }
                
                // Update: h = h * decay + tanh(sum)
                hyst_local[i] = hyst_local[i] * hyst_decay + tanhf(sum);
            }
        }
    }
    
    // Store final state
    scalar_t* x_out_ptr = x_out + idx * dim;
    scalar_t* v_out_ptr = v_out + idx * dim;
    
    for (int i = 0; i < dim; ++i) {
        x_out_ptr[i] = curr_x[i];
        v_out_ptr[i] = curr_v[i];
    }
    
    // Write back hysteresis state
    if (hyst_enabled && h_ptr != nullptr) {
        for (int i = 0; i < dim; ++i) {
            h_ptr[i] = hyst_local[i];
        }
    }
}

} // namespace cuda
} // namespace gfn

// ============================================================================
// PyTorch C++ Interface
// ============================================================================

using namespace gfn::cuda;

std::vector<at::Tensor> leapfrog_fused(
    at::Tensor x,           // [batch, dim]
    at::Tensor v,           // [batch, dim]
    at::Tensor force,       // [batch, dim]
    at::Tensor U,           // [dim, rank]
    at::Tensor W,           // [dim, rank]
    float dt,
    float dt_scale,
    int steps,
    int topology,
    at::Tensor W_forget,    // [dim, feature_dim] or empty
    at::Tensor b_forget,    // [dim] or empty
    float plasticity,
    float R,
    float r,
    
    // Hysteresis tensors
    at::Tensor hysteresis_state,    // [batch, dim] (RW)
    at::Tensor hyst_update_w,       // [dim, in_dim] or empty
    at::Tensor hyst_update_b,       // [dim] or empty
    at::Tensor hyst_readout_w,      // [dim, dim] or empty
    at::Tensor hyst_readout_b,      // [dim] or empty
    float hyst_decay,
    bool hyst_enabled
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(force.is_cuda(), "force must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, dim]");
    TORCH_CHECK(v.dim() == 2, "v must be 2D [batch, dim]");
    
    int batch_size = x.size(0);
    int dim = x.size(1);
    int rank = U.size(1);
    
    // Create output tensors
    auto x_out = at::empty_like(x);
    auto v_out = at::empty_like(v);
    
    // Prepare pointers
    const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;
    const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;
    
    // Prepare hysteresis pointers
    scalar_t* hyst_state_ptr = (hyst_enabled && hysteresis_state.numel() > 0) ? hysteresis_state.data_ptr<scalar_t>() : nullptr;
    const scalar_t* hyst_update_w_ptr = (hyst_enabled && hyst_update_w.numel() > 0) ? hyst_update_w.data_ptr<scalar_t>() : nullptr;
    const scalar_t* hyst_update_b_ptr = (hyst_enabled && hyst_update_b.numel() > 0) ? hyst_update_b.data_ptr<scalar_t>() : nullptr;
    const scalar_t* hyst_readout_w_ptr = (hyst_enabled && hyst_readout_w.numel() > 0) ? hyst_readout_w.data_ptr<scalar_t>() : nullptr;
    const scalar_t* hyst_readout_b_ptr = (hyst_enabled && hyst_readout_b.numel() > 0) ? hyst_readout_b.data_ptr<scalar_t>() : nullptr;
    
    // Determine in_dim based on topology
    int hyst_in_dim = (topology == 1) ? (3 * dim) : (2 * dim);
    
    // Launch kernel
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = div_ceil(batch_size, threads);
    
    leapfrog_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        force.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        W_forget_ptr,
        b_forget_ptr,
        x_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        batch_size,
        dim,
        rank,
        dt,
        dt_scale,
        steps,
        topology,
        plasticity,
        R,
        r,
        
        hyst_state_ptr,
        hyst_update_w_ptr,
        hyst_update_b_ptr,
        hyst_readout_w_ptr,
        hyst_readout_b_ptr,
        hyst_decay,
        hyst_enabled,
        hyst_in_dim
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return {x_out, v_out};
}

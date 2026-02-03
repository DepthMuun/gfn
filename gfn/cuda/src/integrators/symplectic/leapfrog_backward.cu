#include "../geometry/christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

/**
 * Adjoint Leapfrog Backward Kernel.
 * Computes gradients for the entire Kick-Drift-Kick trajectory.
 * Requires a trajectory workspace [batch, steps, 2, dim] for v and [batch, steps+1, dim] for x.
 */
__global__ void leapfrog_backward_kernel(
    const scalar_t* __restrict__ grad_x_out,    // [batch, dim]
    const scalar_t* __restrict__ grad_v_out,    // [batch, dim]
    const scalar_t* __restrict__ traj_x,        // [batch, steps+1, dim]
    const scalar_t* __restrict__ traj_v,        // [batch, steps, 2, dim]
    const scalar_t* __restrict__ force,         // [batch, dim]
    const scalar_t* __restrict__ U,             // [dim, rank]
    const scalar_t* __restrict__ W,             // [dim, rank]
    const scalar_t* __restrict__ W_forget,      // [dim, feature_dim]
    const scalar_t* __restrict__ b_forget,      // [dim]
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
    scalar_t* __restrict__ grad_x_in,           // [batch, dim]
    scalar_t* __restrict__ grad_v_in,           // [batch, dim]
    scalar_t* __restrict__ grad_force,          // [batch, dim]
    scalar_t* __restrict__ grad_U,              // [batch, dim, rank]
    scalar_t* __restrict__ grad_W,              // [batch, dim, rank]
    scalar_t* __restrict__ grad_W_forget,       // [batch, dim, feature_dim]
    scalar_t* __restrict__ grad_b_forget        // [batch, dim]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    scalar_t h = 0.5f * effective_dt;
    
    // Initial adjoints from output gradients
    scalar_t lx[64];
    scalar_t lv[64];
    for (int i = 0; i < dim; ++i) {
        lx[i] = grad_x_out[idx * dim + i];
        lv[i] = grad_v_out[idx * dim + i];
    }
    
    // Batch pointers for parameter gradients
    scalar_t* gU_b = grad_U + idx * dim * rank;
    scalar_t* gW_b = grad_W + idx * dim * rank;
    int f_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    scalar_t* gWf_b = grad_W_forget + idx * dim * f_dim;
    scalar_t* gbf_b = grad_b_forget + idx * dim;
    scalar_t* gf_b = grad_force + idx * dim;
    
    const scalar_t* f_ptr = force + idx * dim;
    
    // Re-initialize parameter gradients for this thread
    for (int i = 0; i < dim * rank; ++i) { gU_b[i] = 0; gW_b[i] = 0; }
    for (int i = 0; i < dim * f_dim; ++i) { gWf_b[i] = 0; }
    for (int i = 0; i < dim; ++i) { gbf_b[i] = 0; gf_b[i] = 0; }
    
    // --- ADJOINT LOOP BACKWARDS ---
    for (int step = steps - 1; step >= 0; --step) {
        // States from trajectory
        const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
        const scalar_t* v_n = traj_v + idx * steps * 2 * dim + (step * 2 + 0) * dim;
        const scalar_t* v_mid = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;
        const scalar_t* x_next = traj_x + idx * (steps + 1) * dim + (step + 1) * dim;
        
        // --- ADJOINT KICK 2 ---
        // Forward: v_next = (v_mid + h(F - gamma(v_mid, x_next))) / (1 + h*mu(x_next))
        scalar_t mu_next[64], gamma_mid[64];
        compute_friction(x_next, f_ptr, W_forget, b_forget, nullptr, dim, topology, mu_next);
        christoffel_device(v_mid, U, W, x_next, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gamma_mid);
        
        scalar_t l_v_mid[64], l_mu_next[64], l_gamma_mid[64];
        for (int i = 0; i < dim; ++i) {
            scalar_t den = 1.0f + h * mu_next[i];
            l_v_mid[i] = lv[i] / den;
            l_mu_next[i] = -h * lv[i] * ((v_mid[i] + h * (f_ptr[i] - gamma_mid[i])) / (den * den));
            l_gamma_mid[i] = -h * lv[i] / den;
            gf_b[i] += h * lv[i] / den;
        }
        
        // Adjoint of friction at x_next
        friction_backward_device(l_mu_next, x_next, f_ptr, W_forget, b_forget, nullptr, dim, topology, gWf_b, gbf_b, nullptr, lx, gf_b);
        
        // Adjoint of christoffel at v_mid, x_next
        scalar_t gv_c[64], gx_c[64];
        vector_zero(gv_c, dim);
        vector_zero(gx_c, dim);
        christoffel_backward_device(l_gamma_mid, gamma_mid, v_mid, U, W, x_next, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gv_c, gU_b, gW_b, gx_c);
        for (int i = 0; i < dim; ++i) {
            l_v_mid[i] += gv_c[i];
            lx[i] += gx_c[i];
        }
        
        // --- ADJOINT DRIFT ---
        // Forward: x_next = x_n + dt * v_mid
        for (int i = 0; i < dim; ++i) {
            l_v_mid[i] += effective_dt * lx[i];
        }
        
        // --- ADJOINT KICK 1 ---
        // Forward: v_mid = (v_n + h(F - gamma(v_n, x_n))) / (1 + h*mu(x_n))
        scalar_t mu_n[64], gamma_n[64];
        compute_friction(x_n, f_ptr, W_forget, b_forget, nullptr, dim, topology, mu_n);
        christoffel_device(v_n, U, W, x_n, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gamma_n);
        
        scalar_t l_v_n[64], l_mu_n[64], l_gamma_n[64];
        for (int i = 0; i < dim; ++i) {
            scalar_t den = 1.0f + h * mu_n[i];
            l_v_n[i] = l_v_mid[i] / den;
            l_mu_n[i] = -h * l_v_mid[i] * ((v_n[i] + h * (f_ptr[i] - gamma_n[i])) / (den * den));
            l_gamma_n[i] = -h * l_v_mid[i] / den;
            gf_b[i] += h * l_v_mid[i] / den;
        }
        
        // Adjoint of friction at x_n
        friction_backward_device(l_mu_n, x_n, f_ptr, W_forget, b_forget, nullptr, dim, topology, gWf_b, gbf_b, nullptr, lx, gf_b);
        
        // Adjoint of christoffel at v_n, x_n
        vector_zero(gv_c, dim);
        vector_zero(gx_c, dim);
        christoffel_backward_device(l_gamma_n, gamma_n, v_n, U, W, x_n, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gv_c, gU_b, gW_b, gx_c);
        for (int i = 0; i < dim; ++i) {
            lv[i] = l_v_n[i] + gv_c[i];
            lx[i] += gx_c[i];
        }
    }
    
    // Store final state gradients
    for (int i = 0; i < dim; ++i) {
        grad_x_in[idx * dim + i] = lx[i];
        grad_v_in[idx * dim + i] = lv[i];
    }
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

// Helper kernel for trajectory re-computation
__global__ void leapfrog_forward_traj_kernel(
    const scalar_t* x_in, const scalar_t* v_in, const scalar_t* force,
    const scalar_t* U, const scalar_t* W, const scalar_t* W_forget, const scalar_t* b_forget,
    int batch_size, int dim, int rank, scalar_t dt, scalar_t dt_scale, int steps,
    int topology_id, scalar_t plasticity, scalar_t R, scalar_t r,
    scalar_t* traj_x, scalar_t* traj_v
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t cx[64], cv[64], friction[64], gamma[64];
    for (int i = 0; i < dim; ++i) { cx[i] = x_in[idx * dim + i]; cv[i] = v_in[idx * dim + i]; }
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    scalar_t h = 0.5f * effective_dt;
    const scalar_t* f_ptr = force + idx * dim;

    for (int step = 0; step < steps; ++step) {
        // Store current x, v
        for (int i = 0; i < dim; ++i) {
            traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
            traj_v[idx * steps * 2 * dim + (step * 2 + 0) * dim + i] = cv[i];
        }
        
        compute_friction(cx, f_ptr, W_forget, b_forget, nullptr, dim, topology, friction);
        christoffel_device(cv, U, W, cx, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gamma);
        for (int i = 0; i < dim; ++i) {
            cv[i] = (cv[i] + h * (f_ptr[i] - gamma[i])) / (1.0f + h * friction[i]);
            traj_v[idx * steps * 2 * dim + (step * 2 + 1) * dim + i] = cv[i]; // Store v_mid
        }
        
        for (int i = 0; i < dim; ++i) { cx[i] += effective_dt * cv[i]; }
        apply_boundary_vector(cx, dim, topology);
        
        compute_friction(cx, f_ptr, W_forget, b_forget, nullptr, dim, topology, friction);
        christoffel_device(cv, U, W, cx, nullptr, dim, rank, plasticity, 1.0, 1.0, topology, R, r, gamma);
        for (int i = 0; i < dim; ++i) {
            cv[i] = (cv[i] + h * (f_ptr[i] - gamma[i])) / (1.0f + h * friction[i]);
        }
    }
    // Store final x
    for (int i = 0; i < dim; ++i) traj_x[idx * (steps + 1) * dim + steps * dim + i] = cx[i];
}

// C++ Wrapper
std::vector<torch::Tensor> leapfrog_backward_cuda(
    torch::Tensor grad_x_out, torch::Tensor grad_v_out,
    torch::Tensor x_in, torch::Tensor v_in, torch::Tensor force,
    torch::Tensor U, torch::Tensor W, torch::Tensor W_forget, torch::Tensor b_forget,
    float dt, float dt_scale, int steps, int topology,
    float plasticity, float R, float r
) {
    int batch_size = x_in.size(0);
    int dim = x_in.size(1);
    int rank = U.size(1);
    int f_dim = (topology == 1) ? 2 * dim : dim;

    auto options = x_in.options();
    auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
    auto traj_v = torch::empty({batch_size, steps, 2, dim}, options); // Stores v_n and v_mid

    auto grad_x_in = torch::zeros_like(x_in);
    auto grad_v_in = torch::zeros_like(v_in);
    auto grad_force = torch::zeros_like(force);
    auto grad_U = torch::zeros({batch_size, dim, rank}, options);
    auto grad_W = torch::zeros({batch_size, dim, rank}, options);
    auto grad_W_forget = torch::zeros({batch_size, dim, f_dim}, options);
    auto grad_b_forget = torch::zeros({batch_size, dim}, options);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // 1. Re-compute trajectory
    leapfrog_forward_traj_kernel<<<blocks, threads>>>(
        x_in.data_ptr<scalar_t>(), v_in.data_ptr<scalar_t>(), force.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
        W_forget.numel() > 0 ? W_forget.data_ptr<scalar_t>() : nullptr,
        b_forget.numel() > 0 ? b_forget.data_ptr<scalar_t>() : nullptr,
        batch_size, dim, rank, dt, dt_scale, steps, topology, plasticity, R, r,
        traj_x.data_ptr<scalar_t>(), traj_v.data_ptr<scalar_t>()
    );

    // 2. Compute Adjoint Gradients
    leapfrog_backward_kernel<<<blocks, threads>>>(
        grad_x_out.data_ptr<scalar_t>(), grad_v_out.data_ptr<scalar_t>(),
        traj_x.data_ptr<scalar_t>(), traj_v.data_ptr<scalar_t>(),
        force.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
        W_forget.numel() > 0 ? W_forget.data_ptr<scalar_t>() : nullptr,
        b_forget.numel() > 0 ? b_forget.data_ptr<scalar_t>() : nullptr,
        batch_size, dim, rank, dt, dt_scale, steps, topology, plasticity, R, r,
        grad_x_in.data_ptr<scalar_t>(), grad_v_in.data_ptr<scalar_t>(), grad_force.data_ptr<scalar_t>(),
        grad_U.data_ptr<scalar_t>(), grad_W.data_ptr<scalar_t>(),
        grad_W_forget.data_ptr<scalar_t>(), grad_b_forget.data_ptr<scalar_t>()
    );

    return {grad_x_in, grad_v_in, grad_force, grad_U.sum(0), grad_W.sum(0), grad_W_forget.sum(0), grad_b_forget.sum(0)};
}

#include "../geometry/christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

/**
 * Adjoint Heun Backward Kernel.
 * Computes gradients for the predictor-corrector (RK2) trajectory.
 */
__global__ void heun_backward_kernel(
    const scalar_t* __restrict__ grad_x_out,    // [batch, dim]
    const scalar_t* __restrict__ grad_v_out,    // [batch, dim]
    const scalar_t* __restrict__ traj_x,        // [batch, steps+1, dim]
    const scalar_t* __restrict__ traj_v,        // [batch, steps+1, dim]
    const scalar_t* __restrict__ traj_acc1,     // [batch, steps, dim] (accelerations at stage 1)
    const scalar_t* __restrict__ force,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ W,
    int batch_size,
    int dim,
    int rank,
    scalar_t dt,
    scalar_t dt_scale,
    int steps,
    int topology_id,
    scalar_t R,
    scalar_t r,
    scalar_t* __restrict__ grad_x_in,
    scalar_t* __restrict__ grad_v_in,
    scalar_t* __restrict__ grad_force,
    scalar_t* __restrict__ grad_U,
    scalar_t* __restrict__ grad_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    scalar_t h_half = 0.5f * effective_dt;
    
    scalar_t lx[64];
    scalar_t lv[64];
    for (int i = 0; i < dim; ++i) {
        lx[i] = grad_x_out[idx * dim + i];
        lv[i] = grad_v_out[idx * dim + i];
    }
    
    scalar_t* gU_b = grad_U + idx * dim * rank;
    scalar_t* gW_b = grad_W + idx * dim * rank;
    scalar_t* gf_b = grad_force + idx * dim;

    for (int i = 0; i < dim * rank; ++i) { gU_b[i] = 0; gW_b[i] = 0; }
    for (int i = 0; i < dim; ++i) { gf_b[i] = 0; }
    
    for (int step = steps - 1; step >= 0; --step) {
        const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
        const scalar_t* v_n = traj_v + idx * (steps + 1) * dim + step * dim;
        const scalar_t* acc1 = traj_acc1 + idx * steps * dim + step * dim;
        
        // RE-COMPUTE PREDICTOR (needed for adjoint)
        scalar_t x_pred[64], v_pred[64];
        for (int i = 0; i < dim; ++i) {
            x_pred[i] = x_n[i] + effective_dt * v_n[i];
            v_pred[i] = v_n[i] + effective_dt * acc1[i];
        }
        apply_boundary_vector(x_pred, dim, topology);
        
        // --- ADJOINT OF CORRECTOR ---
        // Forward: x_next = x_n + h_half * (v_n + v_pred)
        // Forward: v_next = v_n + h_half * (acc1 + acc2)
        scalar_t l_acc1[64], l_acc2[64], l_v_pred[64], l_v_n[64], l_x_n[64];
        for (int i = 0; i < dim; ++i) {
            l_v_n[i] = lx[i] * h_half + lv[i];
            l_v_pred[i] = lx[i] * h_half;
            l_acc1[i] = lv[i] * h_half;
            l_acc2[i] = lv[i] * h_half;
            l_x_n[i] = lx[i];
        }
        
        // --- ADJOINT OF STAGE 2 (acc2 at x_pred, v_pred) ---
        scalar_t gamma2[64];
        christoffel_device(v_pred, U, W, x_pred, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gamma2);
        
        scalar_t l_gamma2[64];
        for (int i = 0; i < dim; ++i) {
            l_gamma2[i] = -l_acc2[i]; // acc2 = F - gamma2
            gf_b[i] += l_acc2[i];
        }
        
        scalar_t gv_c[64], gx_c[64];
        vector_zero(gv_c, dim);
        vector_zero(gx_c, dim);
        christoffel_backward_device(l_gamma2, gamma2, v_pred, U, W, x_pred, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gv_c, gU_b, gW_b, gx_c);
        for (int i = 0; i < dim; ++i) {
            l_v_pred[i] += gv_c[i];
            // Propagate gx_c through predictor step to lx? No, x_pred is x_n + dt*v_n
            // So its adjoint contributes to l_x_n and l_v_n.
        }
        
        // --- ADJOINT OF PREDICTOR ---
        // Forward: x_pred = x_n + dt * v_n
        // Forward: v_pred = v_n + dt * acc1
        for (int i = 0; i < dim; ++i) {
            l_x_n[i] += gx_c[i]; // from S2 through x_pred
            l_v_n[i] += l_v_pred[i] + effective_dt * gx_c[i];
            l_acc1[i] += l_v_pred[i] * effective_dt;
        }
        
        // --- ADJOINT OF STAGE 1 (acc1 at x_n, v_n) ---
        scalar_t gamma1[64];
        christoffel_device(v_n, U, W, x_n, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gamma1);
        
        scalar_t l_gamma1[64];
        for (int i = 0; i < dim; ++i) {
            l_gamma1[i] = -l_acc1[i];
            gf_b[i] += l_acc1[i];
        }
        
        vector_zero(gv_c, dim);
        vector_zero(gx_c, dim);
        christoffel_backward_device(l_gamma1, gamma1, v_n, U, W, x_n, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gv_c, gU_b, gW_b, gx_c);
        for (int i = 0; i < dim; ++i) {
            lx[i] = l_x_n[i] + gx_c[i];
            lv[i] = l_v_n[i] + gv_c[i];
        }
    }
    
    for (int i = 0; i < dim; ++i) {
        grad_x_in[idx * dim + i] = lx[i];
        grad_v_in[idx * dim + i] = lv[i];
    }
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

__global__ void heun_forward_traj_kernel(
    const scalar_t* x_in, const scalar_t* v_in, const scalar_t* force,
    const scalar_t* U, const scalar_t* W, 
    int batch_size, int dim, int rank, scalar_t dt, scalar_t dt_scale, int steps,
    int topology_id, scalar_t R, scalar_t r,
    scalar_t* traj_x, scalar_t* traj_v, scalar_t* traj_acc1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t cx[64], cv[64], gamma[64], x_pred[64], v_pred[64], acc1[64], acc2[64];
    for (int i = 0; i < dim; ++i) { cx[i] = x_in[idx * dim + i]; cv[i] = v_in[idx * dim + i]; }
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    const scalar_t* f_ptr = force + idx * dim;

    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < dim; ++i) {
            traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
            traj_v[idx * (steps + 1) * dim + step * dim + i] = cv[i];
        }
        
        christoffel_device(cv, U, W, cx, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gamma);
        for (int i = 0; i < dim; ++i) {
            acc1[i] = f_ptr[i] - gamma[i];
            traj_acc1[idx * steps * dim + step * dim + i] = acc1[i];
            x_pred[i] = cx[i] + effective_dt * cv[i];
            v_pred[i] = cv[i] + effective_dt * acc1[i];
        }
        apply_boundary_vector(x_pred, dim, topology);
        
        christoffel_device(v_pred, U, W, x_pred, nullptr, dim, rank, 0.0, 1.0, 1.0, topology, R, r, gamma);
        for (int i = 0; i < dim; ++i) {
            acc2[i] = f_ptr[i] - gamma[i];
            cx[i] += (effective_dt / 2.0f) * (cv[i] + v_pred[i]);
            cv[i] += (effective_dt / 2.0f) * (acc1[i] + acc2[i]);
        }
        apply_boundary_vector(cx, dim, topology);
    }
    for (int i = 0; i < dim; ++i) {
        traj_x[idx * (steps + 1) * dim + steps * dim + i] = cx[i];
        traj_v[idx * (steps + 1) * dim + steps * dim + i] = cv[i];
    }
}

// C++ Wrapper
std::vector<torch::Tensor> heun_backward_cuda(
    torch::Tensor grad_x_out, torch::Tensor grad_v_out,
    torch::Tensor x_in, torch::Tensor v_in, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps, int topology,
    float R, float r
) {
    int batch_size = x_in.size(0);
    int dim = x_in.size(1);
    int rank = U.size(1);

    auto options = x_in.options();
    auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
    auto traj_v = torch::empty({batch_size, steps + 1, dim}, options);
    auto traj_acc1 = torch::empty({batch_size, steps, dim}, options);

    auto grad_x_in = torch::zeros_like(x_in);
    auto grad_v_in = torch::zeros_like(v_in);
    auto grad_force = torch::zeros_like(force);
    auto grad_U = torch::zeros({batch_size, dim, rank}, options);
    auto grad_W = torch::zeros({batch_size, dim, rank}, options);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    heun_forward_traj_kernel<<<blocks, threads>>>(
        x_in.data_ptr<scalar_t>(), v_in.data_ptr<scalar_t>(), force.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
        batch_size, dim, rank, dt, dt_scale, steps, topology, R, r,
        traj_x.data_ptr<scalar_t>(), traj_v.data_ptr<scalar_t>(), traj_acc1.data_ptr<scalar_t>()
    );

    heun_backward_kernel<<<blocks, threads>>>(
        grad_x_out.data_ptr<scalar_t>(), grad_v_out.data_ptr<scalar_t>(),
        traj_x.data_ptr<scalar_t>(), traj_v.data_ptr<scalar_t>(), traj_acc1.data_ptr<scalar_t>(),
        force.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
        batch_size, dim, rank, dt, dt_scale, steps, topology, R, r,
        grad_x_in.data_ptr<scalar_t>(), grad_v_in.data_ptr<scalar_t>(), grad_force.data_ptr<scalar_t>(),
        grad_U.data_ptr<scalar_t>(), grad_W.data_ptr<scalar_t>()
    );

    return {grad_x_in, grad_v_in, grad_force, grad_U.sum(0), grad_W.sum(0)};
}

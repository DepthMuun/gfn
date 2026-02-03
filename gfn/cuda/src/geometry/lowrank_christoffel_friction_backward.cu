#include <torch/extension.h>
#include <cuda_runtime.h>
#include "christoffel_impl.cuh"

namespace gfn {
namespace cuda {

/**
 * Memory-efficient backward pass using atomicAdd for shared parameters.
 * Eliminates [batch, dim, rank] temporary tensors.
 */
__global__ void lowrank_christoffel_friction_backward_kernel(
    const scalar_t* __restrict__ grad_out,      // [batch, dim]
    const scalar_t* __restrict__ output,        // [batch, dim]
    const scalar_t* __restrict__ v,             // [batch, dim]
    const scalar_t* __restrict__ U,             // [dim, rank]
    const scalar_t* __restrict__ W,             // [dim, rank]
    const scalar_t* __restrict__ x,             // [batch, dim]
    const scalar_t* __restrict__ V_w,           // [dim] or nullptr
    const scalar_t* __restrict__ force,         // [batch, dim] or nullptr
    const scalar_t* __restrict__ W_forget,      // [dim, feat_dim]
    const scalar_t* __restrict__ b_forget,      // [dim]
    const scalar_t* __restrict__ W_input,       // [dim, dim] or nullptr
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    // Outputs (atomicAdd for parameters)
    scalar_t* __restrict__ grad_v,              // [batch, dim]
    scalar_t* __restrict__ grad_U,              // [dim, rank]
    scalar_t* __restrict__ grad_W,              // [dim, rank]
    scalar_t* __restrict__ grad_x,              // [batch, dim]
    scalar_t* __restrict__ grad_Vw,             // [dim] or nullptr
    scalar_t* __restrict__ grad_force,          // [batch, dim] or nullptr
    scalar_t* __restrict__ grad_Wf,             // [dim, feat_dim]
    scalar_t* __restrict__ grad_bf,             // [dim]
    scalar_t* __restrict__ grad_Wi              // [dim, dim] or nullptr
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const scalar_t* grad_out_b = grad_out + b * dim;
    const scalar_t* output_b = output + b * dim;
    const scalar_t* v_b = v + b * dim;
    const scalar_t* x_b = x + b * dim;
    const scalar_t* force_b = (force != nullptr) ? (force + b * dim) : nullptr;
    
    // Soft Tanh correction
    scalar_t grad_pre[64]; // Max dim 64 recommended for this kernel
    for(int i = 0; i < dim && i < 64; ++i) {
        scalar_t t = output_b[i] / CURVATURE_CLAMP;
        grad_pre[i] = grad_out_b[i] * (1.0f - t * t);
    }

    // Local gradients for non-shared inputs
    scalar_t* g_v_b = grad_v + b * dim;
    scalar_t* g_x_b = grad_x + b * dim;
    scalar_t* g_f_b = (grad_force != nullptr) ? (grad_force + b * dim) : nullptr;
    
    for(int i = 0; i < dim; ++i) { g_v_b[i] = 0; g_x_b[i] = 0; if(g_f_b) g_f_b[i] = 0; }

    // 1. Friction contribution dL/dmu_i = grad_pre[i] * v_b[i]
    scalar_t grad_mu[64];
    for(int i = 0; i < dim && i < 64; ++i) {
        grad_mu[i] = grad_pre[i] * v_b[i];
    }
    
    // BACKWARD FRICTION (Local to parameter gradients handled via atomicAdd)
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    scalar_t features[128];
    if (topology == Topology::TORUS) compute_fourier_features(features, x_b, dim);
    else vector_copy(features, x_b, dim);
    
    for (int i = 0; i < dim; ++i) {
        scalar_t z = b_forget[i];
        for (int j = 0; j < feat_dim; ++j) z += W_forget[i * feat_dim + j] * features[j];
        if (force_b != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) z += W_input[i * dim + j] * force_b[j];
        }
        
        scalar_t s = sigmoid(z);
        scalar_t dz = grad_mu[i] * FRICTION_SCALE * s * (1.0f - s);
        
        atomicAdd(&grad_bf[i], dz);
        for (int j = 0; j < feat_dim; ++j) atomicAdd(&grad_Wf[i * feat_dim + j], dz * features[j]);
        if (force_b != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                atomicAdd(&grad_Wi[i * dim + j], dz * force_b[j]);
                if (g_f_b) g_f_b[j] += dz * W_input[i * dim + j];
            }
        }
        
        if (topology == Topology::TORUS) {
            for (int j = 0; j < dim; ++j) {
                scalar_t d_sin = W_forget[i * feat_dim + j] * dz;
                scalar_t d_cos = W_forget[i * feat_dim + (dim + j)] * dz;
                g_x_b[j] += d_sin * cosf(x_b[j]) - d_cos * sinf(x_b[j]);
            }
        } else {
            for (int j = 0; j < dim; ++j) g_x_b[j] += W_forget[i * feat_dim + j] * dz;
        }
    }
    
    // BACKWARD CHRISTOFFEL
    scalar_t gamma_b[64], gv_l[64], gx_l[64];
    christoffel_device(v_b, U, W, x_b, V_w, dim, rank, plasticity, sing_thresh, sing_strength, topology, R, r, gamma_b);
    
    // Temporary locals for U, W gradients to use atomicAdd
    scalar_t h[64], grad_h[64], grad_q[64];
    scalar_t h_energy = 0.0f;
    for (int i = 0; i < rank; ++i) {
        scalar_t sum = 0.0f;
        for (int j = 0; j < dim; ++j) sum += U[j * rank + i] * v_b[j];
        h[i] = sum;
        h_energy += sum * sum;
    }
    scalar_t norm = sqrtf(h_energy);
    scalar_t S = 1.0f / (1.0f + norm + 1e-4f);
    
    scalar_t M_plas = 1.0f;
    if (plasticity != 0.0f) {
        scalar_t v_e = 0.0f;
        for (int i = 0; i < dim; ++i) v_e += v_b[i] * v_b[i];
        M_plas = (1.0f + plasticity * tanhf(v_e / dim));
    }
    
    scalar_t M_sing = 1.0f;
    scalar_t pot = 0.0f, gate = 0.0f, soft_m = 0.0f;
    if (x_b != nullptr && V_w != nullptr) {
        if (topology == Topology::TORUS) { for (int i = 0; i < dim; ++i) pot += sinf(x_b[i]) * V_w[i]; }
        else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
        gate = sigmoid(pot);
        soft_m = sigmoid(10.0f * (gate - sing_thresh));
        M_sing = (1.0f + (sing_strength - 1.0f) * soft_m);
    }
    scalar_t M = M_plas * M_sing;

    for (int j = 0; j < rank; ++j) {
        grad_q[j] = 0.0f;
        scalar_t q_base = h[j] * h[j] * S * M;
        for (int i = 0; i < dim; ++i) {
            atomicAdd(&grad_W[i * rank + j], grad_pre[i] * q_base);
            grad_q[j] += W[i * rank + j] * grad_pre[i];
        }
    }
    
    scalar_t sum_grad_q_h_sq = 0.0f;
    for (int i = 0; i < rank; ++i) sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
    scalar_t S_sq_M_norm = (norm > 1e-8f) ? (M * S * S / norm) : 0.0f;
    for (int i = 0; i < rank; ++i) grad_h[i] = grad_q[i] * h[i] * 2.0f * S * M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < rank; ++j) {
            atomicAdd(&grad_U[i * rank + j], v_b[i] * grad_h[j]);
            g_v_b[i] += U[i * rank + j] * grad_h[j];
        }
    }

    // Mu contribution to grad_v
    scalar_t mu_b[64];
    compute_friction(x_b, force_b, W_forget, b_forget, W_input, dim, topology, mu_b);
    for(int i = 0; i < dim; ++i) g_v_b[i] += mu_b[i] * grad_pre[i];

    // Singularity gradient for x and V_w
    if (x_b != nullptr && V_w != nullptr) {
        scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
        scalar_t dM_dsoft = (sing_strength - 1.0f);
        scalar_t dsoft_dgate = 10.0f * soft_m * (1.0f - soft_m);
        scalar_t dgate_dpot = gate * (1.0f - gate);
        scalar_t factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;
        
        for (int i = 0; i < dim; ++i) {
            scalar_t dpot_dxi = (topology == Topology::TORUS) ? cosf(x_b[i]) * V_w[i] : V_w[i];
            g_x_b[i] += factor * dpot_dxi;
            atomicAdd(&grad_Vw[i], factor * ((topology == Topology::TORUS) ? sinf(x_b[i]) : x_b[i]));
        }
    }
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

std::vector<torch::Tensor> lowrank_christoffel_friction_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor output,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    torch::Tensor force,
    torch::Tensor W_forget,
    torch::Tensor b_forget,
    torch::Tensor W_input,
    double plasticity,
    double sing_thresh,
    double sing_strength,
    int topology,
    double R,
    double r
) {
    const int batch_size = v.size(0);
    const int dim = v.size(1);
    const int rank = U.size(-1);
    const int feat_dim = (topology == 1) ? 2 * dim : dim;

    auto grad_v = torch::zeros_like(v);
    auto grad_x = torch::zeros_like(x);
    auto grad_force = (force.numel() > 0) ? torch::zeros_like(force) : torch::empty(0, force.options());
    
    // Parameters use direct accumulation via atomicAdd
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    auto grad_Wf = torch::zeros_like(W_forget);
    auto grad_bf = torch::zeros_like(b_forget);
    auto grad_Wi = (W_input.numel() > 0) ? torch::zeros_like(W_input) : torch::empty(0, v.options());
    auto grad_Vw = (V_w.numel() > 0) ? torch::zeros_like(V_w) : torch::empty(0, v.options());

    int threads = 128; // Reduced threads to increase register availability
    int blocks = (batch_size + threads - 1) / threads;

    lowrank_christoffel_friction_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        V_w.numel() > 0 ? V_w.data_ptr<scalar_t>() : nullptr,
        force.numel() > 0 ? force.data_ptr<scalar_t>() : nullptr,
        W_forget.data_ptr<scalar_t>(),
        b_forget.data_ptr<scalar_t>(),
        W_input.numel() > 0 ? W_input.data_ptr<scalar_t>() : nullptr,
        batch_size, dim, rank,
        static_cast<scalar_t>(plasticity),
        static_cast<scalar_t>(sing_thresh),
        static_cast<scalar_t>(sing_strength),
        static_cast<Topology>(topology),
        static_cast<scalar_t>(R),
        static_cast<scalar_t>(r),
        grad_v.data_ptr<scalar_t>(),
        grad_U.data_ptr<scalar_t>(),
        grad_W.data_ptr<scalar_t>(),
        grad_x.data_ptr<scalar_t>(),
        grad_Vw.numel() > 0 ? grad_Vw.data_ptr<scalar_t>() : nullptr,
        grad_force.numel() > 0 ? grad_force.data_ptr<scalar_t>() : nullptr,
        grad_Wf.data_ptr<scalar_t>(),
        grad_bf.data_ptr<scalar_t>(),
        grad_Wi.numel() > 0 ? grad_Wi.data_ptr<scalar_t>() : nullptr
    );

    return { grad_v, grad_U, grad_W, grad_x, grad_Vw, grad_force, grad_Wf, grad_bf, grad_Wi };
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "christoffel_impl.cuh"

namespace gfn {
namespace cuda {

__global__ void christoffel_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ V_w,
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ grad_U,  // Shared: use atomicAdd
    scalar_t* __restrict__ grad_W,  // Shared: use atomicAdd
    scalar_t* __restrict__ grad_x
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const scalar_t* grad_out_b = grad_out + b * dim;
    const scalar_t* gamma_b = gamma + b * dim;
    const scalar_t* v_b = v + b * dim;
    const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;
    
    scalar_t* g_v_b = grad_v + b * dim;
    scalar_t* g_x_b = (grad_x != nullptr) ? (grad_x + b * dim) : nullptr;

    // Use a simplified backward that uses atomicAdd for parameters
    // We re-implement the logic here to avoid huge temporary allocations in the host wrapper
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
        soft_m = sigmoid(SINGULARITY_GATE_SLOPE * (gate - sing_thresh));
        M_sing = (1.0f + (sing_strength - 1.0f) * soft_m);
    }
    scalar_t M = M_plas * M_sing;

    for (int j = 0; j < rank; ++j) {
        grad_q[j] = 0.0f;
        scalar_t q_base = h[j] * h[j] * S * M;
        for (int i = 0; i < dim; ++i) {
            atomicAdd(&grad_W[i * rank + j], grad_out_b[i] * q_base);
            grad_q[j] += W[i * rank + j] * grad_out_b[i];
        }
    }
    
    scalar_t sum_grad_q_h_sq = 0.0f;
    for (int i = 0; i < rank; ++i) sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
    scalar_t S_sq_M_norm = (norm > 1e-8f) ? (M * S * S / norm) : 0.0f;
    for (int i = 0; i < rank; ++i) grad_h[i] = grad_q[i] * h[i] * 2.0f * S * M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    
    for (int i = 0; i < dim; ++i) {
        g_v_b[i] = 0;
        for (int j = 0; j < rank; ++j) {
            atomicAdd(&grad_U[i * rank + j], v_b[i] * grad_h[j]);
            g_v_b[i] += U[i * rank + j] * grad_h[j];
        }
    }

    if (g_x_b != nullptr && V_w != nullptr) {
        scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
        scalar_t factor = dL_dM_sing * (sing_strength - 1.0f) * SINGULARITY_GATE_SLOPE * soft_m * (1.0f - soft_m) * gate * (1.0f - gate);
        for (int i = 0; i < dim; ++i) {
            g_x_b[i] = factor * ((topology == Topology::TORUS) ? cosf(x_b[i]) * V_w[i] : V_w[i]);
        }
    }
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

std::vector<torch::Tensor> christoffel_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor gamma,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    double plasticity,
    double sing_thresh,
    double sing_strength,
    int topology,
    double R,
    double r
) {
    int batch_size = v.size(0);
    int dim = v.size(1);
    int rank = U.size(1);

    auto grad_v = torch::zeros_like(v);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    auto grad_x = torch::zeros_like(x);

    int threads = 128;
    int blocks = (batch_size + threads - 1) / threads;

    christoffel_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<scalar_t>(),
        gamma.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        x.numel() > 0 ? x.data_ptr<scalar_t>() : nullptr,
        V_w.numel() > 0 ? V_w.data_ptr<scalar_t>() : nullptr,
        batch_size, dim, rank,
        plasticity, sing_thresh, sing_strength,
        static_cast<Topology>(topology), R, r,
        grad_v.data_ptr<scalar_t>(),
        grad_U.data_ptr<scalar_t>(),
        grad_W.data_ptr<scalar_t>(),
        grad_x.numel() > 0 ? grad_x.data_ptr<scalar_t>() : nullptr
    );

    return {grad_v, grad_U, grad_W, grad_x};
}

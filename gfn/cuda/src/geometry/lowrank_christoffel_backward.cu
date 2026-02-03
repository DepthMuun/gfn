#include <torch/extension.h>
#include <cuda_runtime.h>
#include "christoffel_impl.cuh"

namespace gfn {
namespace cuda {

__global__ void christoffel_backward_kernel(
    const scalar_t* grad_out,
    const scalar_t* gamma,
    const scalar_t* v,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x,
    const scalar_t* V_w,
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* grad_v,
    scalar_t* grad_U,
    scalar_t* grad_W,
    scalar_t* grad_x
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Direct pointers for this batch item
    const scalar_t* grad_out_b = grad_out + b * dim;
    const scalar_t* gamma_b = gamma + b * dim;
    const scalar_t* v_b = v + b * dim;
    const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;
    
    scalar_t* grad_v_b = grad_v + b * dim;
    scalar_t* grad_U_b = grad_U + b * dim * rank;
    scalar_t* grad_W_b = grad_W + b * dim * rank;
    scalar_t* grad_x_b = grad_x ? grad_x + b * dim : nullptr;

    christoffel_backward_device(
        grad_out_b, gamma_b, v_b, U, W, x_b, V_w, dim, rank,
        plasticity, sing_thresh, sing_strength, topology, R, r,
        grad_v_b, grad_U_b, grad_W_b, grad_x_b
    );
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

// Wrapper for C++
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
    auto grad_U = torch::zeros({batch_size, dim, rank}, v.options());
    auto grad_W = torch::zeros({batch_size, dim, rank}, v.options());
    auto grad_x = torch::zeros_like(x);

    int threads = 256;
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

    return {grad_v, grad_U.sum(0), grad_W.sum(0), grad_x};
}

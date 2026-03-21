#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// ------------------------------------------------------------------------
// Toroidal Distance Loss
// L(y_pred, y_true) = (atan2(sin(y_pred - y_true), cos(y_pred - y_true)))^2
// ------------------------------------------------------------------------

template <typename scalar_t>
__global__ void toroidal_distance_loss_fwd_kernel(
    const scalar_t* __restrict__ y_pred,
    const scalar_t* __restrict__ y_true,
    scalar_t* __restrict__ out,
    const int numel) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t diff = y_pred[idx] - y_true[idx];
        scalar_t wrapped = atan2(sin(diff), cos(diff));
        out[idx] = wrapped * wrapped;
    }
}

template <typename scalar_t>
__global__ void toroidal_distance_loss_bwd_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ y_pred,
    const scalar_t* __restrict__ y_true,
    scalar_t* __restrict__ grad_pred,
    const int numel) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Derivada de atan2(sin(x), cos(x))^2 respecto a x es 2 * atan2(sin(x), cos(x))
        scalar_t diff = y_pred[idx] - y_true[idx];
        scalar_t wrapped = atan2(sin(diff), cos(diff));
        grad_pred[idx] = grad_output[idx] * 2.0 * wrapped;
    }
}

// ------------------------------------------------------------------------
// Wrappers ATen
// ------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor toroidal_distance_loss_fwd(const torch::Tensor& y_pred, const torch::Tensor& y_true) {
    CHECK_INPUT(y_pred);
    CHECK_INPUT(y_true);

    auto out = torch::empty_like(y_pred);
    int numel = y_pred.numel();

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    if (y_pred.scalar_type() == torch::kFloat32) {
        toroidal_distance_loss_fwd_kernel<float><<<blocks, threads>>>(
            y_pred.data_ptr<float>(),
            y_true.data_ptr<float>(),
            out.data_ptr<float>(),
            numel
        );
    } else {
        TORCH_CHECK(false, "toroidal_distance_loss_fwd only supports float32");
    }

    return out;
}

torch::Tensor toroidal_distance_loss_bwd(const torch::Tensor& grad_output, const torch::Tensor& y_pred, const torch::Tensor& y_true) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(y_pred);
    CHECK_INPUT(y_true);

    auto grad_pred = torch::empty_like(y_pred);
    int numel = y_pred.numel();

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    if (y_pred.scalar_type() == torch::kFloat32) {
        toroidal_distance_loss_bwd_kernel<float><<<blocks, threads>>>(
            grad_output.data_ptr<float>(),
            y_pred.data_ptr<float>(),
            y_true.data_ptr<float>(),
            grad_pred.data_ptr<float>(),
            numel
        );
    } else {
        TORCH_CHECK(false, "toroidal_distance_loss_bwd only supports float32");
    }

    return grad_pred;
}

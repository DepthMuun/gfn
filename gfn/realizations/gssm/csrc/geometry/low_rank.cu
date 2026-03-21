#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void low_rank_christoffel_fwd_kernel(
    const scalar_t* __restrict__ v,  // [B, H, D]
    const scalar_t* __restrict__ U,  // [H, D, R]
    const scalar_t* __restrict__ W,  // [H, D, R]
    scalar_t* __restrict__ gamma,    // [B, H, D]
    const int B, const int H, const int D, const int R,
    const scalar_t clamp_val,
    const bool enable_trace_norm,
    const bool is_paper_version) 
{
    // A block computes the output for one (b, h) pair
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int h = bh % H;
    
    // Dynamic shared memory allocations
    extern __shared__ char smem[];
    scalar_t* v_s_d     = reinterpret_cast<scalar_t*>(smem);                // Size: D
    scalar_t* vr_sq_s   = reinterpret_cast<scalar_t*>(&v_s_d[D]);           // Size: R
    scalar_t* gamma_s_d = reinterpret_cast<scalar_t*>(&vr_sq_s[R]);         // Size: D
    
    const scalar_t* v_b = v + bh * D;
    scalar_t* gamma_b = gamma + bh * D;
    
    // Pointers for H offset
    const scalar_t* U_h = U + h * D * R;
    const scalar_t* W_h = W + h * D * R;
    
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    
    // 1. Load v into shared memory
    for (int i = tid; i < D; i += bdim) {
        v_s_d[i] = v_b[i];
    }
    __syncthreads();
    
    // 2. Compute v_r = v @ U -> sq = v_r^2
    for (int r = tid; r < R; r += bdim) {
        scalar_t sum = 0;
        for (int j = 0; j < D; ++j) {
            sum += v_s_d[j] * U_h[j * R + r];
        }
        vr_sq_s[r] = sum * sum;
    }
    __syncthreads();

    // 3. Optional: Paper Low Rank denominator logic
    if (is_paper_version) {
        __shared__ scalar_t block_sum_sq;
        if (tid == 0) block_sum_sq = 0;
        __syncthreads();
        
        scalar_t local_sq = 0;
        for (int r = tid; r < R; r += bdim) {
            local_sq += vr_sq_s[r];
        }
        atomicAdd(&block_sum_sq, local_sq);
        __syncthreads();
        
        scalar_t norm_vr = sqrt(block_sum_sq);
        scalar_t denom = 1.0 + norm_vr;
        
        for (int r = tid; r < R; r += bdim) {
            vr_sq_s[r] = vr_sq_s[r] / denom;
        }
        __syncthreads();
    }
    
    // 4. Compute gamma_raw = sq @ W.T 
    for (int d = tid; d < D; d += bdim) {
        scalar_t sum = 0;
        for (int r = 0; r < R; ++r) {
            sum += vr_sq_s[r] * W_h[d * R + r];
        }
        gamma_s_d[d] = sum;
    }
    __syncthreads();
    
    // 5. Trace normalization (mean subtraction)
    scalar_t mean_val = 0;
    if (enable_trace_norm) {
        __shared__ scalar_t block_sum_gamma;
        if (tid == 0) block_sum_gamma = 0;
        __syncthreads();
        
        scalar_t local_gamma_sum = 0;
        for (int d = tid; d < D; d += bdim) {
            local_gamma_sum += gamma_s_d[d];
        }
        atomicAdd(&block_sum_gamma, local_gamma_sum);
        __syncthreads();
        
        mean_val = block_sum_gamma / D;
    }
    
    // 6. Normalization and storage
    for (int d = tid; d < D; d += bdim) {
        scalar_t g = gamma_s_d[d];
        if (enable_trace_norm) {
            g -= mean_val;
        }
        g = clamp_val * tanh(g / clamp_val);
        gamma_b[d] = g;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor low_rank_christoffel_fwd(
    const torch::Tensor& v, 
    const torch::Tensor& U, 
    const torch::Tensor& W,
    double clamp_val,
    bool enable_trace_norm,
    bool is_paper_version) 
{
    CHECK_INPUT(v);
    CHECK_INPUT(U);
    CHECK_INPUT(W);

    // Ensure shapes: v is [B, H, D], U is [H, D, R], W is [H, D, R]
    int B = v.size(0);
    int H = v.size(1);
    int D = v.size(2);
    int R = U.size(2);

    auto gamma = torch::empty_like(v);

    const int threads = 256;
    const int blocks = B * H;
    
    // Shared memory size: (D + R + D) * sizeof(float)
    const int shared_mem_size = (2 * D + R) * sizeof(float);

    if (v.scalar_type() == torch::kFloat32) {
        low_rank_christoffel_fwd_kernel<float><<<blocks, threads, shared_mem_size>>>(
            v.data_ptr<float>(),
            U.data_ptr<float>(),
            W.data_ptr<float>(),
            gamma.data_ptr<float>(),
            B, H, D, R,
            static_cast<float>(clamp_val),
            enable_trace_norm,
            is_paper_version
        );
    } else {
        TORCH_CHECK(false, "low_rank_christoffel_fwd only supports float32");
    }

    return gamma;
}

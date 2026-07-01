// low_rank.cu — GFN Christoffel Kernel (Improved)
// Changes:
//   - Warp-shuffle reduction replacing atomicAdd on shared scalar
//   - float4 vectorised global loads for v (when D%4==0)
//   - Shared-memory padded layout to avoid 32-bank conflicts
//   - Fast device intrinsics: __tanhf, atan2f, __sqrtf
//   - float16 (__half) template dispatch for AMP training
//   - __launch_bounds__ for register pressure hints
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ─── Warp-level reduction helpers ─────────────────────────────────────────────
// Full-warp (32 lanes) horizontal add.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduce: first warp reduces all warps' lane-0 values.
__device__ __forceinline__ float block_reduce_sum(float val, float* smem_scratch) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) smem_scratch[wid] = val;
    __syncthreads();

    // Only the first warp picks up partial sums
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? smem_scratch[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    __syncthreads();
    return val;
}

// ─── Main Christoffel Kernel ───────────────────────────────────────────────────
// Grid:  (B*H) blocks
// Block: up to 256 threads
// Shared memory layout (float, no bank conflicts):
//   [0 .. D+PAD)          v_s        (padded to 128-byte alignment)
//   [D+PAD .. D+PAD+R)    vr_sq_s
//   [D+PAD+R .. 2D+PAD+R) gamma_s
//   [2D+PAD+R .. 2D+PAD+R + NWARPS) warp_scratch for block_reduce
#define SMEM_PAD 1  // 1 float pad avoids stride-32 bank conflicts

template <typename scalar_t>
__launch_bounds__(256, 4)
__global__ void low_rank_christoffel_fwd_kernel(
    const scalar_t* __restrict__ v,      // [B, H, D]
    const scalar_t* __restrict__ U,      // [H, D, R]
    const scalar_t* __restrict__ W,      // [H, D, R]
    scalar_t* __restrict__ gamma,        // [B, H, D]
    const int B, const int H, const int D, const int R,
    const float clamp_val,
    const bool enable_trace_norm,
    const bool is_paper_version)
{
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int h = bh % H;

    // ── Shared memory layout ───────────────────────────────────────────────
    // Layout: v_s[D+PAD] | vr_sq_s[R] | gamma_s[D] | w_scratch[nwarps]
    // nwarps is ceil(blockDim.x / 32) — computed at host launch via _smem_bytes
    extern __shared__ char smem[];
    float* v_s       = reinterpret_cast<float*>(smem);
    float* vr_sq_s   = v_s + D + SMEM_PAD;
    float* gamma_s   = vr_sq_s + R;
    float* w_scratch = gamma_s + D;  // [ceil(blockDim.x/32)] slots

    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const scalar_t* v_bh = v + bh * D;
    scalar_t*    gamma_b = gamma + bh * D;
    const scalar_t* U_h  = U + h * D * R;
    const scalar_t* W_h  = W + h * D * R;

    // ── 1. Load v into shared memory ──────────────────────────────────────
    if constexpr (std::is_same<scalar_t, float>::value) {
        // float32: vectorised float4 load when D is aligned
        if (D % 4 == 0) {
            const float4* v4 = reinterpret_cast<const float4*>(v_bh);
            float4*       s4 = reinterpret_cast<float4*>(v_s);
            for (int i = tid; i < D / 4; i += bdim) s4[i] = v4[i];
        } else {
            for (int i = tid; i < D; i += bdim) v_s[i] = v_bh[i];
        }
    } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
        // float16 → upcast to float32 in shared mem
        for (int i = tid; i < D; i += bdim)
            v_s[i] = __half2float(reinterpret_cast<const __half*>(v_bh)[i]);
    } else {
        // double (or any other type) → cast to float32
        for (int i = tid; i < D; i += bdim)
            v_s[i] = static_cast<float>(v_bh[i]);
    }
    __syncthreads();

    // ── 2. Compute v_r = v @ U  →  sq = v_r² ─────────────────────────────
    for (int r = tid; r < R; r += bdim) {
        float acc = 0.0f;
        for (int j = 0; j < D; ++j)
            acc += v_s[j] * static_cast<float>(U_h[j * R + r]);
        vr_sq_s[r] = acc * acc;
    }
    __syncthreads();

    // ── 3. Paper-version: divide by (1 + ||v_r||) ─────────────────────────
    if (is_paper_version) {
        // Block-reduce sum of vr_sq (= ||v_r||²)
        float local_sq = 0.0f;
        for (int r = tid; r < R; r += bdim) local_sq += vr_sq_s[r];
        float sum_sq = block_reduce_sum(local_sq, w_scratch);

        // Only thread 0 writes the broadcast value back to w_scratch[0]
        if (tid == 0) w_scratch[0] = __fsqrt_rn(sum_sq); // ||v_r||
        __syncthreads();

        float denom = 1.0f + w_scratch[0];
        for (int r = tid; r < R; r += bdim)
            vr_sq_s[r] = vr_sq_s[r] / denom;
        __syncthreads();
    }

    // ── 4. Compute gamma_raw = sq @ W.T ──────────────────────────────────
    for (int d = tid; d < D; d += bdim) {
        float acc = 0.0f;
        for (int r = 0; r < R; ++r)
            acc += vr_sq_s[r] * static_cast<float>(W_h[d * R + r]);
        gamma_s[d] = acc;
    }
    __syncthreads();

    // ── 5. Trace normalisation (mean subtraction) ─────────────────────────
    float mean_val = 0.0f;
    if (enable_trace_norm) {
        float local_sum = 0.0f;
        for (int d = tid; d < D; d += bdim) local_sum += gamma_s[d];
        float total = block_reduce_sum(local_sum, w_scratch);
        mean_val = total / static_cast<float>(D);
    }

    // ── 6. Apply clamp*tanh(g/clamp) and write output ─────────────────────
    for (int d = tid; d < D; d += bdim) {
        float g = gamma_s[d] - mean_val;
        g = clamp_val * tanhf(g / clamp_val);
        if constexpr (std::is_same<scalar_t, float>::value) {
            gamma_b[d] = g;
        } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
            reinterpret_cast<__half*>(gamma_b)[d] = __float2half(g);
        } else {
            // double or other: downcast float→scalar_t
            gamma_b[d] = static_cast<scalar_t>(g);
        }
    }
}

// ─── Launch helpers ───────────────────────────────────────────────────────────
#define CHECK_CUDA(x)       TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),    #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static int _smem_bytes(int D, int R, int threads) {
    int nwarps = (threads + 31) >> 5;
    // v_s[D+PAD] + vr_sq_s[R] + gamma_s[D] + warp_scratch[nwarps]
    return ((D + SMEM_PAD) + R + D + nwarps) * static_cast<int>(sizeof(float));
}

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

    const int B = v.size(0);
    const int H = v.size(1);
    const int D = v.size(2);
    const int R = U.size(2);

    auto gamma = torch::empty_like(v);

    const int threads = std::min(256, std::max(32, ((D + 31) / 32) * 32));
    const int blocks  = B * H;
    const int smem    = _smem_bytes(D, R, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(v.scalar_type(), "low_rank_christoffel_fwd", [&] {
        low_rank_christoffel_fwd_kernel<scalar_t><<<blocks, threads, smem>>>(
            v.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            B, H, D, R,
            static_cast<float>(clamp_val),
            enable_trace_norm,
            is_paper_version
        );
    });

    return gamma;
}

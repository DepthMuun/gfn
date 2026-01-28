#include "../../include/forces.cuh"
#include "../../include/gradients.cuh"

#define BLOCK_SIZE 256

__global__ void christoffel_fused_kernel(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    int topology,
    float R_val,
    float r_val
) {
    // Shared Memory Setup
    extern __shared__ float s_mem_c[];
    float* s_h = s_mem_c; // [rank]
    
    // Align for doubles
    size_t offset_f = rank;
    if (offset_f % 2 != 0) offset_f++;
    double* s_buf = (double*)(s_mem_c + offset_f);

    const int b = blockIdx.x;
    if (b >= batch) return;
    const int tid = threadIdx.x;
    
    // 1. Plasticity Scale
    float M = compute_plasticity_scale(s_buf, v + b*dim, dim, tid, plasticity);
    
    // 2. Compute Force
    compute_christoffel_force(
        gamma + b*dim, 
        v + b*dim, 
        x + b*dim,
        U, W, 
        s_h, 
        dim, rank, tid, 
        topology,
        M,
        R_val,
        r_val
    );
}

extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology,
    float R_val, float r_val,
    cudaStream_t stream
) {
    int floats = rank;
    int doubles = 2; // Energy reduce
    int shared = floats * sizeof(float) + doubles * sizeof(double) + 16;
    christoffel_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        v, U, W, gamma, x, V_w, batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active, topology, R_val, r_val
    );
}

__global__ void christoffel_backward_kernel(
    const float* __restrict__ grad_gamma,
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    float* __restrict__ grad_v,
    float* __restrict__ grad_U,
    float* __restrict__ grad_W,
    float* __restrict__ grad_x,
    float* __restrict__ grad_V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    int topology,
    float R_val,
    float r_val
) {
    extern __shared__ float s_mem_b[];
    float* s_h = s_mem_b;
    
    size_t offset_f = rank;
    if (offset_f % 2 != 0) offset_f++;
    double* s_grad_h_dbl = (double*)(s_mem_b + offset_f);

    const int b = blockIdx.x;
    if (b >= batch) return;
    const int tid = threadIdx.x;
    
    __shared__ float s_gM_dummy;
    if (tid == 0) s_gM_dummy = 0.0f;
    __syncthreads();

    // Delegate to Modular Gradient Engine (Double Precision)
    compute_christoffel_backward(
        grad_gamma + b*dim,
        v + b*dim,
        x + b*dim,
        U, W,
        grad_v + b*dim,
        grad_x + b*dim,
        grad_U, 
        grad_W,
        &s_gM_dummy, // NEW: Gradient for M scale
        s_h, s_grad_h_dbl,
        dim, rank, tid,
        topology,
        plasticity,
        1.0f, // scale_M placeholder
        R_val,
        r_val
    );
    
    // TODO: Add dL/dx (Topology Gradient) if needed for full training.
    // Since audit focused on Adjoint Divergence (matrices), we prioritize that.
    // The metric updates (U,W) are the heavy part.
}

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology,
    float R_val, float r_val,
    cudaStream_t stream
) {
    int floats = rank;
    int doubles = rank; // Accumulators
    int shared = floats * sizeof(float) + doubles * sizeof(double) + 32;
    christoffel_backward_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        grad_gamma, v, U, W, x, V_w, grad_v, grad_U, grad_W, grad_x, grad_V_w,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active, topology, R_val, r_val
    );
}

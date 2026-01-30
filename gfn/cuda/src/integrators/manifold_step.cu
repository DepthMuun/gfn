#include <cuda.h>
#include <cuda_runtime.h>
#include "../../include/forces.cuh"

#define BLOCK_SIZE 256

__global__ void manifold_step_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ force,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float* __restrict__ christoffel_out,
    const int batch,
    const int dim,
    const int rank,
    const float dt,
    const float dt_scale,
    const float plasticity,
    const float sing_thresh,
    const float sing_strength,
    int topology,
    float R_val,
    float r_val
) {
    extern __shared__ float s_mem[];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;
    
    float* s_v = s_mem;
    float* s_x = s_v + dim;
    float* s_gamma = s_x + dim;
    float* s_h = s_gamma + dim;
    
    // Double alignment for Energy buffer
    size_t offset_f = (3 * dim + rank);
    if (offset_f % 2 != 0) offset_f++;
    double* s_buf_energy = (double*)(s_mem + offset_f);

    for (int i = tid; i < dim; i += blockDim.x) {
        s_v[i] = v_in[b * dim + i];
        s_x[i] = x_in[b * dim + i];
    }
    __syncthreads();
    
    // 1. Plasticity
    float M = compute_plasticity_scale(s_buf_energy, s_v, dim, tid, plasticity);
    
    // 2. Christoffel
    compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
    __syncthreads();
    
    const float eff_dt = dt * dt_scale;
    for (int i = tid; i < dim; i += blockDim.x) {
        float f_val = (force != nullptr) ? force[b * dim + i] : 0.0f;
        float g = s_gamma[i];
        if (christoffel_out != nullptr) christoffel_out[b * dim + i] = g;
        float v_n = s_v[i] + eff_dt * (f_val - g);
        x_out[b * dim + i] = s_x[i] + eff_dt * v_n;
        v_out[b * dim + i] = v_n;
    }
}

extern "C" void launch_manifold_step_fused(
    const float* x_in, const float* v_in, const float* force,
    const float* U, const float* W,
    float* x_out, float* v_out, float* christoffel_out,
    int batch, int dim, int rank,
    float dt, float dt_scale,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, float R_val, float r_val,
    cudaStream_t stream
) {
    int shared_bytes = (3 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    manifold_step_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_in, v_in, force, U, W, x_out, v_out, christoffel_out,
        batch, dim, rank, dt, dt_scale, plasticity, sing_thresh, sing_strength, topology, R_val, r_val
    );
}

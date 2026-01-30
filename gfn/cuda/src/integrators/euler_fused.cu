
#include "../../include/forces.cuh"

#define BLOCK_SIZE 256

__global__ void euler_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale,
    const int batch,
    const int dim,
    const int rank,
    const int steps,
    int topology,
    float R_val,
    float r_val
) {
    extern __shared__ float s_mem_f[];
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_gamma = s_v + dim;
    float* s_h = s_gamma + dim;
    
    // Double alignment for Energy buffer
    size_t offset_f = (3 * dim + rank);
    if (offset_f % 2 != 0) offset_f++;
    double* s_buf_energy = (double*)(s_mem_f + offset_f);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale;

    for (int s = 0; s < steps; s++) {
        // 1. Plasticity
        float M = compute_plasticity_scale(s_buf_energy, s_v, dim, tid, 0.0f); // Default 0 plasticity for Euler?
        // Note: Euler implementation didn't have plasticity param passed in previously?
        // Checking legacy call: passed '0.0f' as plasticity.
        
        // 2. Christoffel Force
        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        __syncthreads();

        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += eff_dt * (f_val - s_gamma[i]);
            s_x[i] += eff_dt * s_v[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_euler_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps, int topology, float R_val, float r_val,
    cudaStream_t stream
) {
    int shared = (3 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    euler_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, x_new, v_new, dt, dt_scale, batch, dim, rank, steps, topology, R_val, r_val
    );
}

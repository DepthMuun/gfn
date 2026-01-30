
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

// Forest-Ruth Coefficients
#define FR_THETA 1.3512071919596578f

__global__ void forest_ruth_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ V_w,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale_scalar,
    const float* __restrict__ dt_scale_tensor,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
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
    
    double* s_mem_d = (double*)(s_h + rank + (rank % 2));
    double* s_E = s_mem_d;
    double* s_P = s_E + 1;
    float* s_M = (float*)(s_P + 1);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    float scale = (dt_scale_tensor != nullptr) ? dt_scale_tensor[b] : dt_scale_scalar;
    float h_dt = dt * scale;

    for (int s = 0; s < steps; s++) {
        // Stage 1
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += FR_THETA * 0.5f * h_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, topology, s_h, s_E, s_P, s_M, R_val, r_val);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_v = (f) ? f[b * dim + i] : 0.0f;
            s_v[i] += FR_THETA * h_dt * (f_v - s_gamma[i]);
        }
        __syncthreads();

        // Stage 2
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += (1.0f - FR_THETA) * 0.5f * h_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, topology, s_h, s_E, s_P, s_M, R_val, r_val);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_v = (f) ? f[b * dim + i] : 0.0f;
            s_v[i] += (1.0f - 2.0f * FR_THETA) * h_dt * (f_v - s_gamma[i]);
        }
        __syncthreads();

        // Stage 3
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += (1.0f - FR_THETA) * 0.5f * h_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, topology, s_h, s_E, s_P, s_M, R_val, r_val);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_v = (f) ? f[b * dim + i] : 0.0f;
            s_v[i] += FR_THETA * h_dt * (f_v - s_gamma[i]);
        }
        __syncthreads();

        // Final Pos
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += FR_THETA * 0.5f * h_dt * s_v[i];
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_forest_ruth_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W, const float* V_w,
    float* x_new, float* v_new,
    float dt, float dt_scale_scalar,
    const float* dt_scale_tensor,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active,
    int steps, int topology, float R_val, float r_val,
    cudaStream_t stream
) {
    int shared = (3 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    forest_ruth_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, V_w, x_new, v_new, dt, dt_scale_scalar, dt_scale_tensor,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active, steps, topology, R_val, r_val
    );
}

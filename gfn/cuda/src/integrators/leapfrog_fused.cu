#include "../../include/forces.cuh"

#define BLOCK_SIZE 256

extern "C" __global__ void leapfrog_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    //  Friction Parameters
    const float* __restrict__ W_forget,
    const float* __restrict__ b_forget,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    const float* __restrict__ dt_scales, //  Pointer to tensor or nullptr
    const int batch,
    const int dim,
    const int rank,
    const int steps,
    int topology,
    float plasticity,
    float R_val,
    float r_val
) {
    // Dynamic Shared Memory
    extern __shared__ float s_mem_f[];
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_h = s_v + dim;
    
    // Alignment padding for doubles
    size_t offset_f = (3 * dim + rank); 
    if (offset_f % 2 != 0) offset_f++;
    
    double* s_buf_energy = (double*)(s_mem_f + offset_f);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    // Load Initial State
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    // ADAPTIVE TIME SCALING
    float scale = (dt_scales != nullptr) ? dt_scales[b] : 1.0f;
    const float eff_dt = dt * scale, half_dt = 0.5f * eff_dt;

    for (int s = 0; s < steps; s++) {
        
        // 1. Reactive Plasticity (Compute M based on Kinetic Energy)
        float M = compute_plasticity_scale(s_buf_energy, s_v, dim, tid, plasticity);
        
        // 2. First Kick (Half Step) with Implicit Friction
        float* s_mu = s_h + rank; 
        if (W_forget != nullptr && b_forget != nullptr) {
            compute_friction_coeff(s_mu, s_x, W_forget, b_forget, dim, tid, topology);
        } else {
            for (int i = tid; i < dim; i += blockDim.x) s_mu[i] = 0.0f;
        }
        __syncthreads();

        float* s_gamma = s_h + rank; 
        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            // Implicit update: v_next = (v_prev + h*(F - gamma)) / (1 + h*mu)
            s_v[i] = (s_v[i] + half_dt * (f_val - s_gamma[i])) / (1.0f + half_dt * s_mu[i]);
        }
        __syncthreads();

        // 3. Drift (Full Step)
        for (int i = tid; i < dim; i += blockDim.x) {
            s_x[i] += eff_dt * s_v[i];
            s_x[i] = apply_boundary(s_x[i], topology); 
        }
        __syncthreads();
        
        // 4. Second Kick (Half Step) with Implicit Friction at new position
        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            // mu[i] should technically be re-evaluated at the new position but s_mu is already buffered.
            // Note: In Python, mu is re-evaluated and here we reuse s_mu. For exactness, we should recompute.
            // Updating mu for the second half step:
            if (W_forget != nullptr && b_forget != nullptr) {
                compute_friction_coeff(s_mu, s_x, W_forget, b_forget, dim, tid, topology);
            }
        }
        __syncthreads();

        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] = (s_v[i] + half_dt * (f_val - s_gamma[i])) / (1.0f + half_dt * s_mu[i]);
        }
        __syncthreads();
    }

    // Write Back
    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_leapfrog_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    const float* W_forget, const float* b_forget,
    float* x_new, float* v_new,
    float dt, const float* dt_scales,
    int batch, int dim, int rank,
    int steps, int topology, float plasticity,
    float R_val, float r_val,
    cudaStream_t stream
) {
    int floats = 3 * dim + rank;
    int shared = floats * sizeof(float) + 32; 
    leapfrog_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, W_forget, b_forget, x_new, v_new, dt, dt_scales, 
        batch, dim, rank, steps, topology, plasticity, R_val, r_val
    );
}

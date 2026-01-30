#include "../../include/forces.cuh"

#define BLOCK_SIZE 256

extern "C" __global__ void rk4_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ W_forget,
    const float* __restrict__ b_forget,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale,
    const int batch,
    const int dim,
    const int rank,
    const int steps,
    int topology,
    float plasticity,
    float R_val,
    float r_val
) {
    extern __shared__ float s_mem_rk[];
    float* s_x = s_mem_rk;
    float* s_v = s_x + dim;
    float* s_h = s_v + dim;
    float* s_gamma = s_h + rank;
    
    // Additional buffers for RK4 intermediate states (k1, k2, k3, k4)?
    // Standard RK4:
    // k1v = a(x, v)
    // k1x = v
    // x2 = x + k1x * dt/2, v2 = v + k1v * dt/2
    // ...
    // This requires storage for intermediate x, v.
    // Let's allocate s_xt, s_vt in shared.
    float* s_xt = s_gamma + dim;
    float* s_vt = s_xt + dim;
    // And accumulators? Or accumulate in place?
    // x_new = x + dt/6 * (k1 + 2k2 + 2k3 + k4)
    // Need accumulators.
    float* s_xa = s_vt + dim;
    float* s_va = s_xa + dim;
    
    size_t offset_f = (7 * dim + rank); 
    if (offset_f % 2 != 0) offset_f++;
    double* s_buf_energy = (double*)(s_mem_rk + offset_f);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale;
    const float half_dt = 0.5f * eff_dt;
    const float sixth_dt = eff_dt / 6.0f;

    for (int s = 0; s < steps; s++) {
        // 1. Reactive Plasticity (Compute M based on Kinetic Energy)
        float M = compute_plasticity_scale(s_buf_energy, s_v, dim, tid, plasticity);
        
        // 2. Thermodynamic Friction (Part A)
        float* s_mu = s_va + dim; // Re-use portion of s_va as mu buffer
        if (W_forget != nullptr && b_forget != nullptr) {
            compute_friction_coeff(s_mu, s_x, W_forget, b_forget, dim, tid, topology);
            apply_friction_damping(s_v, s_mu, dim, tid, half_dt);
        }

        // --- K1 ---
        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc = f_val - s_gamma[i];
            s_xa[i] = s_v[i];       // k1x
            s_va[i] = acc;          // k1v
            
            s_xt[i] = apply_boundary(s_x[i] + s_v[i] * half_dt, topology);
            s_vt[i] = s_v[i] + acc * half_dt;
        }
        __syncthreads();

        // --- K2 ---
        float M2 = compute_plasticity_scale(s_buf_energy, s_vt, dim, tid, plasticity);
        compute_christoffel_force(s_gamma, s_vt, s_xt, U, W, s_h, dim, rank, tid, topology, M2, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc = f_val - s_gamma[i];
            s_xa[i] += 2.0f * s_vt[i];
            s_va[i] += 2.0f * acc;
            s_xt[i] = apply_boundary(s_x[i] + s_vt[i] * half_dt, topology);
            s_vt[i] = s_v[i] + acc * half_dt;
        }
        __syncthreads();

        // --- K3 ---
        float M3 = compute_plasticity_scale(s_buf_energy, s_vt, dim, tid, plasticity);
        compute_christoffel_force(s_gamma, s_vt, s_xt, U, W, s_h, dim, rank, tid, topology, M3, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc = f_val - s_gamma[i];
            s_xa[i] += 2.0f * s_vt[i];
            s_va[i] += 2.0f * acc;
            s_xt[i] = apply_boundary(s_x[i] + s_vt[i] * eff_dt, topology);
            s_vt[i] = s_v[i] + acc * eff_dt;
        }
        __syncthreads();

        // --- K4 ---
        float M4 = compute_plasticity_scale(s_buf_energy, s_vt, dim, tid, plasticity);
        compute_christoffel_force(s_gamma, s_vt, s_xt, U, W, s_h, dim, rank, tid, topology, M4, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc = f_val - s_gamma[i];
            s_xa[i] += s_vt[i];
            s_va[i] += acc;
            s_x[i] = apply_boundary(s_x[i] + s_xa[i] * sixth_dt, topology);
            s_v[i] = s_v[i] + s_va[i] * sixth_dt;
        }
        __syncthreads();
        
        // 6. Thermodynamic Friction (Part B)
        if (W_forget != nullptr && b_forget != nullptr) {
            apply_friction_damping(s_v, s_mu, dim, tid, half_dt);
        }
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_rk4_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    const float* W_forget, const float* b_forget,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps, int topology, float plasticity,
    float R_val, float r_val,
    cudaStream_t stream
) {
    int floats = 7 * dim + rank; 
    int shared = floats * sizeof(float) + 16;
    rk4_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, W_forget, b_forget, x_new, v_new, dt, dt_scale, 
        batch, dim, rank, steps, topology, plasticity, R_val, r_val
    );
}

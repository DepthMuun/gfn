#include "../../include/forces.cuh"

#define BLOCK_SIZE 256

// Yoshida Coefficients (4th Order Symplectic)
#define C1 0.67560359597982881702f
#define C4 0.67560359597982881702f
#define C2 -0.17560359597982881702f
#define C3 -0.17560359597982881702f
#define D1 1.35120719195965763405f
#define D3 1.35120719195965763405f
#define D2 -1.70241438391931526809f

extern "C" __global__ void yoshida_fused_kernel(
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
    extern __shared__ float s_mem_y[];
    float* s_x = s_mem_y;
    float* s_v = s_x + dim;
    float* s_h = s_v + dim;
    float* s_gamma = s_h + rank;
    
    // Double alignment for energy reduction
    size_t offset_f = (3 * dim + rank); 
    if (offset_f % 2 != 0) offset_f++;
    double* s_buf_energy = (double*)(s_mem_y + offset_f);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale, half_dt = 0.5f * eff_dt;
    const float dt_c1 = C1 * eff_dt;
    const float dt_c2 = C2 * eff_dt;
    const float dt_d1 = D1 * eff_dt;
    const float dt_d2 = D2 * eff_dt;

    for (int s = 0; s < steps; s++) {
        // 1. Reactive Plasticity (Constant for the step)
        float M = compute_plasticity_scale(s_buf_energy, s_v, dim, tid, plasticity);
        
        // 2. Thermodynamic Friction (Part A)
        float* s_mu = s_gamma + dim; // Use portion of shared as mu buffer
        if (W_forget != nullptr && b_forget != nullptr) {
            compute_friction_coeff(s_mu, s_x, W_forget, b_forget, dim, tid, topology);
            apply_friction_damping(s_v, s_mu, dim, tid, half_dt);
        }

        // --- SUBSTEP 1 ---
        for (int i = tid; i < dim; i += blockDim.x) {
            s_x[i] += dt_c1 * s_v[i];
            s_x[i] = apply_boundary(s_x[i], topology);
        }
        __syncthreads();
        
        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f; 
            s_v[i] += dt_d1 * (f_val - s_gamma[i]);
        }
        __syncthreads();

        // --- SUBSTEP 2 ---
        for (int i = tid; i < dim; i += blockDim.x) {
            s_x[i] += dt_c2 * s_v[i];
             s_x[i] = apply_boundary(s_x[i], topology);
        }
        __syncthreads();

        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += dt_d2 * (f_val - s_gamma[i]);
        }
        __syncthreads();

        // --- SUBSTEP 3 ---
        for (int i = tid; i < dim; i += blockDim.x) {
             s_x[i] += dt_c2 * s_v[i];
             s_x[i] = apply_boundary(s_x[i], topology);
        }
        __syncthreads();

        compute_christoffel_force(s_gamma, s_v, s_x, U, W, s_h, dim, rank, tid, topology, M, R_val, r_val);
        for (int i = tid; i < dim; i += blockDim.x) {
             float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
             s_v[i] += dt_d1 * (f_val - s_gamma[i]);
        }
        __syncthreads();

        // --- SUBSTEP 4 ---
        for (int i = tid; i < dim; i += blockDim.x) {
             s_x[i] += dt_c1 * s_v[i];
             s_x[i] = apply_boundary(s_x[i], topology);
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

extern "C" void launch_yoshida_fused(
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
    int floats = 3 * dim + rank + dim; // x, v, h, gamma
    int shared = floats * sizeof(float) + 16;
    yoshida_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, W_forget, b_forget, x_new, v_new, dt, dt_scale, 
        batch, dim, rank, steps, topology, plasticity, R_val, r_val
    );
}

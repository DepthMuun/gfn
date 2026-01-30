#pragma once
#include "physics_primitives.cuh"
#include "manifold_metrics.cuh"

// ==========================================
// Level 1: Core Field Operators
// ==========================================


__device__ __forceinline__ void head_mixing_device(
    float* s_x, float* s_v, const float* W_x, const float* W_v, 
    float* s_temp_x, float* s_temp_v, int dim, int tid, int topology
) {
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum_x = 0.0f, sum_v = 0.0f;
        if (topology == TORUS) { 
            for (int j = 0; j < dim; j++) {
                sum_x += sinf(s_x[j]) * W_x[i * (3 * dim) + j];
                sum_x += cosf(s_x[j]) * W_x[i * (3 * dim) + j + dim];
                sum_x += s_v[j]       * W_x[i * (3 * dim) + j + 2 * dim];
            }
        } else { 
            for (int j = 0; j < dim; j++) sum_x += s_x[j] * W_x[i * dim + j];
        }
        for (int j = 0; j < dim; j++) sum_v += s_v[j] * W_v[i * dim + j];
        s_temp_x[i] = sum_x; s_temp_v[i] = sum_v;
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = s_temp_x[i]; s_v[i] = s_temp_v[i]; }
    __syncthreads();
    rmsnorm_device(s_x, dim, tid); rmsnorm_device(s_v, dim, tid);
}

// ==========================================
// Level 2: Dynamics & Forces
// ==========================================

// --------------------------------------------------------
// A. Reactive Plasticity (Curvature Modulation)
// --------------------------------------------------------
__device__ __forceinline__ float compute_plasticity_scale(
    float* s_buf_energy, // Shared memory for reduction
    const float* v,
    int dim,
    int tid,
    float plasticity_alpha
) {
    // 1. Calc Kinetic Energy
    float local_E = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) local_E += v[i] * v[i];
    local_E = warpReduceSum(local_E);
    
    if (tid == 0) *s_buf_energy = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(s_buf_energy, local_E);
    __syncthreads();

    // 2. Modulate M
    // M = 1 + alpha * tanh(E / D)
    float E_mean = *s_buf_energy / (float)dim;
    return 1.0f + plasticity_alpha * tanhf(E_mean);
}

// Double Precision Overload
__device__ __forceinline__ float compute_plasticity_scale(
    double* s_buf_energy, // Shared memory for reduction (Double)
    const float* v,
    int dim,
    int tid,
    float plasticity_alpha
) {
    // 1. Calc Kinetic Energy
    double local_E = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) local_E += (double)v[i] * (double)v[i];
    
    // Warp Reduce for Double? primitives.cuh defaults to float.
    // We can cast to float for warp reduce OR implement warpReduceSum(double).
    // For plasticity, float precision is usually fine, but accumulation might overflow.
    // Let's implement simple double warp reduce inline or use atomicAdd logic directly if dim is small.
    // Or just cast the local sum to float for reduction if we don't have double-warp-reduce.
    // Given audit "Adjoint Divergence" was about gradients, Energy forward is less critical.
    // BUT we passed a double*, so we should write to it.
    
    // Simple inline warp reduce for double
    for (int offset = 16; offset > 0; offset /= 2)
        local_E += __shfl_down_sync(0xffffffff, local_E, offset);
        
    if (tid == 0) *s_buf_energy = 0.0;
    __syncthreads();
    
    if ((tid & 31) == 0) atomicAdd(s_buf_energy, local_E);
    __syncthreads();

    // 2. Modulate M
    float E_mean = (float)(*s_buf_energy / (double)dim);
    return 1.0f + plasticity_alpha * tanhf(E_mean);
}

// --------------------------------------------------------
// B. Thermodynamic Friction ("The Clutch")
// --------------------------------------------------------

// Part 1: Compute Coefficient (Used by Backward Pass to recover mu)
__device__ __forceinline__ void compute_friction_coeff(
    float* __restrict__ mu_out,
    const float* __restrict__ x,
    const float* __restrict__ W_x, 
    const float* __restrict__ b_gate,
    int dim, 
    int tid,
    int topology
) {
     for (int i = tid; i < dim; i += blockDim.x) {
        float gate_activation = b_gate[i];
        if (topology == TORUS) {
            for (int j = 0; j < dim; j++) {
                // Ideally use shared mem or sparse, but standard implementation for correctness:
                gate_activation += W_x[i * (2 * dim) + j] * sinf(x[j]);
                gate_activation += W_x[i * (2 * dim) + j + dim] * cosf(x[j]);
            }
        } else {
             for (int j = 0; j < dim; j++) gate_activation += W_x[i*dim + j] * x[j];
        }
        mu_out[i] = sigmoidf_device(gate_activation) * 5.0f; // Align with Python 5.0 max
    }
    __syncthreads();
}

// Part 2: Apply Damping
__device__ __forceinline__ void apply_friction_damping(
    float* __restrict__ v,
    const float* __restrict__ mu,
    int dim,
    int tid,
    float dt
) {
    for (int i = tid; i < dim; i += blockDim.x) {
        v[i] *= expf(-mu[i] * dt);
    }
    __syncthreads();
}

// Wrapper for Fused Kernels
__device__ __forceinline__ void apply_friction_gate(
    float* __restrict__ v,
    const float* __restrict__ x,
    const float* __restrict__ W_x, 
    const float* __restrict__ b_gate,
    int dim, 
    int tid,
    int topology,
    float dt
) {
    // We need temporary storage for mu?
    // Fused kernels might not have allocated s_mu. 
    // If we assume per-thread register independence... mu[i] depends on x (all j).
    // So we need to compute mu fully before modulating v?
    // Yes, but mu[i] only affects v[i].
    // So we can compute mu[i] and use it immediately without storing all mu.
    // Optimization: Compute & Apply inline.
    
    for (int i = tid; i < dim; i += blockDim.x) {
         float gate_activation = b_gate[i];
        if (topology == TORUS) {
            for (int j = 0; j < dim; j++) {
                gate_activation += W_x[i * (2 * dim) + j] * sinf(x[j]);
                gate_activation += W_x[i * (2 * dim) + j + dim] * cosf(x[j]);
            }
        } else {
             for (int j = 0; j < dim; j++) gate_activation += W_x[i*dim + j] * x[j];
        }
        float mu = sigmoidf_device(gate_activation) * 100.0f;
        v[i] *= expf(-mu * dt);
    }
    __syncthreads();
}


// --------------------------------------------------------
// C. Christoffel Acceleration (Fundamental Force)
// --------------------------------------------------------

// Part 1: Analytical Torus Support
__device__ __forceinline__ void compute_christoffel_torus(
    float* __restrict__ force_out,
    const float* __restrict__ v,
    const float* __restrict__ x,
    int dim,
    int tid,
    float R,
    float r,
    float scale_M
) {
    for (int i = tid; i < dim - 1; i += 2 * blockDim.x) {
        // We process pairs (i, i+1) -> (theta, phi)
        float th = x[i];
        float v_th = v[i];
        float v_ph = v[i+1];
        
        float cos_th = cosf(th);
        float sin_th = sinf(th);

        // Gamma^th_{ph, ph} = (R + r cos th) sin th / r
        // Gamma_th = Gamma^th_ph_ph * v_ph^2
        float term_th = (R + r * cos_th) * sin_th / r;
        float g_th = term_th * (v_ph * v_ph);

        // Gamma^ph_{ph, th} = -(r sin th) / (R + r cos th)
        // Gamma_ph = 2 * Gamma^ph_ph_th * v_ph * v_th
        float denom = R + r * cos_th;
        float denom_sign = (denom >= 0) ? 1.0f : -1.0f;
        float denom_safe = denom + denom_sign * 1e-6f;
        
        float term_ph = -(r * sin_th) / denom_safe;
        float g_ph = 2.0f * term_ph * v_ph * v_th;

        // Apply scaling (Unbounded to match Python)
        force_out[i] = g_th * scale_M * 0.05f;
        force_out[i+1] = g_ph * scale_M * 0.05f;
    }
    __syncthreads();
}

// Part 2: Low-Rank Approximation
__device__ __forceinline__ void compute_christoffel_low_rank(
    float* __restrict__ force_out, 
    const float* __restrict__ v,
    const float* __restrict__ U,   
    const float* __restrict__ W,   
    float* s_h,                    
    int dim,
    int rank,
    int tid,
    float scale_M,
    float* out_S = nullptr,
    float* out_M = nullptr
) {
    // 1. Expand h = U^T v
    for (int r = tid; r < rank; r += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += v[i] * U[i * rank + r];
        s_h[r] = sum;
    }
    __syncthreads();

    // 2. Scalar S = 1 / (1 + ||h||)
    float h_sq = 0.0f;
    for (int r = tid; r < rank; r += blockDim.x) h_sq += s_h[r] * s_h[r];
    h_sq = warpReduceSum(h_sq);
    
    __shared__ float s_norm_shared; // Avoid name conflict with kernel locals
    if (tid == 0) s_norm_shared = 0.0f; __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_norm_shared, h_sq);
    __syncthreads();
    
    float S = 1.0f / (1.0f + sqrtf(s_norm_shared) + 1e-6f);
    if (tid == 0 && out_S) *out_S = S;
    if (tid == 0 && out_M) *out_M = scale_M;

    // 3. Contract Gamma = W * h^2 * S * M
    for (int i = tid; i < dim; i += blockDim.x) {
        float g = 0.0f;
        for (int r = 0; r < rank; r++) g += W[i * rank + r] * s_h[r] * s_h[r];
        
        float res = g * S * scale_M;
        force_out[i] = 20.0f * tanhf(res / 20.0f);
    }
    __syncthreads();
}

// --------------------------------------------------------
// D. Singularity (Black Hole) Logic
// --------------------------------------------------------
__device__ __forceinline__ float compute_singularity_scale(
    const float* __restrict__ x, // State [dim] or [2*dim for Torus]
    const float* __restrict__ W_potential,
    const float* __restrict__ b_potential,
    int dim,
    int tid,
    int topology,
    float threshold,
    float strength
) {
    // 1. Compute Potential (Sigmoid(Wx + b))
    // Note: We need a reduction here.
    float local_act = 0.0f;
    
    // Distribute W*x computation
    // W_potential is [1, input_dim] usually. input_dim = dim (Euclidean) or 2*dim (Torus)
    
    if (topology == TORUS) {
        // Input is [sin, cos] implicit from x.
        // W is [1, 2*dim]. 
        // We assume W is stored as [dim, 2] effectively or flat 2*dim.
        // Flattened: [w_sin_0, w_sin_1... w_cos_0...] ??
        // Standard Linear layout: [out_features, in_features].
        // Here out=1. W is [1, 2*dim].
        // W[j] matches input[j].
        
        for (int i = tid; i < dim; i += blockDim.x) {
             float w_sin = W_potential[i]; 
             float w_cos = W_potential[i + dim];
             local_act += w_sin * sinf(x[i]) + w_cos * cosf(x[i]);
        }
    } else {
        for (int i = tid; i < dim; i += blockDim.x) {
            local_act += W_potential[i] * x[i];
        }
    }
    
    // Reduce
    local_act = warpReduceSum(local_act);
    
    // Shared mem for bias addition (only thread 0)
    __shared__ float s_pot;
    if (tid == 0) s_pot = 0.0f;
    __syncthreads();
    
    if ((tid & 31) == 0) atomicAdd(&s_pot, local_act);
    __syncthreads();
    
    // Thread 0 computes final M scaling
    float scale = 1.0f;
    // We want to return this to all threads.
    
    // Wait, optimization: We only need thread 0 to compute M if we broadcast it?
    // But this function returns float M. Every thread needs it?
    // compute_christoffel_force takes 'scale_M'. 
    // Usually passed as scalar to all threads? No, it's called per block/thread.
    
    if (tid == 0) {
        float total = s_pot + (b_potential ? b_potential[0] : 0.0f);
        float potential = sigmoidf_device(total);
        
        // Continuous transition for differentiability:
        // Use a steep sigmoid to mimic the threshold while remaining differentiable.
        const float stiffness = 20.0f; // Scale for the transition
        float activation = sigmoidf_device(stiffness * (potential - threshold));
        scale = 1.0f + (strength - 1.0f) * activation;
        
        s_pot = scale; // Reuse shared
    }
    __syncthreads();
    
    return s_pot;
}


// Global Dispatcher
__device__ __forceinline__ void compute_christoffel_force(
    float* __restrict__ force_out, 
    const float* __restrict__ v,
    const float* __restrict__ x,
    const float* __restrict__ U,   
    const float* __restrict__ W,   
    float* s_h,                    
    int dim,
    int rank,
    int tid,
    int topology,
    float scale_M,
    float R_val = 2.0f,
    float r_val = 1.0f
) {
    if (topology == TORUS) {
        compute_christoffel_torus(force_out, v, x, dim, tid, R_val, r_val, scale_M);
    } else {
        compute_christoffel_low_rank(force_out, v, U, W, s_h, dim, rank, tid, scale_M);
    }
}

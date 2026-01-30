#pragma once

// ==========================================
// ADAPTER SHIM: LEGACY -> MODULAR PHYSICS
// ==========================================
// This file routes old monolithic calls to the new 
// gfn/cuda/include/forces.cuh architecture.
// ==========================================

#include "forces.cuh"
#include "manifold_metrics.cuh"

// Include primitives for legacy calls that used waprReduceSum directly?
// forces.cuh includes primitives.

// --------------------------------------------------------
// Legacy: Compute Friction (Direct Mapping)
// --------------------------------------------------------
__device__ __forceinline__ void compute_friction_device(
    float* s_mu,          
    const float* s_x,     
    const float* s_u,     // Unused in new logic
    const float* W_x,     
    const float* W_u,     // Unused
    const float* b_gate,  
    int dim,
    int tid,
    int topology          
) {
    compute_friction_coeff(s_mu, s_x, W_x, b_gate, dim, tid, topology);
}

// --------------------------------------------------------
// Legacy: Christoffel Device (Complex Mapping)
// --------------------------------------------------------
__device__ __forceinline__ void christoffel_device(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma_out, 
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    int topology,
    // Shared Memory Pointers from Legacy Kernels
    float* s_h,          
    double* s_E,         
    double* s_P,         
    float* s_M,
    float R_val = 2.0f,
    float r_val = 1.0f
) {
    const int tid = threadIdx.x;
    
    // 1. Compute M (Combined Plasticity + Singularity)
    // Legacy logic separate from forces.cuh which just takes 'scale_M'.
    // We replicate the 'Policy' logic here, calling primitives.
    
    if (tid == 0) { *s_M = 1.0f; *s_P = 0.0; } // Init
    __syncthreads();

    float m_total = 1.0f;
    
    // A. Plasticity (Kinetic Energy)
    float m_plast = compute_plasticity_scale(s_E, v, dim, tid, plasticity);
    // Note: compute_plasticity_scale returns 1.0 + ... 
    // It handles the reduction internally using s_E buffer.
    
    // B. Singularity (Potential Energy)
    if (use_active && x != nullptr && V_w != nullptr) {
        // We need to reduce P. forces.cuh doesn't expose a "Potential Reduction" func.
        // We implement it inline here using Primitives.
        float p_P = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x) {
             p_P += topology_potential_grad(x[i], V_w[i], topology); // Usually V_w is grad? Or V?
             // Legacy: `(topology==1)? sin(x)*V_w : x*V_w`.
             // topology_potential_grad is `cos(x)*V`. Wait. 
             // Legacy P accumulation (Line 114 original): `sin(x[i]) * V_w[i]`.
             // My manifold_metrics `topology_potential_grad` is `cos` (derivative).
             // Potential P is usually integral. `V_w` is weights?
             // Legacy: `p_P += sinf(x[i]) * V_w[i]`.
             // If V_w is "Direction Vector to Singularity", P is proj.
             // Let's implement Legacy logic explicitly.
             if (topology == TORUS) p_P += sinf(x[i]) * V_w[i];
             else p_P += x[i] * V_w[i];
        }
        p_P = warpReduceSum(p_P);
        if (tid == 0) *s_P = 0.0; __syncthreads();
        if ((tid & 31) == 0) atomicAdd(s_P, (double)p_P);
        __syncthreads();
        
        // Compute Soft Singularity Multiplier
        if (tid == 0) {
            float pot_val = (float)*s_P;
            // Sigmoid[-20, 20] logic from legacy
            float sig_pot = 1.0f / (1.0f + expf(-fminf(fmaxf(pot_val, -20.0f), 20.0f)));
            float soft_m = 1.0f / (1.0f + expf(-10.0f * (sig_pot - sing_thresh)));
            float m_sing = (1.0f + (sing_strength - 1.0f) * soft_m);
            *s_M = m_sing; // Store part
        }
    }
    __syncthreads();
    
    // Combine
    // Legacy logic: m *= (1 + plasticity...); m *= m_sing;
    m_total = m_plast * (*s_M);
    
    // 2. Call New Engine
    compute_christoffel_force(
        gamma_out, 
        v, x, U, W, 
        s_h, 
        dim, rank, tid, 
        topology,
        m_total,
        R_val,
        r_val
    );
}

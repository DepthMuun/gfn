#include "../../include/forces.cuh"
#include "../../include/gradients.cuh"

#define BLOCK_SIZE 512

// --- HELPER FUNCTIONS (Layer Specific) ---


__device__ void rmsnorm_backward_device(float* s_gx, const float* s_x, int dim, int tid, float eps = 1e-6f) {
    float sum_sq = 0.0f, dot_gy = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) sum_sq += s_x[i] * s_x[i];
    sum_sq = warpReduceSum(sum_sq);
    __shared__ float s_rms_val;
    if (tid == 0) s_rms_val = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_rms_val, sum_sq);
    __syncthreads();
    float inv_rms = rsqrtf(s_rms_val / (float)dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) dot_gy += s_gx[i] * s_x[i] * inv_rms;
    dot_gy = warpReduceSum(dot_gy);
    __shared__ float s_dot;
    if (tid == 0) s_dot = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_dot, dot_gy);
    __syncthreads();
    float mean_dot = s_dot / (float)dim;
    for (int i = tid; i < dim; i += blockDim.x) s_gx[i] = inv_rms * (s_gx[i] - (s_x[i] * inv_rms) * mean_dot);
    __syncthreads();
}


__global__ void recurrent_manifold_backward_kernel(
    const float* __restrict__ g_x_seq, const float* __restrict__ g_x_f, const float* __restrict__ g_v_f,
    const float* __restrict__ x_init, const float* __restrict__ v_init,
    const float* __restrict__ x_seq, const float* __restrict__ v_seq,
    const float* __restrict__ forces, const float* __restrict__ U_s, const float* __restrict__ W_s,
    float* __restrict__ g_x0, float* __restrict__ g_v0, float* __restrict__ g_forces,
    float* __restrict__ g_U, float* __restrict__ g_W,
    const float* __restrict__ Wmx, const float* __restrict__ Wmv, float* __restrict__ g_Wmx, float* __restrict__ g_Wmv,
    const float* __restrict__ Wf_s, const float* __restrict__ Wi_s, const float* __restrict__ bf_s,
    float* __restrict__ g_Wf, float* __restrict__ g_Wi, float* __restrict__ g_bf,
    const float* __restrict__ Wp_s, const float* __restrict__ bp_s,
    float* __restrict__ g_Wp, float* __restrict__ g_bp,
    const int batch, const int seq_len, const int dim, const int rank, const int num_layers, const int num_heads,
    const float dt, const float* __restrict__ dt_scales, const float* __restrict__ forget_rates, float* __restrict__ g_fr,
    const float plasticity, const float sing_thresh, const float sing_strength, const int topology,
    const float R_val, const float r_val
) {
    extern __shared__ float s_mem[];
    const int dim_h = dim / num_heads, rank_h = rank / num_heads;
    
    // Core Gradients
    float* s_gx = s_mem;
    float* s_gv = s_gx + dim;
    
    // Intra-Step Trace (Checkpoint ALL layers for current timestep)
    float* s_trace_x = s_gv + dim;
    float* s_trace_v = s_trace_x + (num_layers + 1) * dim;
    
    // Step Workspace
    float* s_ga = s_trace_v + (num_layers + 1) * dim;
    float* s_mu = s_ga + dim;
    float* s_dm = s_mu + dim;
    float* s_t1 = s_dm + dim;
    float* s_t2 = s_t1 + dim;
    float* s_h  = s_t2 + dim;
    
    // Double align for energy/gradients
    size_t f_off = (2 * dim + 2 * (num_layers + 1) * dim + 6 * dim + rank + 32) * sizeof(float);
    if (f_off % 8 != 0) f_off += 4;
    double* s_db = (double*)((char*)s_mem + f_off);
    double* s_grad_h_dbl = s_db + 8;

    const int b = blockIdx.x, tid = threadIdx.x;
    if (b >= batch) return;
    const float d_sc = 1.0f / sqrtf((float)num_layers);

    // Initial Gradients from Final Step
    for (int i = tid; i < dim; i += blockDim.x) { 
        s_gx[i] = g_x_f ? g_x_f[b * dim + i] : 0.0f; 
        s_gv[i] = g_v_f ? g_v_f[b * dim + i] : 0.0f; 
    }
    __syncthreads();
    
    for (int t = seq_len - 1; t >= 0; t--) {
        // 0. Inject Sequence Gradient
        if (g_x_seq) { for (int i = tid; i < dim; i += blockDim.x) s_gx[i] += g_x_seq[(b*seq_len+t)*dim+i]; __syncthreads(); }

        // 1. FORWARD REPLAY (Populate Trace)
        for (int i = tid; i < dim; i += blockDim.x) {
            float xi, vi;
            if (t > 0) { xi = x_seq[(b*seq_len + t - 1)*dim + i]; vi = v_seq[(b*seq_len + t - 1)*dim + i]; }
            else { xi = x_init[b*dim + i]; vi = v_init[b*dim + i]; }
            s_trace_x[i] = xi; s_trace_v[i] = vi; 
        }
        __syncthreads();

        for (int l = 0; l < num_layers; l++) {
            float* s_cx = s_trace_x + l*dim, *s_cv = s_trace_v + l*dim;
            float* s_nx = s_trace_x + (l+1)*dim, *s_nv = s_trace_v + (l+1)*dim;
            
            for (int h = 0; h < num_heads; h++) {
                int off = h * dim_h; long long lhi = l * num_heads + h;
                float eff_dt = dt * (dt_scales ? dt_scales[h] : 1.0f) * d_sc, hdt = 0.5f * eff_dt;
                int wxs = (topology == TORUS) ? (2 * dim_h) : dim_h;
                
                float* s_xh = s_cx + off, *s_vh = s_cv + off;
                float* s_nxh = s_nx + off, *s_nvh = s_nv + off;
                float* s_mh = s_mu + off;
                
                // M_total calculation (Mirror Forward)
                float M = 1.0f;
                compute_friction_coeff(s_mh, s_xh, Wf_s + lhi*dim_h*wxs, bf_s + lhi*dim_h, dim_h, tid, topology);
                M *= compute_plasticity_scale((float*)s_db, s_vh, dim_h, tid, plasticity);
                if (Wp_s && bp_s) M *= compute_singularity_scale(s_xh, Wp_s + lhi*wxs, bp_s + lhi, dim_h, tid, topology, sing_thresh, sing_strength);

                for(int i=tid; i<dim_h; i+=blockDim.x) s_nvh[i] = s_vh[i] * expf(-s_mh[i] * hdt);
                __syncthreads();
                
                compute_christoffel_force(s_ga + off, s_nvh, s_xh, U_s + lhi*dim_h*rank_h, W_s + lhi*dim_h*rank_h, s_h + h*rank_h, dim_h, rank_h, tid, topology, M, R_val, r_val);
                for(int i=tid; i<dim_h; i+=blockDim.x) {
                    s_nvh[i] += hdt * (forces[(b*seq_len+t)*dim+off+i] - s_ga[off+i]);
                    s_nxh[i] = apply_boundary(s_xh[i] + eff_dt * s_nvh[i], topology);
                }
                __syncthreads();
                
                // Second Half Replay (Implicitly simplified for now, as in v5.1 forward)
                compute_friction_coeff(s_mh, s_nxh, Wf_s + lhi*dim_h*wxs, bf_s + lhi*dim_h, dim_h, tid, topology);
                float M2 = compute_plasticity_scale((float*)s_db, s_nvh, dim_h, tid, plasticity);
                if (Wp_s && bp_s) M2 *= compute_singularity_scale(s_nxh, Wp_s + lhi*wxs, bp_s + lhi, dim_h, tid, topology, sing_thresh, sing_strength);
                
                compute_christoffel_force(s_ga + off, s_nvh, s_nxh, U_s + lhi*dim_h*rank_h, W_s + lhi*dim_h*rank_h, s_h + h*rank_h, dim_h, rank_h, tid, topology, M2, R_val, r_val);
                for(int i=tid; i<dim_h; i+=blockDim.x) s_nvh[i] += hdt * (forces[(b*seq_len+t)*dim+off+i] - s_ga[off+i]);
                __syncthreads();
                apply_friction_damping(s_nvh, s_mh, dim_h, tid, hdt);
            }
            if (num_heads > 1 && Wmx && l < num_layers - 1) head_mixing_device(s_nx, s_nv, Wmx, Wmv, s_t1, s_t2, dim, tid, topology);
            tanh_bounding_device(s_nv, dim, tid, 100.0f);
        }

        // 2. BACKWARD SWEEP
        for (int l = num_layers - 1; l >= 0; l--) {
            // Tanh bounding backward on final v of this layer
            tanh_bounding_backward_device(s_gv, s_trace_v + (l+1)*dim, dim, tid, 100.0f);
            
            if (num_heads > 1 && Wmx && l < num_layers - 1) 
                head_mixing_backward_device(s_gx, s_gv, s_trace_x + (l+1)*dim, s_trace_v + (l+1)*dim, Wmx, Wmv, g_Wmx, g_Wmv, s_t1, s_t2, dim, tid, topology);

            for (int h = num_heads - 1; h >= 0; h--) {
                int off = h * dim_h; long long lhi = l * num_heads + h;
                float eff_dt = dt * (dt_scales ? dt_scales[h] : 1.0f) * d_sc, hdt = 0.5f * eff_dt;
                int wxs = (topology == TORUS) ? (2 * dim_h) : dim_h;

                float* s_xh = s_trace_x + l*dim + off;
                float* s_vh_init = s_trace_v + l*dim + off;
                float* s_nxh = s_trace_x + (l+1)*dim + off;
                float* s_nvh_final = s_trace_v + (l+1)*dim + off;

                float* s_gxh = s_gx + off, *s_gvh = s_gv + off;

                // --- RE-RUN FORWARD PASS FOR THIS HEAD (To get intermediate states) ---
                // We need v_half (after kick 1) and mu_0, mu_1, M_1, M_2
                // Since shared memory is tight, we recompute.
                
                // mu_0, M_1 (at x_0, v_init)
                compute_friction_coeff(s_mu + off, s_xh, Wf_s + lhi*dim_h*wxs, bf_s + lhi*dim_h, dim_h, tid, topology);
                float M1 = compute_plasticity_scale((float*)s_db, s_vh_init, dim_h, tid, plasticity);
                if (Wp_s && bp_s) M1 *= compute_singularity_scale(s_xh, Wp_s + lhi*wxs, bp_s + lhi, dim_h, tid, topology, sing_thresh, sing_strength);
                
                // Recompute v_half and store in s_t1 temporarily
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    s_t1[off+i] = s_vh_init[i] * expf(-s_mu[off+i] * hdt);
                }
                __syncthreads();
                
                compute_christoffel_force(s_ga + off, s_t1 + off /* junk buffer */, s_xh, U_s+lhi*dim_h*rank_h, W_s+lhi*dim_h*rank_h, s_h+h*rank_h, dim_h, rank_h, tid, topology, M1, R_val, r_val);
                for (int i = tid; i < dim_h; i += blockDim.x) s_t1[off+i] += hdt * (forces[(b*seq_len+t)*dim+off+i] - s_ga[off+i]);
                __syncthreads();

                // mu_1, M_2 (at x_1, v_half)
                compute_friction_coeff(s_mu + off, s_nxh, Wf_s + lhi*dim_h*wxs, bf_s + lhi*dim_h, dim_h, tid, topology);
                float M2 = compute_plasticity_scale((float*)s_db, s_t1+off, dim_h, tid, plasticity);
                if (Wp_s && bp_s) M2 *= compute_singularity_scale(s_nxh, Wp_s + lhi*wxs, bp_s + lhi, dim_h, tid, topology, sing_thresh, sing_strength);

                // --- BACKWARD PASS ---
                
                // 1. Final Friction (mu_1)
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    float damp = expf(-s_mu[off+i] * hdt);
                    float gvn = s_gvh[i];
                    s_gvh[i] = gvn * damp;
                    // dL/dMu_1
                    s_ga[off+i] = -gvn * s_nvh_final[i] * hdt; 
                }
                __syncthreads();
                compute_friction_backward(s_ga+off, s_nxh, Wf_s+lhi*dim_h*wxs, bf_s+lhi*dim_h, g_Wf?g_Wf+lhi*dim_h*wxs:nullptr, g_bf?g_bf+lhi*dim_h:nullptr, s_gxh, dim_h, tid, topology);

                // 2. Kick 2 (at x_1, v_half, M_2)
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    s_dm[off+i] = -s_gvh[i] * hdt; // Gradient for Christoffel force
                    if (g_forces) atomicAdd(&g_forces[(b*seq_len+t)*dim+off+i], s_gvh[i] * hdt);
                }
                __syncthreads();
                
                __shared__ float s_gM2;
                if (tid == 0) s_gM2 = 0.0f; __syncthreads();
                compute_christoffel_backward(s_dm+off, s_t1+off, s_nxh, U_s+lhi*dim_h*rank_h, W_s+lhi*dim_h*rank_h, s_gvh, s_gxh, g_U?g_U+lhi*dim_h*rank_h:nullptr, g_W?g_W+lhi*dim_h*rank_h:nullptr, &s_gM2, s_h+h*rank_h, s_grad_h_dbl, dim_h, rank_h, tid, topology, plasticity, M2, R_val, r_val);
                __syncthreads();
                
                // Propagate s_gM2 to singularity and plasticity at x_1, v_half
                compute_singularity_backward(s_gM2, s_nxh, Wp_s+lhi*wxs, bp_s+lhi, g_Wp?g_Wp+lhi*wxs:nullptr, g_bp?g_bp+lhi:nullptr, s_gxh, dim_h, tid, topology, sing_thresh, sing_strength);
                compute_plasticity_backward(s_gM2, s_t1+off, dim_h, tid, plasticity, nullptr, s_gvh, (double*)s_db);

                // 3. Drift (x_1 = x_0 + dt * v_half)
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    s_gvh[i] += s_gxh[i] * eff_dt;
                }
                __syncthreads();
                
                // 4. Kick 1 (at x_0, v_init, M_1)
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    s_dm[off+i] = -s_gvh[i] * hdt;
                    if (g_forces) atomicAdd(&g_forces[(b*seq_len+t)*dim+off+i], s_gvh[i] * hdt);
                }
                __syncthreads();
                
                __shared__ float s_gM1;
                if (tid == 0) s_gM1 = 0.0f; __syncthreads();
                compute_christoffel_backward(s_dm+off, s_vh_init, s_xh, U_s+lhi*dim_h*rank_h, W_s+lhi*dim_h*rank_h, s_gvh, s_gxh, g_U?g_U+lhi*dim_h*rank_h:nullptr, g_W?g_W+lhi*dim_h*rank_h:nullptr, &s_gM1, s_h+h*rank_h, s_grad_h_dbl, dim_h, rank_h, tid, topology, plasticity, M1, R_val, r_val);
                __syncthreads();

                compute_singularity_backward(s_gM1, s_xh, Wp_s+lhi*wxs, bp_s+lhi, g_Wp?g_Wp+lhi*wxs:nullptr, g_bp?g_bp+lhi:nullptr, s_gxh, dim_h, tid, topology, sing_thresh, sing_strength);
                compute_plasticity_backward(s_gM1, s_vh_init, dim_h, tid, plasticity, nullptr, s_gvh, (double*)s_db);

                // 5. Initial Friction (mu_0)
                compute_friction_coeff(s_mu+off, s_xh, Wf_s + lhi*dim_h*wxs, bf_s + lhi*dim_h, dim_h, tid, topology);
                for (int i = tid; i < dim_h; i += blockDim.x) {
                    float damp = expf(-s_mu[off+i] * hdt);
                    float gv_after = s_gvh[i];
                    s_gvh[i] = gv_after * damp;
                    s_ga[off+i] = -gv_after * s_t1[off+i] * hdt;
                }
                __syncthreads();
                compute_friction_backward(s_ga+off, s_xh, Wf_s+lhi*dim_h*wxs, bf_s+lhi*dim_h, g_Wf?g_Wf+lhi*dim_h*wxs:nullptr, g_bf?g_bf+lhi*dim_h:nullptr, s_gxh, dim_h, tid, topology);
                __syncthreads();
            }
        }
    }
    for (int i = tid; i < dim; i += blockDim.x) { if (g_x0) g_x0[b * dim + i] = s_gx[i]; if (g_v0) g_v0[b * dim + i] = s_gv[i]; }
}

extern "C" void launch_recurrent_manifold_backward(
    const float* g_xs, const float* g_xf, const float* g_vf, const float* xi, const float* vi, const float* xs, const float* vs, const float* f, const float* Us, const float* Ws,
    float* g_x0, float* g_v0, float* g_f, float* g_U, float* g_W,
    const float* Wmx, const float* Wmv, float* g_Wmx, float* g_Wmv,
    const float* Wf_s, const float* Wi_s, const float* bf_s, float* g_Wf, float* g_Wi, float* g_bf,
    const float* Wp_s, const float* bp_s, float* g_Wp, float* g_bp, // NEW
    int batch, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* g_fr,
    float plasticity, float sing_thresh, float sing_strength, int topology, 
    float R_val, float r_val, cudaStream_t stream
) {
    const int shared_bytes = (12 * dim + 2 * (num_layers + 1) * dim + rank + 512) * sizeof(float) + (rank + 16) * sizeof(double);
    recurrent_manifold_backward_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        g_xs, g_xf, g_vf, xi, vi, xs, vs, f, Us, Ws, g_x0, g_v0, g_f, g_U, g_W, Wmx, Wmv, g_Wmx, g_Wmv,
        Wf_s, Wi_s, bf_s, g_Wf, g_Wi, g_bf, Wp_s, bp_s, g_Wp, g_bp, batch, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, g_fr, plasticity, sing_thresh, sing_strength, topology, R_val, r_val
    );
}

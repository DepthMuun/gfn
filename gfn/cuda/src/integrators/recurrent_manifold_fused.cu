#include "../../include/forces.cuh"

#define BLOCK_SIZE 512

// RECURRENT MANIFOLD FUSED KERNEL v5.1 (Intra-Step REPLAY SYNCHRONIZED)

__global__ void recurrent_manifold_fused_kernel(
    float* x_state, float* v_state, const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* v_out_seq, float* reg_loss, const float* W_mix_x, const float* W_mix_v,
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    const float* W_potential_stack, const float* b_potential_stack,
    const int batch, const int seq_len, const int dim, const int rank, const int num_layers, const int num_heads,
    const float dt, const float* __restrict__ dt_scales, const float* __restrict__ forget_rates,
    const float plasticity, const float sing_thresh, const float sing_strength, 
    const int topology, const float R_val, const float r_val
) {
    extern __shared__ float s_mem_f[];
    const int dim_per_head = dim / num_heads;
    const int head_rank = rank / num_heads;
    
    // Shared Memory Layout
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_gamma = s_v + dim;
    float* s_force = s_gamma + dim;
    float* s_temp = s_force + dim;
    float* s_temp2 = s_temp + dim;
    float* s_h = s_temp2 + dim;
    
    // Double align
    size_t float_offset = (6 * dim + num_heads * head_rank) * sizeof(float);
    if (float_offset % 8 != 0) float_offset += 4;
    double* s_buf_energy = (double*)((char*)s_mem_f + float_offset);
    
    const int b = blockIdx.x, tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = x_state[b * dim + i]; s_v[i] = v_state[b * dim + i]; }
    __syncthreads();
    
    const float depth_scale = 1.0f / sqrtf((float)num_layers);

    for (int t = 0; t < seq_len; t++) {
        // Load Input Force
        for (int i = tid; i < dim; i += blockDim.x) s_force[i] = forces[(b * seq_len + t) * dim + i];
        __syncthreads();
        
        for (int l = 0; l < num_layers; l++) {
            for (int h = 0; h < num_heads; h++) {
                int head_offset = h * dim_per_head, layer_head_idx = l * num_heads + h;
                const float h_dt_s = dt_scales ? dt_scales[h] : 1.0f;
                const float eff_dt = dt * h_dt_s * depth_scale, half_dt = 0.5f * eff_dt;
                
                float* s_v_h = s_v + head_offset;
                float* s_x_h = s_x + head_offset; 
                float* s_gamma_h = s_gamma + head_offset;
                float* s_mu_h = s_temp + head_offset; 
                
                const float* U_h = U_stack + (layer_head_idx * dim_per_head * head_rank);
                const float* W_h = W_stack + (layer_head_idx * dim_per_head * head_rank);
                
                int w_x_s = (topology == TORUS) ? (2 * dim_per_head) : dim_per_head;
                const float* W_f_h = W_forget_stack + (layer_head_idx * dim_per_head * w_x_s);
                const float* b_f_h = b_forget_stack + (layer_head_idx * dim_per_head);
                const float* s_f_h = s_force + head_offset;
                float* s_h_s = s_h + h * head_rank; 
                
                // Singularity Support (Black Holes)
                float M_total = 1.0f;
                
                // 1. Friction & Clutch
                compute_friction_coeff(s_mu_h, s_x_h, W_f_h, b_f_h, dim_per_head, tid, topology);
                
                // 2. Plasticity
                float M_plast = compute_plasticity_scale(s_buf_energy, s_v_h, dim_per_head, tid, plasticity);
                M_total *= M_plast;
                
                // 3. Singularity
                if (W_potential_stack && b_potential_stack) {
                    const float* W_pot_h = W_potential_stack + (layer_head_idx * w_x_s); // 1 x In
                    const float* b_pot_h = b_potential_stack + layer_head_idx; // 1
                    float M_sing = compute_singularity_scale(s_x_h, W_pot_h, b_pot_h, dim_per_head, tid, topology, sing_thresh, sing_strength);
                    M_total *= M_sing;
                }

                // 4. Kick 1 (Half) + Friction 1 (Half)
                apply_friction_damping(s_v_h, s_mu_h, dim_per_head, tid, half_dt); // Pre-damp
                
                compute_christoffel_force(s_gamma_h, s_v_h, s_x_h, U_h, W_h, s_h_s, dim_per_head, head_rank, tid, topology, M_total, R_val, r_val);
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_v_h[i] += half_dt * (s_f_h[i] - s_gamma_h[i]);
                __syncthreads();
                
                // 5. Drift (Full)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_x_h[i] += eff_dt * s_v_h[i];
                    s_x_h[i] = apply_boundary(s_x_h[i], topology);
                }
                __syncthreads();
                
                // 6. Recalculate State-Dependent Terms
                compute_friction_coeff(s_mu_h, s_x_h, W_f_h, b_f_h, dim_per_head, tid, topology);
                float M_plast2 = compute_plasticity_scale(s_buf_energy, s_v_h, dim_per_head, tid, plasticity);
                M_total = M_plast2; // Reset and re-apply Singularity at new X
                
                 if (W_potential_stack && b_potential_stack) {
                    const float* W_pot_h = W_potential_stack + (layer_head_idx * w_x_s);
                    const float* b_pot_h = b_potential_stack + layer_head_idx;
                    float M_sing2 = compute_singularity_scale(s_x_h, W_pot_h, b_pot_h, dim_per_head, tid, topology, sing_thresh, sing_strength);
                    M_total *= M_sing2;
                }
                
                // 7. Kick (Second Half)
                compute_christoffel_force(s_gamma_h, s_v_h, s_x_h, U_h, W_h, s_h_s, dim_per_head, head_rank, tid, topology, M_total, R_val, r_val);
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_v_h[i] += half_dt * (s_f_h[i] - s_gamma_h[i]);
                __syncthreads();
                
                // Post-Damp
                apply_friction_damping(s_v_h, s_mu_h, dim_per_head, tid, half_dt);
                __syncthreads();
            } 
            
            if (num_heads > 1 && W_mix_x != nullptr && l < num_layers - 1) head_mixing_device(s_x, s_v, W_mix_x, W_mix_v, s_temp, s_temp2, dim, tid, topology);
            tanh_bounding_device(s_v, dim, tid, 100.0f);
        } 
        __syncthreads();
        if (x_out_seq) for (int i = tid; i < dim; i += blockDim.x) x_out_seq[(b * seq_len + t) * dim + i] = s_x[i];
        if (v_out_seq) for (int i = tid; i < dim; i += blockDim.x) v_out_seq[(b * seq_len + t) * dim + i] = s_v[i];
        __syncthreads();
    }
    for (int i = tid; i < dim; i += blockDim.x) { x_state[b * dim + i] = s_x[i]; v_state[b * dim + i] = s_v[i]; }
}

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state, const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* v_out_seq, float* reg_loss, int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v, 
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    const float* W_potential_stack, const float* b_potential_stack, // NEW
    int topology, float R_val, float r_val, cudaStream_t stream
) {
    const int shared_bytes = (6 * dim + rank + 128) * sizeof(float) + 16 * sizeof(double);
    cudaMemsetAsync(reg_loss, 0, batch * sizeof(float), stream);
    recurrent_manifold_fused_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_state, v_state, forces, U_stack, W_stack, x_out_seq, v_out_seq, reg_loss, W_mix_x, W_mix_v,
        W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack,
        batch, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, plasticity, sing_thresh, sing_strength, topology, R_val, r_val
    );
}

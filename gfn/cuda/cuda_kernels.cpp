
#ifdef _WIN32
#define NOMINMAX
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CRITICAL FIX for CUDA + PyTorch conflict:
#include <ATen/cuda/CUDAContext.h>

// Forward declarations of CUDA launchers (Raw C interface)
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology, float R_val, float r_val, cudaStream_t stream
);

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology, float R_val, float r_val, cudaStream_t stream
);

extern "C" void launch_reactive_christoffel_forward(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, float R_val, float r_val, cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* v_out_seq, float* reg_loss,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v,
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    const float* W_potential_stack, const float* b_potential_stack,
    int topology, float R_val, float r_val,
    cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_backward(
    const float* grad_x_seq, const float* grad_x_final, const float* grad_v_final,
    const float* x_init, const float* v_init,
    const float* x_seq, const float* v_seq,
    const float* forces, const float* U_stack, const float* W_stack,
    float* grad_x_init, float* grad_v_init, float* grad_forces,
    float* grad_U, float* grad_W,
    const float* W_mix_x, const float* W_mix_v,
    float* grad_W_mix_x, float* grad_W_mix_v,
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    float* grad_W_forget, float* grad_W_input, float* grad_b_forget,
    const float* W_potential_stack, const float* b_potential_stack,
    float* grad_W_potential, float* grad_b_potential,
    int batch_total, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* grad_forget_rates,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, float R_val, float r_val,
    cudaStream_t stream
);

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
);

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
);

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
);

extern "C" void launch_dormand_prince_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    const float* W_forget, const float* b_forget,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps, int topology, float plasticity,
    float R_val, float r_val,
    cudaStream_t stream
);

extern "C" void launch_leapfrog_backward(const float* gx_n, const float* gv_n, const float* x, const float* v, const float* f, const float* U, const float* W, float* gx, float* gv, float* gf, float* gU, float* gW, int batch, int dim, int rank, float dt, float dt_s, int steps, int topology, float R_val, float r_val, cudaStream_t stream);
extern "C" void launch_euler_fused(const float* x, const float* v, const float* f, const float* U, const float* W, float* x_new, float* v_new, float dt, float dt_scale, int batch, int dim, int rank, int steps, int topology, float R_val, float r_val, cudaStream_t stream);
extern "C" void launch_verlet_fused(const float* x, const float* v, const float* f, const float* U, const float* W, float* x_new, float* v_new, float dt, float dt_scale, int batch, int dim, int rank, int steps, int topology, float R_val, float r_val, cudaStream_t stream);
extern "C" void launch_heun_fused(const float* x, const float* v, const float* f, const float* U, const float* W, float* x_new, float* v_new, float dt, float dt_scale, int batch, int dim, int rank, int steps, int topology, float R_val, float r_val, cudaStream_t stream);

// --- PyTorch Wrappers (Tensor safe) ---

torch::Tensor christoffel_fused_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology, float R, float r) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (x_ptr != nullptr && V_ptr != nullptr);
    launch_christoffel_fused(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, topology, R, r, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> christoffel_backward_cuda(torch::Tensor grad_gamma, torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology, float R, float r) {
    auto grad_v = torch::zeros_like(v); auto grad_U = torch::zeros_like(U); auto grad_W = torch::zeros_like(W);
    torch::Tensor grad_x, grad_V_w; const float *x_ptr = nullptr, *V_ptr = nullptr; float *gx_ptr = nullptr, *gV_ptr = nullptr;
    bool use_active = false;
    if (x.numel() > 0) { grad_x = torch::zeros_like(x); x_ptr = x.data_ptr<float>(); gx_ptr = grad_x.data_ptr<float>(); use_active = true; }
    if (V_w.numel() > 0) { grad_V_w = torch::zeros_like(V_w); V_ptr = V_w.data_ptr<float>(); gV_ptr = grad_V_w.data_ptr<float>(); use_active = true; }
    if (plasticity != 0.0f) use_active = true;
    launch_christoffel_backward(grad_gamma.data_ptr<float>(), v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_ptr, V_ptr, grad_v.data_ptr<float>(), grad_U.data_ptr<float>(), grad_W.data_ptr<float>(), gx_ptr, gV_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, topology, R, r, at::cuda::getCurrentCUDAStream());
    return {grad_v, grad_U, grad_W, grad_x, grad_V_w};
}

// GENERIC INTEGRATOR WRAPPER
std::vector<torch::Tensor> generic_integrator_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W,
    torch::Tensor W_forget, torch::Tensor b_forget,
    float dt, torch::Tensor dt_scales, int steps, int topology, float plasticity,
    float R_val, float r_val,
    int type // 0=Leapfrog, 1=Yoshida, 2=RK4, 3=DormandPrince
) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    const float* W_f_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<float>() : nullptr;
    const float* b_f_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<float>() : nullptr;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int batch = x.size(0); int dim = x.size(1); int rank = U.size(1);

    if (type == 0) {
        const float* dt_s_ptr = (dt_scales.numel() > 0 && dt_scales.numel() == batch) ? dt_scales.data_ptr<float>() : nullptr;
        launch_leapfrog_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), W_f_ptr, b_f_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_s_ptr, batch, dim, rank, steps, topology, plasticity, R_val, r_val, stream);
    }
    else if (type == 1) {
         float s = (dt_scales.numel() >= 1) ? dt_scales.reshape({-1})[0].item<float>() : 1.0f;
         launch_yoshida_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), W_f_ptr, b_f_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, s, batch, dim, rank, steps, topology, plasticity, R_val, r_val, stream);
    }
    else if (type == 2) {
         float s = (dt_scales.numel() >= 1) ? dt_scales.reshape({-1})[0].item<float>() : 1.0f;
         launch_rk4_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), W_f_ptr, b_f_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, s, batch, dim, rank, steps, topology, plasticity, R_val, r_val, stream);
    }
    else if (type == 3) {
         float s = (dt_scales.numel() >= 1) ? dt_scales.reshape({-1})[0].item<float>() : 1.0f;
         launch_dormand_prince_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), W_f_ptr, b_f_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, s, batch, dim, rank, steps, topology, plasticity, R_val, r_val, stream);
    }
    
    return {x_new, v_new};
}

std::vector<torch::Tensor> leapfrog_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor Wf, torch::Tensor bf, float dt, torch::Tensor dt_scale, int steps, int topology, float plasticity, float R, float r) {
    return generic_integrator_cuda(x, v, f, U, W, Wf, bf, dt, dt_scale, steps, topology, plasticity, R, r, 0);
}

std::vector<torch::Tensor> leapfrog_backward_cuda(torch::Tensor g_xn, torch::Tensor g_vn, torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps, int topology, float R, float r) {
    auto g_x = torch::zeros_like(x); auto g_v = torch::zeros_like(v); auto g_f = torch::zeros_like(f); auto g_U = torch::zeros_like(U); auto g_W = torch::zeros_like(W);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    float* gf_ptr = (f.numel() > 0) ? g_f.data_ptr<float>() : nullptr;
    launch_leapfrog_backward(g_xn.data_ptr<float>(), g_vn.data_ptr<float>(), x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), g_x.data_ptr<float>(), g_v.data_ptr<float>(), gf_ptr, g_U.data_ptr<float>(), g_W.data_ptr<float>(), x.size(0), x.size(1), U.size(1), dt, dt_scale, steps, topology, R, r, at::cuda::getCurrentCUDAStream());
    return {g_x, g_v, g_f, g_U, g_W};
}

std::vector<torch::Tensor> euler_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps, int topology, float R, float r) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    launch_euler_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, topology, R, r, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

torch::Tensor reactive_christoffel_forward_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology, float R, float r) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    launch_reactive_christoffel_forward(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, topology, R, r, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> recurrent_manifold_fused_cuda(
    torch::Tensor x_state, torch::Tensor v_state, torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v,
    torch::Tensor W_forget_stack, torch::Tensor W_input_stack, torch::Tensor b_forget_stack,
    torch::Tensor W_potential_stack, torch::Tensor b_potential_stack,
    int topology, float R, float r
) {
    int batch = x_state.size(0); int dim = x_state.size(1); int seq_len = forces.size(1);
    int rank = U_stack.size(2); int num_layers = U_stack.size(0) / num_heads;
    auto x_out_seq = torch::empty({batch, seq_len, dim}, x_state.options());
    auto v_out_seq = torch::empty({batch, seq_len, dim}, v_state.options());
    auto reg_loss = torch::zeros({batch}, x_state.options());
    
    const float* mx = (mix_x.numel() > 0) ? mix_x.data_ptr<float>() : nullptr;
    const float* mv = (mix_v.numel() > 0) ? mix_v.data_ptr<float>() : nullptr;
    const float* wf = (W_forget_stack.numel() > 0) ? W_forget_stack.data_ptr<float>() : nullptr;
    const float* wi = (W_input_stack.numel() > 0) ? W_input_stack.data_ptr<float>() : nullptr;
    const float* bf = (b_forget_stack.numel() > 0) ? b_forget_stack.data_ptr<float>() : nullptr;
    
    const float* wp = (W_potential_stack.numel() > 0) ? W_potential_stack.data_ptr<float>() : nullptr;
    const float* bp = (b_potential_stack.numel() > 0) ? b_potential_stack.data_ptr<float>() : nullptr;

    launch_recurrent_manifold_fused(x_state.data_ptr<float>(), v_state.data_ptr<float>(), forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(), x_out_seq.data_ptr<float>(), v_out_seq.data_ptr<float>(), reg_loss.data_ptr<float>(), batch, seq_len, dim, rank, num_layers, dt, dt_scales.data_ptr<float>(), forget_rates.data_ptr<float>(), num_heads, plasticity, sing_thresh, sing_strength, mx, mv, wf, wi, bf, wp, bp, topology, R, r, at::cuda::getCurrentCUDAStream());
    return {x_state, v_state, x_out_seq, v_out_seq, reg_loss};
}

std::vector<torch::Tensor> recurrent_manifold_backward_cuda(
    torch::Tensor grad_x_seq, torch::Tensor grad_x_final, torch::Tensor grad_v_final,
    torch::Tensor x_init, torch::Tensor v_init,
    torch::Tensor x_seq, torch::Tensor v_seq,
    torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v,
    torch::Tensor W_forget_stack, torch::Tensor W_input_stack, torch::Tensor b_forget_stack,
    torch::Tensor W_potential_stack, torch::Tensor b_potential_stack, // NEW
    int topology, float R, float r
) {
    int batch_total = x_seq.size(0); int dim = x_seq.size(2); int seq_len = x_seq.size(1);
    int rank = U_stack.size(2); int num_layers = U_stack.size(0) / num_heads;
    
    auto g_x0 = torch::zeros_like(x_init); auto g_v0 = torch::zeros_like(v_init);
    auto g_f = torch::zeros_like(forces); auto g_U = torch::zeros_like(U_stack); auto g_W = torch::zeros_like(W_stack);
    auto g_mx = mix_x.defined() && mix_x.numel() > 0 ? torch::zeros_like(mix_x) : torch::empty({0}, x_seq.options());
    auto g_mv = mix_v.defined() && mix_v.numel() > 0 ? torch::zeros_like(mix_v) : torch::empty({0}, x_seq.options());
    auto g_fr = torch::zeros_like(forget_rates);
    auto g_wf = W_forget_stack.defined() && W_forget_stack.numel() > 0 ? torch::zeros_like(W_forget_stack) : torch::empty({0}, x_seq.options());
    auto g_wi = W_input_stack.defined() && W_input_stack.numel() > 0 ? torch::zeros_like(W_input_stack) : torch::empty({0}, x_seq.options());
    auto g_bf = b_forget_stack.defined() && b_forget_stack.numel() > 0 ? torch::zeros_like(b_forget_stack) : torch::empty({0}, x_seq.options());
    
    // NEW GRADIENTS
    auto g_wp = W_potential_stack.defined() && W_potential_stack.numel() > 0 ? torch::zeros_like(W_potential_stack) : torch::empty({0}, x_seq.options());
    auto g_bp = b_potential_stack.defined() && b_potential_stack.numel() > 0 ? torch::zeros_like(b_potential_stack) : torch::empty({0}, x_seq.options());

    const float* gx_s_ptr = (grad_x_seq.numel() > 0) ? grad_x_seq.data_ptr<float>() : nullptr;
    const float* gxf_ptr = (grad_x_final.numel() > 0) ? grad_x_final.data_ptr<float>() : nullptr;
    const float* gvf_ptr = (grad_v_final.numel() > 0) ? grad_v_final.data_ptr<float>() : nullptr;
    
    const float* mx = (mix_x.numel() > 0) ? mix_x.data_ptr<float>() : nullptr;
    const float* mv = (mix_v.numel() > 0) ? mix_v.data_ptr<float>() : nullptr;
    const float* wf = (W_forget_stack.numel() > 0) ? W_forget_stack.data_ptr<float>() : nullptr;
    const float* wi = (W_input_stack.numel() > 0) ? W_input_stack.data_ptr<float>() : nullptr;
    const float* bf = (b_forget_stack.numel() > 0) ? b_forget_stack.data_ptr<float>() : nullptr;
    
    const float* wp = (W_potential_stack.numel() > 0) ? W_potential_stack.data_ptr<float>() : nullptr;
    const float* bp = (b_potential_stack.numel() > 0) ? b_potential_stack.data_ptr<float>() : nullptr;

    float* g_mx_ptr = (g_mx.numel() > 0) ? g_mx.data_ptr<float>() : nullptr;
    float* g_mv_ptr = (g_mv.numel() > 0) ? g_mv.data_ptr<float>() : nullptr;
    float* g_wf_ptr = (g_wf.numel() > 0) ? g_wf.data_ptr<float>() : nullptr;
    float* g_wi_ptr = (g_wi.numel() > 0) ? g_wi.data_ptr<float>() : nullptr;
    float* g_bf_ptr = (g_bf.numel() > 0) ? g_bf.data_ptr<float>() : nullptr;
    
    float* g_wp_ptr = (g_wp.numel() > 0) ? g_wp.data_ptr<float>() : nullptr;
    float* g_bp_ptr = (g_bp.numel() > 0) ? g_bp.data_ptr<float>() : nullptr;

    launch_recurrent_manifold_backward(gx_s_ptr, gxf_ptr, gvf_ptr, x_init.data_ptr<float>(), v_init.data_ptr<float>(), x_seq.data_ptr<float>(), v_seq.data_ptr<float>(), forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(), g_x0.data_ptr<float>(), g_v0.data_ptr<float>(), g_f.data_ptr<float>(), g_U.data_ptr<float>(), g_W.data_ptr<float>(), mx, mv, g_mx_ptr, g_mv_ptr, wf, wi, bf, g_wf_ptr, g_wi_ptr, g_bf_ptr, wp, bp, g_wp_ptr, g_bp_ptr, batch_total, seq_len, dim, rank, num_layers, num_heads, dt, dt_scales.data_ptr<float>(), forget_rates.data_ptr<float>(), g_fr.data_ptr<float>(), plasticity, sing_thresh, sing_strength, topology, R, r, at::cuda::getCurrentCUDAStream());
    return {g_x0, g_v0, g_f, g_U, g_W, g_mx, g_mv, g_fr, g_wf, g_wi, g_bf, g_wp, g_bp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda, py::arg("v"), py::arg("U"), py::arg("W"), py::arg("x"), py::arg("V_w"), py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"), py::arg("topology")=0, py::arg("R")=2.0f, py::arg("r")=1.0f);
    m.def("christoffel_backward", &christoffel_backward_cuda, py::arg("grad_gamma"), py::arg("v"), py::arg("U"), py::arg("W"), py::arg("x"), py::arg("V_w"), py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"), py::arg("topology")=0, py::arg("R")=2.0f, py::arg("r")=1.0f);
    m.def("reactive_christoffel_forward", &reactive_christoffel_forward_cuda, py::arg("v"), py::arg("U"), py::arg("W"), py::arg("x"), py::arg("V_w"), py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"), py::arg("topology")=0, py::arg("R")=2.0f, py::arg("r")=1.0f);

    m.def("recurrent_manifold_fused", &recurrent_manifold_fused_cuda,
          py::arg("x_state"), py::arg("v_state"), py::arg("forces"), py::arg("U_stack"), py::arg("W_stack"),
          py::arg("dt"), py::arg("dt_scales"), py::arg("forget_rates"), py::arg("num_heads"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("mix_x"), py::arg("mix_v"),
          py::arg("W_forget_stack"), py::arg("W_input_stack"), py::arg("b_forget_stack"),
          py::arg("W_potential_stack"), py::arg("b_potential_stack"),
          py::arg("topology"), py::arg("R"), py::arg("r"));

    m.def("recurrent_manifold_backward", &recurrent_manifold_backward_cuda,
          py::arg("grad_x_seq"), py::arg("grad_x_final"), py::arg("grad_v_final"),
          py::arg("x_init"), py::arg("v_init"),
          py::arg("x_seq"), py::arg("v_seq"),
          py::arg("forces"), py::arg("U_stack"), py::arg("W_stack"),
          py::arg("dt"), py::arg("dt_scales"), py::arg("forget_rates"), py::arg("num_heads"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("mix_x"), py::arg("mix_v"),
          py::arg("W_forget_stack"), py::arg("W_input_stack"), py::arg("b_forget_stack"),
          py::arg("W_potential_stack"), py::arg("b_potential_stack"),
          py::arg("topology"), py::arg("R"), py::arg("r"));
    m.def("leapfrog_fused", &leapfrog_fused_cuda, py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("Wf"), py::arg("bf"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("plasticity"), py::arg("R")=2.0f, py::arg("r")=1.0f);
    m.def("yoshida_fused", [](torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor Wf, torch::Tensor bf, float dt, torch::Tensor dt_scale, int steps, int topology, float plasticity, float R, float r) {
        return generic_integrator_cuda(x, v, f, U, W, Wf, bf, dt, dt_scale, steps, topology, plasticity, R, r, 1);
    }, py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("Wf"), py::arg("bf"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("plasticity"), py::arg("R")=2.0f, py::arg("r")=1.0f);

    m.def("rk4_fused", [](torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor Wf, torch::Tensor bf, float dt, torch::Tensor dt_scale, int steps, int topology, float plasticity, float R, float r) {
        return generic_integrator_cuda(x, v, f, U, W, Wf, bf, dt, dt_scale, steps, topology, plasticity, R, r, 2);
    }, py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("Wf"), py::arg("bf"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("plasticity"), py::arg("R")=2.0f, py::arg("r")=1.0f);

    m.def("dormand_prince_fused", [](torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor Wf, torch::Tensor bf, float dt, torch::Tensor dt_scale, int steps, int topology, float plasticity, float R, float r) {
        return generic_integrator_cuda(x, v, f, U, W, Wf, bf, dt, dt_scale, steps, topology, plasticity, R, r, 3);
    }, py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("Wf"), py::arg("bf"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("plasticity"), py::arg("R")=2.0f, py::arg("r")=1.0f);
    m.def("leapfrog_backward", &leapfrog_backward_cuda, py::arg("gx_n"), py::arg("gv_n"), py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("R")=2.0f, py::arg("r")=1.0f);
    m.def("euler_fused", &euler_fused_cuda, py::arg("x"), py::arg("v"), py::arg("f"), py::arg("U"), py::arg("W"), py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"), py::arg("R")=2.0f, py::arg("r")=1.0f);
}

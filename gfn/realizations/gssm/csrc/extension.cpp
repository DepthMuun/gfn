#include <torch/extension.h>
#include <vector>
#include "integrators/integrators.h"

// Declaración de funciones (Toroidal Loss)
torch::Tensor toroidal_distance_loss_fwd(const torch::Tensor& y_pred, const torch::Tensor& y_true);
torch::Tensor toroidal_distance_loss_bwd(const torch::Tensor& grad_output, const torch::Tensor& y_pred, const torch::Tensor& y_true);

// Declaración de funciones (Low Rank Christoffel)
torch::Tensor low_rank_christoffel_fwd(
    const torch::Tensor& v, const torch::Tensor& U, const torch::Tensor& W,
    double clamp_val, bool enable_trace_norm, bool is_paper_version);

// Implementación de Backward puro en ATen C++ (Compilado por MSVC, evadiendo bug de NVCC CICC)
std::vector<torch::Tensor> low_rank_christoffel_bwd(
    const torch::Tensor& grad_gamma, 
    const torch::Tensor& v, 
    const torch::Tensor& U, 
    const torch::Tensor& W,
    const torch::Tensor& gamma_out, 
    double clamp_val, 
    bool enable_trace_norm, 
    bool is_paper_version) 
{
    // Fast pure ATen operations avoiding Python overhead
    auto g_norm = gamma_out / clamp_val;
    auto d_tanh = 1.0 - g_norm.pow(2);
    auto grad_raw = grad_gamma * d_tanh;  // [B, H, D]

    if (enable_trace_norm) {
        auto mean_d = grad_raw.mean(-1, /*keepdim=*/true);
        grad_raw = grad_raw - mean_d;
    }

    // Explicit Batched Matrix Multiplication (bmm) to avoid matmul broadcast crashes
    // W is [H, D, R], grad_raw is [B, H, D]
    auto grad_raw_h = grad_raw.permute({1, 0, 2}); // [H, B, D]
    auto d_sq_h = torch::bmm(grad_raw_h, W);       // [H, B, D] @ [H, D, R] -> [H, B, R]
    auto d_sq = d_sq_h.permute({1, 0, 2});         // [B, H, R]
    
    auto v_h = v.permute({1, 0, 2});               // [H, B, D]
    auto v_r_h = torch::bmm(v_h, U);               // [H, B, D] @ [H, D, R] -> [H, B, R]
    auto v_r = v_r_h.permute({1, 0, 2});           // [B, H, R]
    
    torch::Tensor d_vr;
    if (is_paper_version) {
        auto vr_norm = torch::norm(v_r, 2, -1, true);
        auto denom = 1.0 + vr_norm;
        
        // Correct Chain Rule for normalized denominator Coupling:
        // grad_vr_j = grad_phi_j * (2v_j / denom) - v_j * Sum_k(grad_phi_k * v_k^2) / (||v|| * denom^2)
        auto S = (d_sq * v_r.pow(2)).sum(-1, true);
        auto term1 = d_sq * (2.0 * v_r / denom);
        auto term2 = (v_r * S) / (vr_norm * denom.pow(2) + 1e-8);
        d_vr = term1 - term2;
    } else {
        d_vr = d_sq * 2.0 * v_r;
    }

    auto d_vr_h = d_vr.permute({1, 0, 2});         // [H, B, R]
    auto U_t = U.transpose(-1, -2);                // [H, R, D]
    auto d_v_h = torch::bmm(d_vr_h, U_t);          // [H, B, R] @ [H, R, D] -> [H, B, D]
    auto d_v = d_v_h.permute({1, 0, 2});           // [B, H, D]

    // We accumulate W and U gradients over Batch:
    auto sq = is_paper_version ? v_r.pow(2) / (1.0 + torch::norm(v_r, 2, -1, true)) : v_r.pow(2); // [B, H, R]
    auto sq_h = sq.permute({1, 0, 2});             // [H, B, R]
    
    auto grad_raw_h_t = grad_raw_h.transpose(-1, -2); // [H, D, B]
    auto d_W = torch::bmm(grad_raw_h_t, sq_h);        // [H, D, B] @ [H, B, R] -> [H, D, R]

    auto v_h_t = v_h.transpose(-1, -2);               // [H, D, B]
    auto d_U = torch::bmm(v_h_t, d_vr_h);             // [H, D, B] @ [H, B, R] -> [H, D, R]

    return {d_v, d_U, d_W};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("toroidal_distance_loss_fwd", &toroidal_distance_loss_fwd, "Toroidal Distance Loss Forward (CUDA)");
    m.def("toroidal_distance_loss_bwd", &toroidal_distance_loss_bwd, "Toroidal Distance Loss Backward (CUDA)");
    
    m.def("low_rank_christoffel_fwd", &low_rank_christoffel_fwd, "Low Rank Christoffel Forward Kernel");
    m.def("low_rank_christoffel_bwd", &low_rank_christoffel_bwd, "Low Rank Christoffel Backward ATen");
    
    m.def("yoshida_fwd", &yoshida_fwd_aten, "Yoshida C++ Macro Integrator Step");
    m.def("leapfrog_fwd", &leapfrog_fwd_aten, "Leapfrog C++ Macro Integrator Step");
}

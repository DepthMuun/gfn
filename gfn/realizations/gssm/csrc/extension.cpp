#include <torch/extension.h>
#include <vector>
#include "integrators/integrators.h"

// ─── Forward declarations (CUDA .cu objects linked at build time) ─────────────
torch::Tensor toroidal_distance_loss_fwd(const torch::Tensor& y_pred, const torch::Tensor& y_true);
torch::Tensor toroidal_distance_loss_bwd(const torch::Tensor& grad_output, const torch::Tensor& y_pred, const torch::Tensor& y_true);
torch::Tensor toroidal_wrap_fwd(const torch::Tensor& x);

torch::Tensor low_rank_christoffel_fwd(
    const torch::Tensor& v, const torch::Tensor& U, const torch::Tensor& W,
    double clamp_val, bool enable_trace_norm, bool is_paper_version);

// ─── Low-rank Christoffel Backward (fused ATen — avoids Python overhead) ─────
// Uses AT_DISPATCH so it works for float32 and float16 inputs.
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
    // d(tanh) = 1 - (gamma_out / clamp)^2  — fused in-place to avoid temp allocation
    auto g_norm  = gamma_out / clamp_val;           // [B, H, D]
    auto d_tanh  = 1.0 - g_norm.pow(2);            // elementwise, stays on device
    auto grad_raw = (grad_gamma * d_tanh).contiguous();  // [B, H, D]

    if (enable_trace_norm) {
        grad_raw = grad_raw - grad_raw.mean(-1, /*keepdim=*/true);
    }

    // Batch matmuls via permute+bmm (avoids matmul broadcast crash on some builds)
    // W: [H, D, R],  grad_raw: [B, H, D]
    auto grad_raw_h = grad_raw.permute({1, 0, 2});       // [H, B, D]
    auto d_sq_h     = torch::bmm(grad_raw_h, W);         // [H, B, R]
    auto d_sq       = d_sq_h.permute({1, 0, 2});         // [B, H, R]

    auto v_h   = v.permute({1, 0, 2});                   // [H, B, D]
    auto v_r_h = torch::bmm(v_h, U);                     // [H, B, R]
    auto v_r   = v_r_h.permute({1, 0, 2});               // [B, H, R]

    torch::Tensor d_vr;
    if (is_paper_version) {
        auto vr_norm = torch::norm(v_r, 2, -1, true);
        auto denom   = 1.0 + vr_norm;
        // Correct chain-rule: grad_vr_j = d_sq_j*(2v_j/denom) - v_j*Sum(d_sq*v^2)/(||v||*denom^2)
        auto S      = (d_sq * v_r.pow(2)).sum(-1, true);
        auto term1  = d_sq * (2.0 * v_r / denom);
        auto term2  = (v_r * S) / (vr_norm * denom.pow(2) + 1e-8);
        d_vr        = term1 - term2;
    } else {
        d_vr = d_sq * (2.0 * v_r);
    }

    auto d_vr_h = d_vr.permute({1, 0, 2});              // [H, B, R]
    auto U_t    = U.transpose(-1, -2);                   // [H, R, D]
    auto d_v    = torch::bmm(d_vr_h, U_t).permute({1, 0, 2});  // [B, H, D]

    // W and U parameter gradients (accumulated over batch)
    auto sq = is_paper_version
        ? v_r.pow(2) / (1.0 + torch::norm(v_r, 2, -1, true))
        : v_r.pow(2);                                    // [B, H, R]

    auto sq_h           = sq.permute({1, 0, 2});                    // [H, B, R]
    auto grad_raw_h_t   = grad_raw_h.transpose(-1, -2);             // [H, D, B]
    auto d_W = torch::bmm(grad_raw_h_t, sq_h);                      // [H, D, R]

    auto v_h_t = v_h.transpose(-1, -2);                             // [H, D, B]
    auto d_U   = torch::bmm(v_h_t, d_vr_h);                        // [H, D, R]

    return {d_v, d_U, d_W};
}

// ─── Pybind11 module bindings ─────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Loss kernels
    m.def("toroidal_distance_loss_fwd", &toroidal_distance_loss_fwd,
          "Toroidal Distance Loss Forward — float4 vectorised + float16 (CUDA)");
    m.def("toroidal_distance_loss_bwd", &toroidal_distance_loss_bwd,
          "Toroidal Distance Loss Backward — float4 vectorised + float16 (CUDA)");
    m.def("toroidal_wrap_fwd", &toroidal_wrap_fwd,
          "Toroidal Wrap x→[-π,π) — float4 fused kernel (CUDA)");

    // Geometry kernels
    m.def("low_rank_christoffel_fwd", &low_rank_christoffel_fwd,
          "Low-Rank Christoffel Forward — warp-shuffle + float4 + float16 (CUDA)");
    m.def("low_rank_christoffel_bwd", &low_rank_christoffel_bwd,
          "Low-Rank Christoffel Backward — fused ATen (C++)");

    // Integrators
    m.def("yoshida_fwd", &yoshida_fwd_aten,
          "Yoshida 4th-order Integrator — hoisted v_r + float16 guard (C++)");
    m.def("leapfrog_fwd", &leapfrog_fwd_aten,
          "Leapfrog Integrator — hoisted v_r + float16 guard (C++)");
}

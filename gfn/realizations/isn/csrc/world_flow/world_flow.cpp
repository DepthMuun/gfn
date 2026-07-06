#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <cmath>

// Fast C++ implementation for the GFN World Flow
// v2.7.3: adds symplectic (leapfrog / yoshida) integrator path on top of the
// original forward-Euler path.

// ────────────────────────────────────────────────────────────────────────────
// Euler path (preserved verbatim for backward compatibility)
// ────────────────────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> world_flow_forward(
    torch::Tensor initial_state, // [B, D]
    torch::Tensor f_ext_all,     // [B, L, D] pre-computed diffusion(impulses)
    torch::Tensor drift_weight,  // [D, D]
    torch::Tensor drift_bias,    // [D]
    double noise_std
) {
    auto batch_size = f_ext_all.size(0);
    auto seq_len = f_ext_all.size(1);

    std::vector<torch::Tensor> emitted_embeddings;
    std::vector<torch::Tensor> energy_trace;
    emitted_embeddings.reserve(seq_len);
    energy_trace.reserve(seq_len);

    auto state = initial_state.clone();

    for (int t = 0; t < seq_len; ++t) {
        auto drift_linear = torch::linear(state, drift_weight, drift_bias);
        auto v_drift = torch::tanh(drift_linear);

        auto f_ext = f_ext_all.select(1, t); // [B, D]
        state = state + v_drift + f_ext;

        if (noise_std > 0) {
            auto noise = torch::randn_like(state) * noise_std;
            state = state + noise;
        }

        emitted_embeddings.push_back(state.unsqueeze(1));
        auto energy = torch::norm(state, /*p=*/2, /*dim=*/-1, /*keepdim=*/true);
        energy_trace.push_back(energy.unsqueeze(1));
    }

    auto final_embs = torch::cat(emitted_embeddings, 1);
    auto energies = torch::cat(energy_trace, 1);

    return std::make_tuple(final_embs, energies, state);
}

// ────────────────────────────────────────────────────────────────────────────
// Symplectic path (v2.7.3)
//
// Implements Störmer-Verlet (leapfrog) and Yoshida 4th-order composition on a
// (position, velocity) state pair. Optional friction damps velocity linearly.
//
// Args:
//   initial_x : [B, D]
//   initial_v : [B, D]
//   f_ext_all : [B, L, D]
//   drift_weight, drift_bias : for drift(x) = tanh(x W^T + b)
//   dt        : scalar timestep
//   friction  : scalar linear damping coefficient on velocity (>= 0)
//   noise_std : scalar Gaussian noise on position after each step
//   use_yoshida: bool; if true apply Yoshida 4th-order composition
//
// Returns:
//   final_embs : [B, L, D] — emitted positions
//   energies   : [B, L, 1] — ||x_t|| per step
//   final_x    : [B, D]    — last position
//   final_v    : [B, D]    — last velocity
// ────────────────────────────────────────────────────────────────────────────

static inline torch::Tensor compute_acceleration(
    const torch::Tensor& x,
    const torch::Tensor& f_ext,
    const torch::Tensor& drift_weight,
    const torch::Tensor& drift_bias
) {
    auto drift_linear = torch::linear(x, drift_weight, drift_bias);
    return torch::tanh(drift_linear) + f_ext;
}

static inline void apply_friction(
    torch::Tensor& v,
    double fric,
    double dt_scale
) {
    if (fric > 0) {
        double factor = std::max(0.0, 1.0 - fric * dt_scale);
        v = v * factor;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
world_forward_leapfrog(
    torch::Tensor initial_x,
    torch::Tensor initial_v,
    torch::Tensor f_ext_all,
    torch::Tensor drift_weight,
    torch::Tensor drift_bias,
    double dt,
    double friction,
    double noise_std,
    bool use_yoshida
) {
    auto batch_size = f_ext_all.size(0);
    auto seq_len = f_ext_all.size(1);

    std::vector<torch::Tensor> emitted_embeddings;
    std::vector<torch::Tensor> energy_trace;
    emitted_embeddings.reserve(seq_len);
    energy_trace.reserve(seq_len);

    auto x = initial_x.clone();
    auto v = initial_v.clone();

    // Yoshida 4th-order weights: w1 + w0 + w1 = 1, sub-steps w1, w0, w1.
    // The composition is three half-kicks/drifts with those weights.
    const double cbrt2 = std::cbrt(2.0);
    const double w1 = 1.0 / (2.0 - cbrt2);
    const double w0 = -cbrt2 * w1;

    for (int t = 0; t < seq_len; ++t) {
        auto f_ext = f_ext_all.select(1, t); // [B, D]

        if (use_yoshida) {
            const double ws[3] = {w1, w0, w1};
            for (int s = 0; s < 3; ++s) {
                auto a = compute_acceleration(x, f_ext, drift_weight, drift_bias);
                v = v + ws[s] * dt * a;
                x = x + ws[s] * dt * v;
                apply_friction(v, friction, ws[s] * dt);
            }
            auto a_final = compute_acceleration(x, f_ext, drift_weight, drift_bias);
            v = v + 0.5 * dt * a_final;
        } else {
            // Störmer-Verlet leapfrog: single symmetric step
            auto a = compute_acceleration(x, f_ext, drift_weight, drift_bias);
            auto v_half = v + 0.5 * dt * a;
            x = x + dt * v_half;
            auto a_new = compute_acceleration(x, f_ext, drift_weight, drift_bias);
            v = v_half + 0.5 * dt * a_new;
            apply_friction(v, friction, dt);
        }

        if (noise_std > 0) {
            x = x + torch::randn_like(x) * noise_std;
        }

        emitted_embeddings.push_back(x.unsqueeze(1));
        // [B, 1] energy; cat → [B, L, 1] consistent with Python path.
        auto energy = torch::norm(x, /*p=*/2, /*dim=*/-1, /*keepdim=*/true);
        energy_trace.push_back(energy);
    }

    auto final_embs = torch::cat(emitted_embeddings, 1);
    auto energies = torch::cat(energy_trace, 1);

    return std::make_tuple(final_embs, energies, x, v);
}

// ────────────────────────────────────────────────────────────────────────────
// Scanner flow (unchanged)
// ────────────────────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> scanner_flow_forward(
    torch::Tensor initial_state, // [B, D]
    torch::Tensor impulses,      // [B, L, D]
    torch::Tensor gate_weight,   // [D, D]
    torch::Tensor gate_bias      // [D]
) {
    auto seq_len = impulses.size(1);

    std::vector<torch::Tensor> outputs;
    outputs.reserve(seq_len);

    auto state = initial_state.clone();

    for (int t = 0; t < seq_len; ++t) {
        auto f_ext = impulses.select(1, t); // [B, D]
        auto state_ext = state + f_ext;
        auto linear_out = torch::linear(state_ext, gate_weight, gate_bias);
        state = torch::tanh(linear_out);
        outputs.push_back(state.unsqueeze(1));
    }

    auto final_outputs = torch::cat(outputs, 1);
    return std::make_tuple(final_outputs, state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("world_forward", &world_flow_forward, "GFN World Flow Forward (Euler, ATen/C++)");
    m.def(
        "world_forward_leapfrog",
        &world_forward_leapfrog,
        "GFN World Flow Forward (Leapfrog / Yoshida symplectic, v2.7.3)"
    );
    m.def("scanner_forward", &scanner_flow_forward, "GFN Scanner Flow Forward (ATen/C++)");
}
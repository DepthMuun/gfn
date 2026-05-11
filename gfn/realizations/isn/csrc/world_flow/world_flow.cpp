#include <torch/extension.h>
#include <vector>
#include <tuple>

// Fast C++ implementation for the GFN World Flow
// This eliminates Python GIL overhead during the sequential O(L) loop.

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

// Fast C++ implementation for the GFN Scanner Flow
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
    m.def("world_forward", &world_flow_forward, "GFN World Flow Forward (ATen/C++)");
    m.def("scanner_forward", &scanner_flow_forward, "GFN Scanner Flow Forward (ATen/C++)");
}
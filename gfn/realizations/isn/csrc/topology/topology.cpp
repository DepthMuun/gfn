#include <torch/extension.h>
#include <vector>
#include <tuple>

using namespace torch::indexing;

// Fast C++ O(1) Memory Causal Topological Engine - Purified Bilinear Version 1.0

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> topological_forward(
    torch::Tensor embeddings_seq, // [B, L, D]
    torch::Tensor mask_op_seq,    // [B, L]
    torch::Tensor mask_entity_seq,// [B, L]
    
    // Bilinear Geodesic Operator Weights
    torch::Tensor op_w1, torch::Tensor op_b1,
    torch::Tensor op_w2, torch::Tensor op_b2,
    torch::Tensor op_w3, torch::Tensor op_b3,
    
    // Purified Emitter Weights (Single Projections)
    torch::Tensor em_w_energy, torch::Tensor em_b_energy,
    torch::Tensor em_w_out, torch::Tensor em_b_out,
    
    double threshold_base,
    double noise_std,
    int max_burst
) {
    auto batch_size = embeddings_seq.size(0);
    auto seq_len = embeddings_seq.size(1);
    auto d_embedding = embeddings_seq.size(2);
    auto vocab_size = em_w_out.size(0);
    auto device = embeddings_seq.device();

    // The O(1) Memory Pointer Stack in native C++
    std::vector<std::vector<torch::Tensor>> active_stack(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        active_stack[b].reserve(100);
    }
    
    std::vector<std::vector<torch::Tensor>> emitted_logits(batch_size);
    std::vector<std::vector<torch::Tensor>> emitted_embeddings(batch_size);
    std::vector<torch::Tensor> all_energies;
    all_energies.reserve(seq_len);

    for (int t = 0; t < seq_len; ++t) {
        auto mask_op = mask_op_seq.select(1, t);       // [B]
        auto mask_entity = mask_entity_seq.select(1, t); // [B]
        auto embeddings = embeddings_seq.select(1, t);   // [B, D]
        
        auto step_energy = torch::zeros({batch_size, 1}, torch::TensorOptions().device(device));
        
        for (int b = 0; b < batch_size; ++b) {
            bool is_op = mask_op[b].item<bool>();
            bool is_entity = mask_entity[b].item<bool>();
            
            // 1. Process incoming entity
            if (is_entity && active_stack[b].size() < 100) {
                active_stack[b].push_back(embeddings[b]);
            }

            // 2. Process operations / Reaction Bursts (Purified Bilinear Physics)
            if (is_op) {
                int burst_count = 0;
                while (active_stack[b].size() >= 2 && burst_count < max_burst) {
                    auto e2 = active_stack[b].back(); active_stack[b].pop_back();
                    auto e1 = active_stack[b].back(); active_stack[b].pop_back();
                    
                    // GFN-Op v1.0: Pure Bilinear Geodesic Interaction
                    auto combined_linear = torch::linear(e1, op_w1, op_b1) + 
                                          torch::linear(e2, op_w2, op_b2) + 
                                          torch::linear(e1 * e2, op_w3, op_b3);
                    
                    auto op_result = torch::layer_norm(combined_linear, {d_embedding});

                    // Thermal Noise
                    if (noise_std > 0) {
                        auto noise = torch::randn_like(op_result) * noise_std;
                        op_result = op_result + noise;
                    }
                    
                    // Purified Emitter Projections
                    auto meta_energy = torch::sigmoid(torch::linear(op_result.unsqueeze(0), em_w_energy, em_b_energy));
                    auto logits = torch::linear(op_result.unsqueeze(0), em_w_out, em_b_out);
                    
                    bool emits = meta_energy.item<double>() > threshold_base;
                    
                    // Telemetry
                    step_energy.index_put_({b, 0}, meta_energy.squeeze(0));
                    
                    if (emits) {
                        emitted_logits[b].push_back(logits);
                        emitted_embeddings[b].push_back(op_result.unsqueeze(0).unsqueeze(0));
                    } else {
                        if (active_stack[b].size() < 100) {
                            active_stack[b].push_back(op_result);
                        }
                    }
                    burst_count++;
                }
            }
        }
        all_energies.push_back(step_energy);
    }
    
    // Aggregate outputs
    std::vector<torch::Tensor> padded_logits;
    std::vector<torch::Tensor> padded_embs;
    int max_emitted = 0;
    
    for (int b = 0; b < batch_size; ++b) {
        if (emitted_logits[b].size() > max_emitted) {
            max_emitted = emitted_logits[b].size();
        }
    }
    
    if (max_emitted > 0) {
        for (int b = 0; b < batch_size; ++b) {
            torch::Tensor l_tensor, e_tensor;
            if (emitted_logits[b].size() > 0) {
                l_tensor = torch::cat(emitted_logits[b], 0).unsqueeze(0); 
                e_tensor = torch::cat(emitted_embeddings[b], 1);          
            } else {
                l_tensor = torch::zeros({1, 0, vocab_size}, torch::TensorOptions().device(device));
                e_tensor = torch::zeros({1, 0, d_embedding}, torch::TensorOptions().device(device));
            }
            
            int pad_len = max_emitted - l_tensor.size(1);
            if (pad_len > 0) {
                auto pad_l = torch::zeros({1, pad_len, vocab_size}, torch::TensorOptions().device(device));
                auto pad_e = torch::zeros({1, pad_len, d_embedding}, torch::TensorOptions().device(device));
                l_tensor = torch::cat({l_tensor, pad_l}, 1);
                e_tensor = torch::cat({e_tensor, pad_e}, 1);
            }
            padded_logits.push_back(l_tensor);
            padded_embs.push_back(e_tensor);
        }
    } else {
        auto final_l = torch::zeros({batch_size, 0, vocab_size}, torch::TensorOptions().device(device));
        auto final_e = torch::zeros({batch_size, 0, d_embedding}, torch::TensorOptions().device(device));
        return std::make_tuple(final_l, final_e, torch::cat(all_energies, 1));
    }
    
    return std::make_tuple(torch::cat(padded_logits, 0), torch::cat(padded_embs, 0), torch::cat(all_energies, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &topological_forward, "O(1) Topological Purified Bilinear Forward");
}

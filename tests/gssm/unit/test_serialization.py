import torch
import os
import sys
import shutil

# Añadir el raíz del proyecto al sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import gfn
from gfn.realizations.gssm.models.factory import ModelFactory

def test_serialization():
    print("Testing GFN Model Serialization (HF Style)...")
    
    # 1. Create a model (Categorical to check parameters)
    vocab_size = 50
    dim = 64
    heads = 4
    model = gfn.create(
        vocab_size=vocab_size, 
        dim=dim, 
        heads=heads, 
        topology_type='torus',
        holographic=False,
        initial_spread=0.0
    )
    model.eval()
    
    # 2. Save the model
    save_path = "tmp_model_save"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    gfn.save(model, save_path)
    
    # Verify files exist
    assert os.path.exists(os.path.join(save_path, "config.json")), "config.json missing"
    assert os.path.exists(os.path.join(save_path, "pytorch_model.bin")), "pytorch_model.bin missing"
    
    # 3. Load the model
    # Capture return of load_state_dict to check for missing/unexpected keys
    # Note: ModelFactory.from_pretrained calls load_state_dict internally.
    # We will do it manually here to see the result.
    loaded_model = ModelFactory.create(config=gfn.load(save_path).config)
    state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"), map_location='cpu', weights_only=True)
    load_result = loaded_model.load_state_dict(state_dict)
    print(f"Load state dict result: {load_result}")
    loaded_model.eval()
    
    # Debug: Check weights
    def get_readout_weight(m):
        for name, param in m.named_parameters():
            if 'readout_plugin' in name:
                return param
        return None

    w_orig = get_readout_weight(model)
    w_load = get_readout_weight(loaded_model)
    
    if w_orig is not None and w_load is not None:
        print(f"Readout weight match (bit-exact): {torch.equal(w_orig, w_load)}")
        max_w_diff = (w_orig - w_load).abs().max().item()
        print(f"Max Readout weight diff: {max_w_diff}")
    else:
        print(f"Readout parameters not found. Keys: {[n for n, _ in model.named_parameters() if 'readout' in n]}")

    # 4. Compare outputs
    test_input = torch.randint(0, vocab_size, (1, 10))
    
    with torch.no_grad():
        orig_logits, (orig_xf, orig_vf), orig_info = model(test_input)
        load_logits, (load_xf, load_vf), load_info = loaded_model(test_input)
    
    # Check sequence drift
    orig_x_seq = orig_info['x_seq'] # [B, L, H, HD]
    load_x_seq = load_info['x_seq']
    
    for i in range(orig_x_seq.shape[1]):
        x_d = (orig_x_seq[:, i] - load_x_seq[:, i]).abs().max().item()
        l_d = (orig_logits[:, i] - load_logits[:, i]).abs().max().item()
        print(f"Step {i}: X-diff={x_d:.2e}, Logit-diff={l_d:.2e}")

    # Check if match
    xf_diff = (orig_xf - load_xf).abs().max().item()
    logits_diff = (orig_logits - load_logits).abs().max().item()
    
    xf_match = torch.allclose(orig_xf, load_xf, atol=1e-7)
    logits_match = torch.allclose(orig_logits, load_logits, atol=1e-7)
    
    print(f"X final max diff: {xf_diff:.2e}")
    print(f"Logits max diff: {logits_diff:.2e}")
    
    assert logits_match, "Logits do not match after load!"
    assert xf_match, "X states do not match after load!"
    
    # Clean up
    shutil.rmtree(save_path)
    print("Serialization test PASSED!")

if __name__ == "__main__":
    test_serialization()

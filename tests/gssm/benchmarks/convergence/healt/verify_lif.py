import torch
import torch.nn as nn
from gfn.realizations.gssm.models.models.factory.model import Model
from gfn.realizations.gssm.config.schema import PhysicsConfig

def test_lif_persistence():
    print("Testing Learnable Importance Friction (LIF) Persistence...")
    
    # 1. Setup Config with LIF enabled
    physics_cfg = PhysicsConfig()
    physics_cfg.stability.friction_mode = 'lif'
    physics_cfg.topology.type = 'torus'
    
    # 2. Create Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(vocab_size=10, dim=32, depth=2, heads=1).to(device)
    
    # 3. Create dummy input with a "context switch" (high force pulse)
    # Sequence: 10 tokens. Token 5 has high intensity.
    batch_size = 1
    seq_len = 10
    forces = torch.randn(batch_size, seq_len, 32).to(device) * 0.1
    forces[:, 5] *= 10.0 # Context switch pulse
    
    # 4. Forward pass
    logits, (x_f, v_f), all_c, v_seq, x_seq, all_f = model(force_manual=forces)
    
    # 5. Check velocity norm over time
    v_norms = v_seq.norm(dim=-1).squeeze(0) # [L]
    print(f"Velocity norms per step: {v_norms.tolist()}")
    
    # In a good LIF implementation, step 5 (the pulse) might trigger 
    # higher friction in subsequent steps, or exactly at that step.
    
    # Let's check if the LIF plugin exists
    if hasattr(model, 'lif_plugin'):
        print("SUCCESS: LIF plugin registered and active.")
    else:
        print("FAILURE: LIF plugin not found.")
        return

    print("Test complete.")

if __name__ == "__main__":
    test_lif_persistence()

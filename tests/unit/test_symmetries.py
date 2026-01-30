import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.losses import noether_loss

def test_isomeric_heads_weight_sharing():
    print("Testing Isomeric Heads Weight Sharing...")
    
    # 1. Config with symmetric heads
    physics_config = {
        'symmetries': {
            'enabled': True,
            'isomeric_groups': [[0, 1]] # Heads 0 and 1 are symmetric
        }
    }
    
    # Init model: dim=64, heads=4 -> head_dim=16
    model = Manifold(vocab_size=10, dim=64, heads=4, physics_config=physics_config)
    
    # Check layer 0
    layer = model.layers[0]
    
    # Christoffels for head 0 and 1 should be the EXACT same instance
    assert layer.christoffels[0] is layer.christoffels[1], "Head 0 and 1 should share the same Christoffel instance"
    print(">> Hard weight sharing: PASS")
    
    # Verify parameter IDs
    id_U0 = id(layer.christoffels[0].U)
    id_U1 = id(layer.christoffels[1].U)
    assert id_U0 == id_U1, "Parameter U should be identical for isomeric heads"
    print(">> Parameter identity: PASS")

def test_noether_loss_consistency():
    print("\nTesting Noether Loss Consistency...")
    
    # Mock Christoffel outputs (Γ(v)) for 4 heads
    # Head 0 and 1 are supposed to be symmetric
    c0 = torch.randn(4, 16)
    c1 = c0.clone() # Identical (Zero loss)
    c2 = torch.randn(4, 16)
    c3 = torch.randn(4, 16)
    
    christoffel_outputs = [c0, c1, c2, c3]
    groups = [[0, 1]]
    
    loss_val = noether_loss(christoffel_outputs, groups, lambda_n=1.0)
    assert loss_val.item() == 0, f"Identical outputs should yield zero Noether loss, got {loss_val.item()}"
    print(">> Identical outputs (Zero loss): PASS")
    
    # Divergent outputs
    c1_div = c0 + 1.0 
    christoffel_outputs_div = [c0, c1_div, c2, c3]
    loss_div = noether_loss(christoffel_outputs_div, groups, lambda_n=1.0)
    assert loss_div.item() > 0, "Divergent outputs should yield positive Noether loss"
    print(f">> Divergent outputs (Positive loss: {loss_div.item():.4f}): PASS")

if __name__ == "__main__":
    try:
        test_isomeric_heads_weight_sharing()
        test_noether_loss_consistency()
        print("\nAll Semantic Symmetry tests: PASS")
    except Exception as e:
        print(f"\nSymmetry tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

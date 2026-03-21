import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn import api

def test_full_orchestration_assembly():
    """Verificar que el factory api.create ensambla un modelo funcional."""
    print("Testing Model (RealityModel) Orchestration...")
    
    vocab_size = 100
    d_model = 128
    d_emb = 64
    d_prop = 16
    
    model = api.create(
        vocab_size=vocab_size,
        d_model=d_model,
        d_embedding=d_emb,
        d_properties=d_prop
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_ids = torch.randint(0, vocab_size, (2, 8)).to(device)
    
    print("Executing full forward pass...")
    outputs = model(input_ids)
    
    assert 'logits' in outputs
    assert 'energy_trace' in outputs
    
    # Shapes
    print(f"Logits shape: {outputs['logits'].shape}")
    assert outputs['logits'].shape[0] == 2
    assert outputs['logits'].shape[2] == vocab_size
    
    # En V4 Polish, el world_coherence debe estar presente
    assert 'world_coherence' in outputs
    assert outputs['world_coherence'].shape[0] == 2
    
    print("✓ Full orchestration (Scanner->World->Emitter) test passed")

if __name__ == "__main__":
    test_full_orchestration_assembly()
    print("\n[SUCCESS] Full orchestration unit tests passed.")

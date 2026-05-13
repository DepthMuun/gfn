import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.components.emitters.threshold_emitter import ThresholdEmitter

def test_emitter_output():
    """Verificar que el emitter proyecta correctamente a logits."""
    print("Testing ThresholdEmitter physics...")
    
    batch_size = 4
    num_emissions = 3
    d_emb = 256
    vocab_size = 5000
    
    emitter = ThresholdEmitter(d_embedding=d_emb, vocab_size=vocab_size)
    
    # Sin mocks: enviamos tensores de forma real
    emitted_embeddings = torch.randn(batch_size, num_emissions, d_emb)
    
    logits = emitter(emitted_embeddings)
    
    print(f"Input embeddings shape: {emitted_embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, num_emissions, vocab_size), "Logits shape mismatch"
    assert not torch.isnan(logits).any(), "Emitter produced NaNs"
    
    # Verificar que el gradiente fluye
    logits.sum().backward()
    assert emitter.emission.weight.grad is not None, "Gradient flow broken in Emitter"
    
    print("✓ Emitter output and gradient flow tests passed")

if __name__ == "__main__":
    test_emitter_output()
    print("\n[SUCCESS] ThresholdEmitter unit tests passed.")

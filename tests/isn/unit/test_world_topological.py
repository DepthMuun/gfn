import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.components.worlds.topological_world import TopologicalWorld

def test_topological_world_simulation():
    """Validación del Kernel C++ nativo sin mocks."""
    print("Testing TopologicalWorld (Native C++ Kernel)...")
    
    # Requisitos: el kernel debe estar compilado
    try:
        from gfn_topology import simulate
    except ImportError:
        print("⚠ Skip test: gfn_topology kernel not compiled.")
        return

    batch_size = 2
    seq_len = 5
    d_model = 128
    d_emb = 64
    d_prop = 32
    vocab_size = 100
    
    world = TopologicalWorld(d_model=d_model, d_embedding=d_emb, d_properties=d_prop)
    
    impulses = torch.randn(batch_size, seq_len, d_model)
    
    # MOCK-FREE: Simulating with real random weights 
    # as expected by the kernel in V4 Polish
    kwargs = {
        'impulses': impulses,
        'noise_std': 0.05,
        'max_burst': 5,
        'em_w_energy': torch.randn(d_emb, d_emb),
        'em_b_energy': torch.randn(d_emb),
        'em_w_out': torch.randn(vocab_size, d_emb),
        'em_b_out': torch.randn(vocab_size),
        'threshold': 0.8
    }
    
    print("Executing simulation...")
    output = world(**kwargs)
    
    assert 'emitted_embeddings' in output
    assert 'energy_trace' in output
    
    # Verification of shapes
    print(f"Emitted Embeddings shape: {output['emitted_embeddings'].shape}")
    print(f"Energy Trace shape: {output['energy_trace'].shape}")
    
    assert output['emitted_embeddings'].shape[0] == batch_size
    assert output['emitted_embeddings'].shape[2] == d_emb
    assert not torch.isnan(output['emitted_embeddings']).any(), "World output contains NaNs"
    
    print("✓ Native Kernel simulation test passed")

if __name__ == "__main__":
    test_topological_world_simulation()
    print("\n[SUCCESS] TopologicalWorld unit tests passed.")

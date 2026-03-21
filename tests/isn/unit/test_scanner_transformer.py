import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.components.scanners.transformer_scanner import TransformerScanner

def test_transformer_scanner_output_shape():
    """Verificar que el scanner devuelve las dimensiones correctas."""
    print("Testing TransformerScanner output shapes...")
    
    vocab_size = 100
    d_model = 256
    seq_len = 10
    batch_size = 2
    
    scanner = TransformerScanner(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    impulses = scanner(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {impulses.shape}")
    
    assert impulses.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {impulses.shape}"
    assert not torch.isnan(impulses).any(), "Found NaNs in scanner output"
    print("✓ Output shape and stability test passed")

def test_transformer_scanner_causality():
    """Verificar que el scanner es causal (el futuro no afecta al pasado)."""
    print("\nTesting TransformerScanner causality...")
    
    vocab_size = 100
    d_model = 128
    seq_len = 5
    batch_size = 1
    
    scanner = TransformerScanner(vocab_size=vocab_size, d_model=d_model)
    scanner.eval() # Disable dropout for deterministic output
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run once
    out1 = scanner(input_ids)
    
    # Change last token
    input_ids2 = input_ids.clone()
    input_ids2[0, -1] = (input_ids[0, -1] + 1) % vocab_size
    
    # Run again
    out2 = scanner(input_ids2)
    
    # Everything before the last token should be IDENTICAL
    diff = torch.abs(out1[:, :-1, :] - out2[:, :-1, :]).max().item()
    print(f"Max difference in past sequence after future change: {diff}")
    
    assert diff < 1e-6, "Causality broken! Scanner is attending to the future."
    print("✓ Causality test passed")

if __name__ == "__main__":
    test_transformer_scanner_output_shape()
    test_transformer_scanner_causality()
    print("\n[SUCCESS] all TransformerScanner unit tests passed.")

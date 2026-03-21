"""
ISN V4 Modularity Verification
Test swapping components (Scanner) without breaking the simulation loop.
"""

import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn import api
from gfn.realizations.isn.components.scanners.linear_scanner import LinearScanner
from gfn.realizations.isn.components.scanners.transformer_scanner import TransformerScanner

def test_modular_swap():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 100
    batch_size = 2
    seq_len = 16
    
    print("--- Phase 1: Default Modular Model (LinearScanner) ---")
    model_linear = api.create(vocab_size=vocab_size, scanner_cls=LinearScanner)
    model_linear.to(device)
    
    input_ids = torch.randint(3, vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        out_linear = model_linear(input_ids)
        print(f"Linear Path Logits Shape: {out_linear['logits'].shape}")

    print("\n--- Phase 2: Swapping to TransformerScanner ---")
    # We can swap at creation time via the factory
    model_trans = api.create(vocab_size=vocab_size, scanner_cls=TransformerScanner)
    model_trans.to(device)
    
    with torch.no_grad():
        out_trans = model_trans(input_ids)
        print(f"Transformer Path Logits Shape: {out_trans['logits'].shape}")
        
    print("\n--- Phase 3: Dynamic Component Swapping ---")
    # We can also swap dynamically on an existing model instance
    new_scanner = TransformerScanner(vocab_size=vocab_size, d_model=model_linear.scanner.d_model).to(device)
    model_linear.scanner = new_scanner
    
    with torch.no_grad():
        out_dynamic = model_linear(input_ids)
        print(f"Dynamic Swap Logits Shape: {out_dynamic['logits'].shape}")

    print("\n[SUCCESS] ISN V4 Architecture is fully modular.")

if __name__ == "__main__":
    test_modular_swap()

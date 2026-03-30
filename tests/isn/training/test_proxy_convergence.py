import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
from gfn.realizations.isn.training.direct_projection_trainer import DirectProjectionTrainer

# Mock data: Copy Task (Predict the first token)
def get_mock_data(seq_len=64, batch_size=32):
    # Model should learn that targets[:, -1] = x[:, 0]
    x = torch.randint(0, 86, (batch_size, seq_len))
    y = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y[:, -1] = x[:, 0]
    return {'input_ids': x, 'output_ids': y}

def test_convergence():
    print("Testing Proxy Convergence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainer = DirectProjectionTrainer(
        vocab_size=86,
        d_embedding=128,
        d_model=128,
        n_layers=3,
        lr=1e-3,
        device=device
    )
    
    # Simple training loop for 2000 steps
    for i in range(2000):
        batch = get_mock_data(batch_size=128)
        metrics = trainer.step(batch['input_ids'].to(device), batch['output_ids'].to(device))
        if i % 200 == 0:
            # Check gradients
            grad_sum = sum(p.grad.abs().sum() for p in trainer.model.parameters() if p.grad is not None)
            print(f"Step {i}: Loss {metrics['loss']:.4f} | Grad Sum {grad_sum:.6f}")
            
    print("Final Loss:", metrics['loss'])
    # With L=64, it's harder but should definitely beat 3.6 (BoW limit)
    assert metrics['loss'] < 3.0, "Should break the 3.6 loss stall"
    print("Convergence test passed (L=64 Copy Task).")

if __name__ == "__main__":
    test_convergence()

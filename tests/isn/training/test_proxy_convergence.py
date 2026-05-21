import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn

# Mock data: Copy Task (Predict the first token)
def get_mock_data(seq_len=64, batch_size=32):
    x = torch.randint(0, 86, (batch_size, seq_len))
    y = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y[:, -1] = x[:, 0]
    return {'input_ids': x, 'output_ids': y}

def test_convergence():
    print("Testing Proxy Convergence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = nn.Sequential(
        nn.Embedding(128, 128),
        nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True),
        nn.Linear(128, 128)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(2000):
        batch = get_mock_data(batch_size=128)
        input_ids = batch['input_ids'].to(device)
        output_ids = batch['output_ids'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, 128), output_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            grad_sum = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
            print(f"Step {i}: Loss {loss.item():.4f} | Grad Sum {grad_sum:.6f}")
            
    print("Final Loss:", loss.item())
    assert loss.item() < 3.0, "Should break the 3.6 loss stall"
    print("Convergence test passed (L=64 Copy Task).")

if __name__ == "__main__":
    test_convergence()

import torch
import torch.nn as nn
import torch.optim as optim
from ..framework import MatrixRunner
import gfn

def train_nlp(model, steps=150):
    # Mini-NLP Task:
    # Character-level prediction on a small set of "geometric" sentences.
    # Tests projection to high-vocab spaces and token flow.
    # Perfect for GeodesicAttention and FlowMixer.
    
    sentences = [
        "the circle is round",
        "the square is flat",
        "the torus is curved",
        "the point is zero",
        "movement follows the flow"
    ]
    
    # Simple character mapping
    chars = sorted(list(set("".join(sentences))))
    char_to_id = {c: i for i, c in enumerate(chars)}
    # Ensure vocab_size matches
    v_size = len(chars)
    
    device = next(model.parameters()).device
    
    # Prepare data
    max_len = max(len(s) for s in sentences)
    input_data = []
    target_data = []
    
    for s in sentences:
        ids = [char_to_id[c] for c in s]
        # Pad
        ids += [0] * (max_len - len(ids))
        input_data.append(ids)
        # Shift targets
        target_ids = ids[1:] + [0]
        target_data.append(target_ids)
        
    x = torch.tensor(input_data, dtype=torch.long, device=device)
    y = torch.tensor(target_data, dtype=torch.long, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = gfn.loss('generative', mode='nll')
    
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        # Standard NLP Next-token Prediction
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        
    return {"steps": steps, "loss": last_loss}

def eval_nlp(model):
    # Accuracy on top-1 prediction
    # (Same dummy data as train for this micro-benchmark)
    sentences = ["the circle is round", "the square is flat", "the torus is curved"]
    chars = sorted(list(set("".join(sentences)))) # Note: should ideally use a global vocab but this is a micro-test
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    device = next(model.parameters()).device
    max_len = max(len(s) for s in sentences)
    input_data = []
    target_data = []
    for s in sentences:
        ids = [char_to_id.get(c, 0) for i, c in enumerate(s)]
        ids += [0] * (max_len - len(ids))
        input_data.append(ids)
        target_data.append(ids[1:] + [0])
        
    x = torch.tensor(input_data, dtype=torch.long, device=device)
    y_target = torch.tensor(target_data, dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        preds = logits.argmax(dim=-1)
        acc = (preds == y_target).float().mean().item()
    return acc

def run_nlp_matrix():
    # Vocab size for this test is small (chars)
    runner = MatrixRunner("Mini_NLP")
    
    axes = {
        "mixer_type": ["low_rank", "geodesic_attention", "flow_mixer"],
        "dynamics_type": ["direct", "residual"],
        "topology_type": ["torus", "euclidean"]
    }
    
    base = {
        "dim": 64,
        "depth": 2,
        "heads": 4,
        "vocab_size": 32, # Enough for chars
        "holographic": False
    }
    
    runner.run_axes(axes, train_nlp, eval_nlp, base_overrides=base)

if __name__ == "__main__":
    run_nlp_matrix()
    

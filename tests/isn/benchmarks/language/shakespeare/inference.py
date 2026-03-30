"""
isn.Inference Script
Load a trained GFN model and generate text.
"""

import os
import torch
import json
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
from gfn import isn

def load_model(checkpoint_path, config_path, vocab_size, device):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = isn.create(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        d_embedding=config['model']['d_embedding'],
        d_properties=config['model']['d_properties'],
        scanner_cls=isn.GFNScanner,
        world_cls=isn.GFNWorld,
        emitter_cls=isn.GFNEmitter
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path} (Loss: {checkpoint['loss']:.4f})")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = base_dir / "tests" / "isn" / "benchmarks" / "language" / "data" / "tinyshakespeare.txt"
    config_path = base_dir / "configs" / "shakespeare_config.json"
    checkpoint_path = base_dir / "checkpoints" / "shakespeare" / "best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # To get vocab mapping
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    model = load_model(checkpoint_path, config_path, vocab_size, device)

    # Inference loop
    while True:
        prompt = input("\nEnter prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
        
        # Tokenize prompt
        input_ids = torch.tensor([char_to_ix.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        
        print("\nGenerating...", end="", flush=True)
        # isn.Model.generate implementation
        generated_ids, _ = model.generate(
            input_ids, 
            max_length=200, 
            temperature=0.8
        )
        
        # Decode
        output_text = "".join([ix_to_char.get(i.item(), '') for i in generated_ids[0]])
        print(f"\n--- Result ---\n{output_text}\n--- End ---\n")

if __name__ == "__main__":
    main()

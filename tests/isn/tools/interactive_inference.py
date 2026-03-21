import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.models.model import Model as RealityModel
from gfn.realizations.isn.training.coherence_loss import MultiDimensionalLoss

def load_model(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    tokenizer_data = checkpoint['tokenizer']
    
    # Reconstruct tokenizer
    # Note: CharTokenizer needs the original text to build the vocab normally,
    # but we can bypass it by setting stoi/itos directly if we modify it or 
    # just use the saved data.
    class LoadedTokenizer:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        def encode(self, s):
            return [self.stoi[c] for c in s if c in self.stoi]
        def decode(self, l):
            return ''.join([self.itos[i] for i in l if i in self.itos and i > 2])

    tokenizer = LoadedTokenizer(tokenizer_data['stoi'], tokenizer_data['itos'])
    
    model = RealityModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['encoder']['d_model'],
        d_embedding=config['model']['entity']['d_embedding'],
        d_properties=config['model']['entity']['d_properties'],
        encoder_layers=config['model']['encoder']['n_layers'],
        decoder_layers=config['model']['decoder']['n_layers'],
        n_heads=config['model']['encoder']['n_heads'],
        d_ff=config['model']['encoder']['d_ff'],
        dropout=0.0,
        max_entities=config['model']['entity']['max_entities'],
        max_timesteps=config['model']['world']['max_timesteps'],
        coherence_threshold=config['model']['world']['coherence_threshold']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    
    # Find latest checkpoint if multiple exist
    checkpoints = list(checkpoint_dir.glob("epoch_*/reality_lang_weights*.pt"))
    if not checkpoints:
        print("No checkpoints found in " + str(checkpoint_dir))
        return
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    model, tokenizer = load_model(str(latest_checkpoint), device)
    
    print("\n" + "="*50)
    print("RealityModel Interactive Inference")
    print("="*50)
    print(f"Model loaded: {latest_checkpoint.name}")
    print("Commands: 'exit' to quit, '/temp [val]' to change creativity (default 1.0).\n")
    
    current_temp = 0.5
    
    while True:
        prompt = input(f"[{current_temp}] Prompt: ")
        if prompt.lower() == 'exit':
            break
        
        if prompt.startswith('/temp'):
            try:
                current_temp = float(prompt.split()[1])
                print(f"Temperature set to {current_temp}")
            except:
                print("Invalid temperature format. Use /temp 0.8")
            continue
            
        if not prompt:
            continue
            
        # Encode
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids]).to(device)
        
        print("\nGenerating...")
        with torch.no_grad():
            # Generate
            # RealityModel.generate returns the FULL sequence (prompt + new)
            full_sequence_ids, info = model.generate(
                input_ids, 
                max_length=1000,
                temperature=current_temp
            )
            
            # Extract only the NEWLY generated tokens
            new_tokens = full_sequence_ids[0, len(prompt_ids):].tolist()
            output_text = tokenizer.decode(new_tokens)
            
            print("-" * 30)
            print(f"Prompt: {prompt}")
            print(f"Generated: {output_text}")
            print("-" * 30)
            print(f"World Coherence: {info['coherence'].mean().item():.4f}")
            print(f"World Coherence: {info['coherence'].mean().item():.4f}")
            print("="*50 + "\n")

if __name__ == "__main__":
    run_inference()

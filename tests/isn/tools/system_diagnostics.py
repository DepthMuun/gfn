import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.models.model import Model as RealityModel

def diagnose_identity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init model with random weights (mimic Epoch 1)
    vocab_size = 68
    model = RealityModel(vocab_size=vocab_size, d_model=512, d_embedding=256)
    model.to(device)
    model.eval()
    
    # Prompt: "hi" (encoded as random ints for now)
    input_ids = torch.tensor([[10, 20]], device=device) # 'h', 'i'
    
    with torch.no_grad():
        # Check type classification
        impulses = model.innerver(input_ids)
        type_logits = model.world_network.classify_type(impulses)
        type_idx = type_logits.argmax(dim=-1)
        
        print(f"Input IDs: {input_ids.cpu().numpy()}")
        print(f"Assigned Types: {type_idx.cpu().numpy()}")
        
        # Check forward
        outputs = model(input_ids, return_world_state=True)
        print(f"Emitted Logits Shape: {outputs['logits'].shape}")
        if outputs['logits'].size(1) > 0:
            emitted_ids = outputs['logits'].argmax(dim=-1)
            print(f"Emitted IDs: {emitted_ids.cpu().numpy()}")
        else:
            print("No emissions triggered.")

if __name__ == "__main__":
    diagnose_identity()

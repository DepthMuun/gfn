import torch
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from gfn.realizations.isn.models.model import Model as RealityModel

def run_profile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on {device}")
    
    model = RealityModel(
        vocab_size=68,
        d_model=256,
        d_embedding=256,
        d_properties=64,
        encoder_layers=2,
        decoder_layers=2,
        n_heads=4,
        d_ff=1024,
        max_entities=100,
        max_timesteps=50,
        coherence_threshold=0.7
    ).to(device)
    
    # Escala moderada/alta para medir overhead
    batch_size = 32
    seq_len = 512
    input_ids = torch.randint(0, 68, (batch_size, seq_len), device=device)
    
    print("Warmup...")
    with torch.no_grad():
        model(input_ids)
        
    print(f"Profiling B={batch_size}, L={seq_len}...")
    pr = cProfile.Profile()
    pr.enable()
    
    with torch.no_grad():
        model(input_ids)
        
    pr.disable()
    s = io.StringIO()
    sortby = 'cumtime' # Ordenar por Tiempo Acumulativo
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(40) # Imprimir el top 40 de funciones ms costosas
    print(s.getvalue())

if __name__ == "__main__":
    run_profile()

"""
Configuración del modelo GSSM para ARC-AGI-2
"""

import sys
from pathlib import Path

# Añadir gfn al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

from gfn import create


def get_arc_agi2_config(
    dim: int = 900,  # Holographic: state IS the output (grid size)
    heads: int = 8,
    depth: int = 6,
    max_grid_size: int = 30,
    color_values: int = 10
) -> dict:
    """
    Configuración optimizada de GSSM para ARC-AGI-2.
    
    Args:
        dim: Dimensión del modelo
        heads: Número de heads
        depth: Profundidad (número de capas)
        max_grid_size: Tamaño máximo de grid (30x30)
        color_values: Valores de color (0-9)
    
    Returns:
        Dict con configuración completa
    """
    
    # ARC-AGI-2 tiene grids de hasta 30x30 con 10 colores
    # 30x30 = 900 celdas
    vocab_size = color_values
    grid_size = max_grid_size * max_grid_size  # 900
    
    config = {
        'vocab_size': vocab_size,
        'dim': 64,  # XOR proven: dim=8, escalamos a 64 para grids
        'heads': heads,
        'depth': depth,
        'max_seq_len': grid_size,
        
        # Embedding: modo continuo para grids
        'embedding_mode': 'continuous',
        'continuous_input_dim': grid_size,  # 900 para 30x30
        'store_full_sequence': True,  # Need all timesteps for few-shot
        
        # Topology: Torus para estabilidad
        'physics': {
            'embedding': {
                'type': 'functional',
                'mode': 'continuous',
                'coord_dim': grid_size,
                'impulse_scale': 80.0,  # XOR proven: 80.0 (critical!)
            },
            'readout': {
                'type': 'implicit',  # XOR proven: implicit readout
                'coord_dim': 16,      # XOR proven: coord_dim=16
                'out_dim': grid_size,  # Project to grid size
            },
            'topology': {
                'type': 'torus',
                'R': 3.0,
                'r': 1.0,
                'learnable_R': True,
                'learnable_r': True
            },
            'stability': {
                'base_dt': 0.05,  # Conservador para ARC
                'dt_min': 0.0001,
                'dt_max': 0.2,
                'friction': 2.0,  # XOR proven: 2.0 (critical!)
                'velocity_saturation': 15.0,  # XOR proven: 15.0
                'curvature_clamp': 1.0
            },
            'active_inference': {
                'enabled': True,
                'hysteresis': {
                    'enabled': False,
                    'strength': 0.1,
                    'decay': 0.9
                },
                'curiosity': {
                    'enabled': False,
                    'strength': 0.01
                },
                'stochasticity': {
                    'enabled': False  # Desactivado para determinismo
                }
            },
            'integrator': {
                'type': 'leapfrog',  # Default symplectic
                'adaptive_dt': False
            },
            'losses': {
                'lambda_geo': 0.001,
                'lambda_ham': 0.0,
                'lambda_kin': 0.001
            }
        },
        
        # Readout: XOR proven settings
        'readout_type': 'implicit',
        'readout_hidden_dim': 64,
        'readout_out_dim': grid_size  # Project to grid size
    }
    
    return config


def create_arc_agi2_model(config: dict = None, device: str = 'cuda'):
    """
    Crea modelo GSSM configurado para ARC-AGI-2.
    
    Args:
        config: Configuración (usa default si None)
        device: Dispositivo ('cuda' o 'cpu')
    
    Returns:
        Modelo GSSM listo para entrenar
    """
    if config is None:
        config = get_arc_agi2_config()
    
    # Crear modelo con gfn.create
    model = create(
        'gssm',
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        heads=config['heads'],
        depth=config['depth'],
        max_seq_len=config['max_seq_len'],
        embedding_mode=config['embedding_mode'],
        continuous_input_dim=config['continuous_input_dim'],
        physics=config['physics'],
        readout_type=config['readout_type'],
        readout_hidden_dim=config.get('readout_hidden_dim', 128),
        readout_out_dim=config.get('readout_out_dim', config['max_seq_len']),  # Project to grid size
        holographic=True,  # SIEMPRE activado
        device=device
    )
    
    return model


# Configuraciones predefinidas (XOR-proven scaling)
CONFIGS = {
    'small': {
        'dim': 64,
        'heads': 4,
        'depth': 4
    },
    'medium': {
        'dim': 128,
        'heads': 8,
        'depth': 6
    },
    'large': {
        'dim': 256,
        'heads': 16,
        'depth': 8
    }
}


def get_config(name: str = 'medium') -> dict:
    """Obtiene configuración predefinida."""
    base_config = get_arc_agi2_config()
    if name in CONFIGS:
        base_config.update(CONFIGS[name])
    return base_config


if __name__ == "__main__":
    # Test de configuración
    print("Testing ARC-AGI-2 Model Configuration...")
    
    config = get_arc_agi2_config()
    print(f"Config: {config}")
    
    print("\nConfiguration test passed!")

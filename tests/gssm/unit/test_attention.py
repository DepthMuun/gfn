"""
tests/unit/test_attention.py
Tests del GeodesicAttentionMixer.

NOTA: GeodesicAttentionMixer.forward() devuelve una tupla (x_mixed, aux)
      conforme a la API de FlowMixer. Los tests verifican la forma de x_mixed.
"""
import pytest
import torch
from gfn.realizations.gssm.models.components.mixer import GeodesicAttentionMixer
from gfn.realizations.gssm.config.schema import PhysicsConfig


@pytest.fixture
def default_config():
    cfg = PhysicsConfig()
    cfg.topology.type = 'torus'
    return cfg


def test_attention_routing_topological():
    """
    Geodesic attention debe despachar distancias correctamente.
    Formato de entrada al mixer (viene de ManifoldLayer): [B, H, head_dim].
    Salida del mixer: ([B, dim], [B, dim]).
    """
    dim, heads = 16, 4  # head_dim = 4
    attn = GeodesicAttentionMixer(dim, heads=heads, topology='euclidean')

    B, H, head_dim = 2, heads, dim // heads
    x = torch.randn(B, H, head_dim)
    out = attn(x)

    # El mixer devuelve (tensor, aux_info) o tensor directo — manejar ambos
    x_out = out[0] if isinstance(out, tuple) else out
    assert x_out.shape == (B, dim), \
        f"Attention mixer salida incorrecta: esperado {(B, dim)}, got {x_out.shape}"


def test_attention_sparsity():
    """
    Temperatura baja debe crear asignaciones near-sparse sin NaN.
    """
    dim, heads = 16, 4
    head_dim = dim // heads
    attn = GeodesicAttentionMixer(dim, heads=heads, topology='euclidean', temperature=0.01)

    B = 2
    x = torch.zeros(B, heads, head_dim)
    x[0, 0, :] = 100.0

    out = attn(x)
    x_out = out[0] if isinstance(out, tuple) else out
    assert not torch.isnan(x_out).any(), \
        "Geodesic Attention colapsó con distancias grandes (Falta Softmax numéricamente estable)"


def test_attention_output_requires_grad():
    """El output del attention mixer debe propagar gradientes."""
    dim, heads = 8, 2  # head_dim = 4
    attn = GeodesicAttentionMixer(dim, heads=heads, topology='torus')
    x = torch.randn(2, heads, dim // heads, requires_grad=True)
    out = attn(x)
    x_out = out[0] if isinstance(out, tuple) else out
    x_out.sum().backward()
    assert x.grad is not None


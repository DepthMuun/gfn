import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add the gfn/cuda directory so gfn_cuda.pyd/.so is importable
GFN_CUDA_DIR = PROJECT_ROOT / "gfn" / "cuda"
if str(GFN_CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(GFN_CUDA_DIR))

@pytest.fixture(scope="session")
def device():
    """Global device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pytest
import torch
import unittest.mock as mock
from gfn import Leapfrog as LeapfrogIntegrator
from gfn import LowRankChristoffel

class TestErrorRecovery:
    
    def test_cuda_fallback_on_failure(self, logger, device):
        """
        Verify that if CUDA kernel fails (raises Exception), 
        the integrator catches it, logs a warning, and falls back to Python.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        dim = 16
        christoffel = LowRankChristoffel(dim, rank=4).to(device)
        integrator = LeapfrogIntegrator(christoffel, dt=0.1).to(device)
        
        x = torch.randn(2, dim, device=device)
        v = torch.randn(2, dim, device=device)
        
        # Patch the new dispatcher for CUDA kernels to return a mock failing kernel
        with mock.patch('gfn.computation.kernels.integrator_kernels._get_cuda_integrators') as mock_get:
            mock_kernel = mock.MagicMock(side_effect=RuntimeError("Simulated CUDA Kernel Explosion"))
            mock_get.return_value = (None, None, mock_kernel)
            with mock.patch('gfn.computation.backend.registry.BackendRegistry.is_cuda_active', return_value=True):
                # We must also ensure x.is_cuda is True (it is) and collect_christ is False
                
                # Run
                # Should NOT raise RuntimeError
                # Should return valid result (via Python fallback)
                try:
                    x_out, v_out, _ = integrator(x, v, collect_christ=False)
                    
                    logger.log_metric(0, "fallback_success", 1)
                    print("\n[Recovery] Caught CUDA error and recovered successfully.")
                    
                except RuntimeError as e:
                    if "Simulated" in str(e):
                        pytest.fail("Integrator did NOT catch the CUDA exception!")
                    else:
                        raise e
                
                # Verify mock was called
                mock_get.assert_called_once()


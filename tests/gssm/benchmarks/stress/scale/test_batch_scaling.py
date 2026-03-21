
import pytest
import torch
from gfn import Manifold

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for OOM test")
class TestStressScaling:
    
    def test_batch_scaling_limit(self, logger, device):
        """
        Increase batch size exponentialy until OOM.
        Logs maximum safe batch size.
        """
        vocab_size = 100
        dim = 1024 # Large model
        depth = 6
        seq_len = 128
        
        try:
            model = Manifold(vocab_size=vocab_size, dim=dim, depth=depth).to(device)
        except RuntimeError:
            pytest.skip("GPU too small for initial model allocation")
            
        max_safe_batch = 0
        batch_size = 1
        
        logger.log_metric(0, "model_params", sum(p.numel() for p in model.parameters()))
        
        while batch_size <= 65536: # Hard limit
            try:
                # Alloc Input
                x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                
                # Forward Pass
                torch.cuda.empty_cache()
                _ = model(x)
                
                # Success
                max_safe_batch = batch_size
                logger.log_metric(batch_size, "status", "pass")
                print(f"[Stress] Batch {batch_size} OK")
                
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[Stress] OOM at Batch {batch_size}")
                    logger.log_metric(batch_size, "status", "OOM")
                    break
                else:
                    raise e
                    
        print(f"[Stress] Max Safe Batch Size: {max_safe_batch}")
        logger.log_metric(0, "max_safe_batch", max_safe_batch)
        
        # We expect at least batch=1 to work on any reasonable GPU
        assert max_safe_batch >= 1


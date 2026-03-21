import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDistanceLoss(nn.Module):
    """
    Replaces standard computational CrossEntropy.
    Measures the purely topological alignment (Cosine Distance) between an 
    entity's physical embedding and the ideal 'Base Imprint' of the target token.
    """
    def __init__(self, use_l2: bool = False):
        super().__init__()
        self.use_l2 = use_l2

    def forward(self, emitted_embeddings: torch.Tensor, target_ids: torch.Tensor, vocab_basis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emitted_embeddings: The actual D-dimensional vectors produced by the simulator (batch, seq_len, d_emb)
            target_ids: The ground truth integer IDs (batch, seq_len)
            vocab_basis: The Emitter's projection weight matrix acting as the coordinate basis (vocab_size, d_emb)
        """
        # Obtain the ideal geometric position for the target concepts
        ideal_embeddings = vocab_basis[target_ids] # Shape: (batch, seq_len, d_emb)
        
        mask = (target_ids != 0) # Assumes 0 is padding
        
        if not mask.any():
            return torch.tensor(0.0, device=emitted_embeddings.device, requires_grad=True)

        if self.use_l2:
            # L2 Metric Space (Euclidean)
            distances = F.mse_loss(emitted_embeddings, ideal_embeddings, reduction='none')
            distances = distances.mean(dim=-1) # average over embedding dim
        else:
            # Topologic Angle (Cosine)
            cos_sim = F.cosine_similarity(emitted_embeddings, ideal_embeddings, dim=-1)
            # Distance is 1 - CosineSimilarity
            distances = 1.0 - cos_sim
            
        # Mask out padding and mean
        return (distances * mask).sum() / mask.sum()

import torch
import torch.nn as nn
import torch.nn.functional as F

class ThresholdModulationLoss(nn.Module):
    """
    Trains the Threshold Entity Emitter by modulating the internal graph energy.
    Uses a Hinge Loss to push interaction energies over the threshold when emission 
    is required, and pull them under when silence is required.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, energy_trace: torch.Tensor, emission_target_mask: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy_trace: The kinetic energy of collisions at each timestep (batch, seq_len)
            emission_target_mask: Boolean/Float mask (batch, seq_len) where 1 means "Should Emit" and 0 means "Should Stay Silent".
            threshold: The scalar Parameter from the Emitter defining the emission barrier.
        """
        # If the target is to EMIT (mask == 1):
        # We want energy > threshold + margin  -> Loss = relu(threshold + margin - energy)
        emit_loss = F.relu(threshold + self.margin - energy_trace)
        
        # If the target is to STAY SILENT (mask == 0):
        # We want energy < threshold - margin -> Loss = relu(energy - threshold + margin)
        silence_loss = F.relu(energy_trace - threshold + self.margin)
        
        # Combine using the mask
        total_loss = (emission_target_mask * emit_loss) + ((1.0 - emission_target_mask) * silence_loss)
        
        return total_loss.mean()

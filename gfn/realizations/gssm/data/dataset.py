import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """Simple sequence dataset for (X, Y) pairs."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

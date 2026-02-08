from .lowrank import LowRankChristoffel
from .reactive import ReactiveChristoffel
from .hyper import HyperChristoffel
from .analytical import EuclideanChristoffel, HyperbolicChristoffel, SphericalChristoffel
from .toroidal import ToroidalChristoffel
from ..integrators import HeunIntegrator, LeapfrogIntegrator, DormandPrinceIntegrator, EulerIntegrator
import torch
import torch.nn as nn

class TimeDilationHead(nn.Module):
    def __init__(self, dim, range_min=0.1, range_max=5.0, topology=0):
        super().__init__()
        self.range_min = range_min
        self.range_max = range_max
        self.topology = topology
        input_dim = 2 * dim if topology == 1 else dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        with torch.no_grad():
            nn.init.constant_(self.net[2].bias, 2.0)
            nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
            nn.init.xavier_uniform_(self.net[2].weight, gain=0.1)
    def forward(self, x, v=None, f=None):
        if self.topology == 1:
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        scale = self.net(x)
        return self.range_min + scale * (self.range_max - self.range_min)

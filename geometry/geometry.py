from dataclasses import dataclass
import torch


@dataclass
class FairGeometry:
    D_X: torch.Tensor
    L: torch.Tensor
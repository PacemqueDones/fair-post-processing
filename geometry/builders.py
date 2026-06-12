from dataclasses import dataclass
import torch


@dataclass
class FairGeometry:
    D_X: torch.Tensor
    W: torch.Tensor
    L: torch.Tensor

    tau: float | None = None
    theta: float | None = None
    distance_name: str | None = None
from dataclasses import dataclass
import torch


@dataclass
class FairGeometry:
    D_X: torch.Tensor
    W: torch.Tensor
    L: torch.Tensor

    laplacian_type: str
    distance_name: str = "mahalanobis"

    theta: float = 1.0
    tau_quantile: float = 0.25
    tau: float | None = None
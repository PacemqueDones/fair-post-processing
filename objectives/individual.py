import torch
import torch.nn.functional as F
from .objective import Objective

class LaplacianFairnessObjective(Objective):
    name = "laplacian_fairness"

    def __init__(
        self,
        L,
        W=None,
        fairness_weight=1.0,
        ce_weight=0,
        normalize="samples",
        symmetrize=False,
        eps=1e-8,
    ):
        valid_normalizations = {None, "samples", "edges"}

        if normalize not in valid_normalizations:
            raise ValueError("normalize must be None, 'samples' or 'edges'.")

        if normalize == "edges" and W is None:
            raise ValueError("W must be provided when normalize='edges'.")

        L = L.float()

        if symmetrize:
            L = 0.5 * (L + L.T)

        self.L = L
        self.W = None if W is None else W.float()

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.normalize = normalize
        self.symmetrize = symmetrize
        self.eps = eps

    def __call__(self, logits, y_true, sensitive_attr):
        F_scores = logits.float()

        if F_scores.ndim == 1:
            F_scores = F_scores.view(-1, 1)

        L = self.L.to(device=F_scores.device, dtype=F_scores.dtype,)

        if L.shape[0] != F_scores.shape[0]:
            raise ValueError(
                f"L tem shape {L.shape}, mas logits têm shape "
                f"{F_scores.shape}. O Laplaciano precisa ser "
                "gerado sobre o mesmo conjunto usado no fit."
            )

        # Forma densa compatível com TorchJD:
        # tr(F^T L F) = sum(F * (L F))
        LF = L @ F_scores
        fairness = torch.sum(F_scores * LF)

        if self.normalize == "samples":
            fairness = fairness / F_scores.shape[0]

        elif self.normalize == "edges":
            W = self.W.to(device=F_scores.device, dtype=F_scores.dtype,)

            if W.shape != L.shape:
                raise ValueError(f"W tem shape {W.shape}, mas L tem shape {L.shape}.")

            edge_weight_sum = W.sum()

            fairness = (2.0 * fairness / edge_weight_sum.clamp_min(self.eps))

        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, y_true)
        else:
            ce = 0

        loss = (self.fairness_weight * fairness + self.ce_weight * ce)

        return [loss], [self.name] 
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from .objective import Objective

class CrossEntropyObjective(Objective):
    name = "cross_entropy"

    def __call__(self, logits, y_true, sensitive_attr):
        return F.cross_entropy(logits, y_true)

class DemographicParityObjective(Objective):
    name = "demographic_parity"

    def __init__(self, fairness_weight=1.0, ce_weight=0.0):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        group0 = preds_pos[sensitive_attr == 0]
        group1 = preds_pos[sensitive_attr == 1]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=logits.device)
        else:
            fairness = torch.abs(group0.mean() - group1.mean())

        ce = F.cross_entropy(logits, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
    
class EqualityOpportunityObjective(Objective):
    name = "equality_opportunity"

    def __init__(self, fairness_weight=1.0, ce_weight=0):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = (y_true == 1)

        group0 = preds_pos[(sensitive_attr == 0) & mask_pos]
        group1 = preds_pos[(sensitive_attr == 1) & mask_pos]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=logits.device)
        else:
            fairness = torch.abs(group0.mean() - group1.mean())

        ce = F.cross_entropy(logits, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
    
class DemographicParityKLObjective(Objective):
    name = "demographic_parity_kl"

    def __init__(self, fairness_weight=1.0, ce_weight=0, eps=1e-7):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def _kl_bern(self, p0, p1):
        p0 = torch.clamp(p0, self.eps, 1 - self.eps)
        p1 = torch.clamp(p1, self.eps, 1 - self.eps)

        return p0 * torch.log(p0 / p1) + (1 - p0) * torch.log((1 - p0) / (1 - p1))

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        group0 = preds_pos[sensitive_attr == 0]
        group1 = preds_pos[sensitive_attr == 1]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=logits.device)
        else:
            p0 = group0.mean()
            p1 = group1.mean()
            fairness = self._kl_bern(p0, p1) + self._kl_bern(p1, p0)

        ce = F.cross_entropy(logits, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce

class EqualityOpportunityKLObjective(Objective):
    name = "equality_opportunity_kl"

    def __init__(self, fairness_weight=1.0, ce_weight=0, eps=1e-7):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def _kl_bern(self, p0, p1):
        p0 = torch.clamp(p0, self.eps, 1 - self.eps)
        p1 = torch.clamp(p1, self.eps, 1 - self.eps)

        return p0 * torch.log(p0 / p1) + (1 - p0) * torch.log((1 - p0) / (1 - p1))

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = (y_true == 1)

        group0 = preds_pos[(sensitive_attr == 0) & mask_pos]
        group1 = preds_pos[(sensitive_attr == 1) & mask_pos]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=logits.device)
        else:
            p0 = group0.mean()
            p1 = group1.mean()
            fairness = self._kl_bern(p0, p1) + self._kl_bern(p1, p0)

        ce = F.cross_entropy(logits, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
    
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
            raise ValueError(
                "normalize deve ser None, 'samples' ou 'edges'."
            )

        if normalize == "edges" and W is None:
            raise ValueError(
                "W deve ser fornecida quando normalize='edges'."
            )

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

        L = self.L.to(
            device=F_scores.device,
            dtype=F_scores.dtype,
        )

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
            W = self.W.to(
                device=F_scores.device,
                dtype=F_scores.dtype,
            )

            if W.shape != L.shape:
                raise ValueError(
                    f"W tem shape {W.shape}, mas L tem shape {L.shape}."
                )

            edge_weight_sum = W.sum()

            fairness = (
                2.0 * fairness
                / edge_weight_sum.clamp_min(self.eps)
            )

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
    
class WassersteinEqualityOpportunityObjective(Objective):
    name = "equality_opportunity_wasserstein"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=1,
        blur=0.05,
        scaling=0.9,
        debias=True,
        backend="auto",
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

        self.wasserstein = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            group0 = group0.reshape(-1, 1)
            group1 = group1.reshape(-1, 1)

            fairness = self.wasserstein(group0, group1)

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
    
class WassersteinEqualityOpportunityObjective(Objective):
    name = "equality_opportunity_wasserstein"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=1,
        blur=0.05,
        scaling=0.9,
        debias=True,
        backend="auto",
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

        self.wasserstein = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            group0 = group0.reshape(-1, 1)
            group1 = group1.reshape(-1, 1)

            fairness = self.wasserstein(group0, group1)

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
    
class WassersteinEqualityOpportunityQuantileObjective(Objective):
    name = "equality_opportunity_wasserstein_quantile"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=2,
        num_quantiles=10**4,
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.p = p
        self.num_quantiles = num_quantiles

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]
        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            quantile_levels = torch.linspace(
                0.0,
                1.0,
                self.num_quantiles,
                device=logits.device,
                dtype=logits.dtype,
            )

            q0 = torch.quantile(group0, quantile_levels)
            q1 = torch.quantile(group1, quantile_levels)

            fairness = torch.mean(
                torch.abs(q0 - q1) ** self.p
            )

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
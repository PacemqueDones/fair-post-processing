import torch
import torch.nn.functional as F
from .objective import Objective

class CrossEntropyObjective(Objective):
    name = "cross_entropy"

    def __call__(self, scores, y_true, sensitive_attr):
        return F.cross_entropy(scores, y_true)

class DemographicParityObjective(Objective):
    name = "demographic_parity"

    def __init__(self, fairness_weight=1.0, ce_weight=0.001):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

    def __call__(self, scores, y_true, sensitive_attr):
        preds_pos = torch.softmax(scores, dim=1)[:, 1]

        group0 = preds_pos[sensitive_attr == 0]
        group1 = preds_pos[sensitive_attr == 1]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=scores.device)
        else:
            fairness = torch.abs(group0.mean() - group1.mean())

        ce = F.cross_entropy(scores, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
    
class EqualityOpportunityObjective(Objective):
    name = "equality_opportunity"

    def __init__(self, fairness_weight=1.0, ce_weight=0.001):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

    def __call__(self, scores, y_true, sensitive_attr):
        preds_pos = torch.softmax(scores, dim=1)[:, 1]

        mask_pos = (y_true == 1)

        group0 = preds_pos[(sensitive_attr == 0) & mask_pos]
        group1 = preds_pos[(sensitive_attr == 1) & mask_pos]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=scores.device)
        else:
            fairness = torch.abs(group0.mean() - group1.mean())

        ce = F.cross_entropy(scores, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
    
class DemographicParityKLObjective(Objective):
    name = "demographic_parity_kl"

    def __init__(self, fairness_weight=1.0, ce_weight=0.001, eps=1e-7):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def _kl_bern(self, p0, p1):
        p0 = torch.clamp(p0, self.eps, 1 - self.eps)
        p1 = torch.clamp(p1, self.eps, 1 - self.eps)

        return p0 * torch.log(p0 / p1) + (1 - p0) * torch.log((1 - p0) / (1 - p1))

    def __call__(self, scores, y_true, sensitive_attr):
        preds_pos = torch.softmax(scores, dim=1)[:, 1]

        group0 = preds_pos[sensitive_attr == 0]
        group1 = preds_pos[sensitive_attr == 1]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=scores.device)
        else:
            p0 = group0.mean()
            p1 = group1.mean()
            fairness = self._kl_bern(p0, p1) + self._kl_bern(p1, p0)

        ce = F.cross_entropy(scores, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce

class EqualityOpportunityKLObjective(Objective):
    name = "equality_opportunity_kl"

    def __init__(self, fairness_weight=1.0, ce_weight=0.001, eps=1e-7):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def _kl_bern(self, p0, p1):
        p0 = torch.clamp(p0, self.eps, 1 - self.eps)
        p1 = torch.clamp(p1, self.eps, 1 - self.eps)

        return p0 * torch.log(p0 / p1) + (1 - p0) * torch.log((1 - p0) / (1 - p1))

    def __call__(self, scores, y_true, sensitive_attr):
        preds_pos = torch.softmax(scores, dim=1)[:, 1]

        mask_pos = (y_true == 1)

        group0 = preds_pos[(sensitive_attr == 0) & mask_pos]
        group1 = preds_pos[(sensitive_attr == 1) & mask_pos]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = torch.tensor(0.0, device=scores.device)
        else:
            p0 = group0.mean()
            p1 = group1.mean()
            fairness = self._kl_bern(p0, p1) + self._kl_bern(p1, p0)

        ce = F.cross_entropy(scores, y_true)

        return self.fairness_weight * fairness + self.ce_weight * ce
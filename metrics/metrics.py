import torch
from .metric import Metric

class AccuracyMetric(Metric):
    name = "acc"
    direction = "max"   # importante para TOPSIS/Pareto depois
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        return (y_pred == y_true).float().mean().item()

class BalancedAccuracyMetric(Metric):
    name = "bacc"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        classes = torch.unique(y_true)

        recalls = []

        for c in classes:
            mask = y_true == c

            # Recall da classe c:
            # entre todos os exemplos verdadeiros da classe c,
            # quantos foram previstos corretamente como c?
            recall_c = (y_pred[mask] == c).float().mean()

            recalls.append(recall_c)

        return torch.stack(recalls).mean().item()
    
class PrecisionMetric(Metric):
    name = "prec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        pred_pos = (y_pred == 1)

        if pred_pos.sum() == 0:
            return 0.0

        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        pp = pred_pos.sum().float()

        precision = tp / pp
        return precision.item()

class RecallMetric(Metric):
    name = "rec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        positive = (y_true == 1)
        if positive.sum() == 0:
            return 0.0
        return y_pred[positive].float().mean().item()
    
class F1ScoreMetric(Metric):
    name = "f1"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        fn = ((y_pred == 0) & (y_true == 1)).sum().float()

        precision_denom = tp + fp
        recall_denom = tp + fn

        if precision_denom == 0 or recall_denom == 0:
            return 0.0

        precision = tp / precision_denom
        recall = tp / recall_denom

        denom = precision + recall
        if denom == 0:
            return 0.0

        f1 = 2 * (precision * recall) / denom
        return f1.item()

class DemographicParityMetric(Metric):
    name = "ddp"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        g0 = (sensitive_attr == 0)
        g1 = (sensitive_attr == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        rate0 = y_pred[g0].float().mean()
        rate1 = y_pred[g1].float().mean()
        return torch.abs(rate0 - rate1).item()
    
class EqualityOpportunityMetric(Metric):
    name = "deo"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        g0 = (sensitive_attr == 0) & (y_true == 1)
        g1 = (sensitive_attr == 1) & (y_true == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        tpr0 = y_pred[g0].float().mean()
        tpr1 = y_pred[g1].float().mean()
        return torch.abs(tpr0 - tpr1).item()

class GeneralizedEntropyMetric(Metric):
    name = "ge"
    direction = "min"
    type = "fairness"

    def __init__(self, alpha=2.0, eps=1e-12):
        self.alpha = alpha
        self.eps = eps

    def _benefit(self, y_true, y_pred):
        """
        Benefício individual do paper:

            b_i = y_pred_i - y_true_i + 1

        Para classificação binária:
            TN -> 1
            TP -> 1
            FP -> 2
            FN -> 0
        """
        y_true = y_true.view(-1).float()
        y_pred = y_pred.view(-1).float()

        return y_pred - y_true + 1.0

    def _generalized_entropy(self, benefits):
        b = benefits.view(-1).float()

        if b.numel() == 0:
            return torch.tensor(0.0, device=b.device)

        mu = b.mean()

        if mu <= self.eps:
            return torch.tensor(0.0, device=b.device)

        ratio = b / (mu + self.eps)

        if self.alpha == 0:
            ge = torch.mean(torch.log((mu + self.eps) / (b + self.eps)))

        elif self.alpha == 1:
            ge = torch.mean(ratio * torch.log(ratio + self.eps))

        else:
            ge = torch.mean(ratio.pow(self.alpha) - 1.0)
            ge = ge / (self.alpha * (self.alpha - 1.0))

        return ge

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        benefits = self._benefit(y_true, y_pred)
        return self._generalized_entropy(benefits).item()
    
class GeneralizedEntropyBetweenMetric(GeneralizedEntropyMetric):
    name = "ge_between"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        if sensitive_attr is None:
            raise ValueError("sensitive_attr é necessário para calcular ge_between.")

        benefits = self._benefit(y_true, y_pred)

        b = benefits.view(-1).float()
        s = sensitive_attr.view(-1)

        if b.numel() == 0:
            return 0.0

        mu = b.mean()

        if mu <= self.eps:
            return 0.0

        between = torch.tensor(0.0, device=b.device)

        n = b.numel()
        groups = torch.unique(s)

        for g in groups:
            mask = s == g

            if mask.sum() == 0:
                continue

            b_g = b[mask]
            n_g = b_g.numel()
            mu_g = b_g.mean()

            ratio_g = mu_g / (mu + self.eps)

            if self.alpha == 0:
                # Caso limite alpha -> 0
                # Forma compatível com decomposição GE(0)
                between = between + (n_g / n) * torch.log((mu + self.eps) / (mu_g + self.eps))

            elif self.alpha == 1:
                # Caso limite alpha -> 1, índice de Theil
                between = between + (n_g / n) * ratio_g * torch.log(ratio_g + self.eps)

            else:
                between = between + (n_g / n) * (ratio_g.pow(self.alpha) - 1.0)

        if self.alpha not in [0, 1]:
            between = between / (self.alpha * (self.alpha - 1.0))

        return between.item()
    
class GeneralizedEntropyWithinMetric(GeneralizedEntropyMetric):
    name = "ge_within"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None):
        if sensitive_attr is None:
            raise ValueError("sensitive_attr é necessário para calcular ge_within.")

        benefits = self._benefit(y_true, y_pred)

        b = benefits.view(-1).float()
        s = sensitive_attr.view(-1)

        if b.numel() == 0:
            return 0.0

        mu = b.mean()

        if mu <= self.eps:
            return 0.0

        within = torch.tensor(0.0, device=b.device)

        n = b.numel()
        groups = torch.unique(s)

        for g in groups:
            mask = s == g

            if mask.sum() == 0:
                continue

            b_g = b[mask]
            n_g = b_g.numel()
            mu_g = b_g.mean()

            if mu_g <= self.eps:
                continue

            ge_g = self._generalized_entropy(b_g)

            if self.alpha == 0:
                # Para alpha = 0, o peso muda.
                weight = n_g / n

            elif self.alpha == 1:
                # Para alpha = 1, o peso é proporcional à média relativa.
                weight = (n_g / n) * (mu_g / (mu + self.eps))

            else:
                weight = (n_g / n) * (mu_g / (mu + self.eps)).pow(self.alpha)

            within = within + weight * ge_g

        return within.item()
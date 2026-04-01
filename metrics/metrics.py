import torch
from .metric import Metric

class AccuracyMetric(Metric):
    name = "acc"
    direction = "max"   # importante para TOPSIS/Pareto depois
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        return (y_pred == y_true).float().mean().item()
    
class PrecisionMetric(Metric):
    name = "prec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
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

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        positive = (y_true == 1)
        if positive.sum() == 0:
            return 0.0
        return y_pred[positive].float().mean().item()
    
class F1ScoreMetric(Metric):
    name = "f1"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
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

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        g0 = (sensitive_attr == 0)
        g1 = (sensitive_attr == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        rate0 = y_pred[g0].float().mean()
        rate1 = y_pred[g1].float().mean()
        return torch.abs(rate0 - rate1).item()
    
class DEOMetric(Metric):
    name = "deo"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        g0 = (sensitive_attr == 0) & (y_true == 1)
        g1 = (sensitive_attr == 1) & (y_true == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        tpr0 = y_pred[g0].float().mean()
        tpr1 = y_pred[g1].float().mean()
        return torch.abs(tpr0 - tpr1).item()
import torch
from .soft_metric import SoftMetric

class SoftAccuracyMetric(SoftMetric):
    name = "acc"

    def __call__(self, y_true, scores, sensitive_attr=None):
        p = scores[:, 1]
        y = y_true.float()

        acc = y * p + (1 - y) * (1 - p)
        return acc.mean()


class SoftPrecisionMetric(SoftMetric):
    name = "precision"

    def __call__(self, y_true, scores, sensitive_attr=None):
        p = scores[:, 1]
        y = y_true.float()

        tp = (p * y).sum()
        pp = p.sum()

        return tp / (pp + 1e-8)


class SoftRecallMetric(SoftMetric):
    name = "rec"

    def __call__(self, y_true, scores, sensitive_attr=None):
        p = scores[:, 1]
        y = y_true.float()

        tp = (p * y).sum()
        pos = y.sum()

        return tp / (pos + 1e-8)


class SoftF1ScoreMetric(SoftMetric):
    name = "f1"

    def __call__(self, y_true, scores, sensitive_attr=None):
        p = scores[:, 1]
        y = y_true.float()

        tp = (p * y).sum()
        pp = p.sum()
        pos = y.sum()

        precision = tp / (pp + 1e-8)
        recall = tp / (pos + 1e-8)

        return 2 * precision * recall / (precision + recall + 1e-8)
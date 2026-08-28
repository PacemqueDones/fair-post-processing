import torch

from .metric import Metric


class AccuracyMetric(Metric):
    name = "acc"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true)
        y_pred = torch.as_tensor(y_pred)

        return (y_pred == y_true).float().mean().item()


class BalancedAccuracyMetric(Metric):
    name = "bacc"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        recalls = []

        for label in torch.unique(y_true):
            mask = y_true == label
            recall = (y_pred[mask] == label).float().mean()
            recalls.append(recall)

        return torch.stack(recalls).mean().item()


class PrecisionMetric(Metric):
    name = "prec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        predicted_positive = y_pred == 1

        if predicted_positive.sum() == 0:
            return 0.0

        true_positive = ((y_pred == 1) & (y_true == 1)).sum().float()
        precision = true_positive / predicted_positive.sum()

        return precision.item()


class RecallMetric(Metric):
    name = "rec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        positive = y_true == 1

        if positive.sum() == 0:
            return 0.0

        return (y_pred[positive] == 1).float().mean().item()


class F1ScoreMetric(Metric):
    name = "f1"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        true_positive = ((y_pred == 1) & (y_true == 1)).sum().float()
        false_positive = ((y_pred == 1) & (y_true == 0)).sum().float()
        false_negative = ((y_pred == 0) & (y_true == 1)).sum().float()

        denominator = 2 * true_positive + false_positive + false_negative

        if denominator == 0:
            return 0.0

        return (2 * true_positive / denominator).item()

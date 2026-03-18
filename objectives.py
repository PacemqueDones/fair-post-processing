import torch
import torch.nn.functional as F

class CrossEntropyObjective:
    name = "cross_entropy"

    def __call__(self, scores, y_true, sensitive_attr):
        return F.cross_entropy(scores, y_true)

class DemographicParityObjective:
    name = "demographic_parity"

    def __call__(self, scores, y_true, sensitive_attr):
        preds_pos = scores[:, 1]   # supondo binário por enquanto
        group0 = preds_pos[sensitive_attr == 0]
        group1 = preds_pos[sensitive_attr == 1]
        return torch.abs(group0.mean() - group1.mean())
import torch
from .objective import Objective
import torch.nn.functional as F

class CrossEntropyObjective(Objective):
    name = "cross_entropy"

    def __call__(self, logits, y_true, sensitive_attr):
        loss = F.cross_entropy(logits, y_true,)
        return [loss], [self.name]
import torch
import torch.nn as nn

class ThresholdModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0):
        super().__init__()
        self.thresholds = nn.Parameter(torch.randn(num_classes))
        self.alpha = alpha

    def forward(self, probs):
        margins = probs - self.thresholds
        smooth_scores = torch.softmax(self.alpha * margins, dim=1)
        return smooth_scores
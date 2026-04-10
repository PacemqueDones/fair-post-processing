import torch
import torch.nn as nn

class ThresholdMarginModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha

    def forward(self, probs):
        margins = probs - self.thresholds
        return self.alpha * margins

class ThresholdNormalizedMarginModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs):
        margins = (probs - self.thresholds) / (self.thresholds + self.eps)
        return self.alpha * margins
    
class ThresholdRatioModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs):
        ratios = probs / (self.thresholds + self.eps)
        return self.alpha * ratios

class ThresholdLogRatioModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs):
        log_ratios = torch.log(probs + self.eps) - torch.log(self.thresholds + self.eps)
        return self.alpha * log_ratios
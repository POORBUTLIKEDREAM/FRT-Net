import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.
    """

    def __init__(self, classes, smoothing=0.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        """
        pred: (batch_size, num_classes) - model logits
        target: (batch_size) - class indices
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # Create target distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
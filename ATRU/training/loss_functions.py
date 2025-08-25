# fault_diagnosis_project/training/loss_functions.py
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    实现标签平滑损失函数。
    这有助于防止模型对其预测过于自信，从而提高泛化能力。
    """

    def __init__(self, classes, smoothing=0.1):
        """
        初始化损失函数。
        Args:
            classes (int): 类别的总数。
            smoothing (float): 平滑因子，介于0和1之间。
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        """
        计算损失。
        Args:
            pred (torch.Tensor): 模型的原始输出 (logits)，形状为 (batch_size, num_classes)。
            target (torch.Tensor): 真实的类别标签，形状为 (batch_size)。
        """
        # 对模型的输出计算log_softmax
        pred = pred.log_softmax(dim=-1)

        # 创建平滑后的真实标签分布
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            # 将所有类别的概率设置为一个小的平滑值
            true_dist.fill_(self.smoothing / (self.classes - 1))
            # 将真实类别的概率设置为较高的置信度值
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 计算KL散度损失的均值
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
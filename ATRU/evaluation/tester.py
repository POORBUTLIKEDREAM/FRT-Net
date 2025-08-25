# fault_diagnosis_project/evaluation/tester.py
import torch
from tqdm import tqdm


class Tester:
    """
    负责在给定数据加载器上评估模型的类。
    """

    def __init__(self, model, data_loader, device):
        """
        初始化Tester。

        Args:
            model (torch.nn.Module): 要评估的已训练模型。
            data_loader (torch.utils.data.DataLoader): 用于评估的数据加载器。
            device (torch.device): 运行评估的设备 (例如, 'cuda' 或 'cpu')。
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def run(self, mode="Test"):
        """
        执行评估循环。

        Args:
            mode (str): 用于在进度条中显示的模式描述 (例如, "Test" 或 "Validate")。

        Returns:
            tuple: 包含以下元素的元组:
                - all_labels (list): 所有样本的真实标签列表。
                - all_preds (list): 模型对所有样本的预测标签列表。
                - all_probs (list): 模型对所有样本的输出概率列表。
                - accuracy (float): 在数据集上的整体准确率。
        """
        self.model.eval()  # 将模型设置为评估模式
        all_labels, all_preds, all_probs = [], [], []
        correct_predictions, total_samples = 0, 0

        # 禁用梯度计算以加速评估并减少内存使用
        with torch.no_grad():
            # 使用tqdm显示进度条
            for signals, labels in tqdm(self.data_loader, desc=f"{mode}ing"):
                # 将数据移动到指定设备
                signals = signals.to(self.device)

                # 模型前向传播
                outputs = self.model(signals)

                # 计算概率和预测类别
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # 收集结果
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

                # 计算准确率所需的统计数据
                total_samples += labels.size(0)
                correct_predictions += preds.cpu().eq(labels).sum().item()

        # 计算最终的整体准确率
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return all_labels, all_preds, all_probs, accuracy

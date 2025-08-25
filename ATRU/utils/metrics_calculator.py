# fault_diagnosis_project/utils/metrics.py
import numpy as np


def calculate_class_accuracy(cm):
    """
    根据混淆矩阵计算每个类别的准确率。

    Args:
        cm (numpy.ndarray): 混淆矩阵。

    Returns:
        dict: 包含每个类别准确率的字典。
    """
    class_accuracy = {}
    # 遍历混淆矩阵的每一行
    for i in range(len(cm)):
        # 对角线上的值是正确预测的数量 (真阳性)
        true_positives = cm[i, i]
        # 这一行的总和是该类别的样本总数
        total_samples_per_class = np.sum(cm[i, :])

        # 计算准确率，处理分母为0的情况
        accuracy = true_positives / total_samples_per_class if total_samples_per_class > 0 else 0.0
        class_accuracy[f'class_{i}'] = accuracy

    return class_accuracy

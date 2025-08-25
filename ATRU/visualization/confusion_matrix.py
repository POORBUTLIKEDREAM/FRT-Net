# fault_diagnosis_project/visualization/confusion_matrix.py
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_and_save_cm(cm, config, title='混淆矩阵 (Confusion Matrix)'):
    """
    生成并保存混淆矩阵图。

    Args:
        cm (numpy.ndarray): 混淆矩阵数组，由scikit-learn计算得出。
        config (EasyDict): 包含所有路径和参数的配置对象。
        title (str): 图表的标题。
    """
    # 从配置中构建保存路径
    save_path = os.path.join(config.paths.figure_dir, f'cm_{config.data.noise_level}.png')

    # 从配置中获取类别名称列表
    class_names = [str(i) for i in range(config.data.num_classes)]

    # 创建一个新的图形
    plt.figure(figsize=(12, 10))

    # 使用seaborn的热力图功能来绘制混淆矩阵
    sns.heatmap(
        cm,
        annot=True,  # 在每个单元格上显示数值
        fmt='d',  # 将数值格式化为整数
        cmap=plt.cm.Blues,  # 使用蓝色系配色方案
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8}  # 设置注释文字的大小
    )

    # 设置图表的标题和坐标轴标签
    plt.title(title)
    plt.ylabel('真实标签 (True Label)')
    plt.xlabel('预测标签 (Predicted Label)')

    # 自动调整布局以防止标签重叠
    plt.tight_layout()

    # 保存图表到指定路径
    try:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix plot: {e}")

    # 关闭图形以释放内存
    plt.close()

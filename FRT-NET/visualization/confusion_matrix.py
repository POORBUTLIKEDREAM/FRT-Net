import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵已保存为 {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_curves(history, save_path=None):
    """绘制并保存训练曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Acc', color='tab:blue')
    plt.plot(history['test_acc'], label='Test Acc', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"训练曲线已保存为 {save_path}")
    else:
        plt.show()
    plt.close()
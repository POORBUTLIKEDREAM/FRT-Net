import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """绘制混淆矩阵"""
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
    plt.close()


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失曲线
    ax1.plot(history['loss'], label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 绘制准确率曲线
    ax2.plot(history['acc'], label='Training Accuracy')
    ax2.plot(history['test_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_class_accuracy(class_acc, save_path=None):
    """绘制每个类别的准确率"""
    classes = list(class_acc.keys())
    accuracies = list(class_acc.values())

    plt.figure(figsize=(12, 6))
    plt.bar(classes, accuracies)
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(classes)
    plt.ylim(0, 1)

    # 在柱状图上添加数值标签
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    if save_path:
        plt.savefig(save_path)
    plt.close()
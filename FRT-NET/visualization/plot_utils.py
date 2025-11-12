import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, classes, save_path):
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history, save_path):
    """
    Plots and saves training loss and accuracy curves.
    """
    plt.figure(figsize=(12, 5))

    # --- Loss Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # --- Accuracy Curve ---
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Acc', color='tab:blue')
    # Corrected key from 'test_trans' to 'test_acc'
    plt.plot(history['test_acc'], label='Test Acc', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")
import numpy as np

def calculate_class_accuracy(cm):
    """
    Calculates per-class accuracy from a confusion matrix.
    """
    class_accuracy = {}
    for i in range(len(cm)):
        true_positives = cm[i, i]
        total_samples = np.sum(cm[i, :])
        accuracy = true_positives / total_samples if total_samples > 0 else 0.0
        class_accuracy[i] = accuracy
    return class_accuracy
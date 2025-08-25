# fault_diagnosis_project/evaluation/metrics_calculator.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
# --- CORRECTED IMPORT STATEMENT BELOW ---
from utils.metrics_calculator import calculate_class_accuracy


def calculate_all_metrics(all_labels, all_preds, all_probs):
    """
    Calculates and returns a dictionary of all relevant metrics.

    Args:
        all_labels (list): A list of all true labels.
        all_preds (list): A list of all predicted labels from the model.
        all_probs (list): A list of output probabilities from the model.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    metrics = {}
    metrics['overall_accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        # Concatenate list of probability arrays into a single array
        all_probs_array = np.concatenate(all_probs)
        metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs_array, multi_class="ovr", average="macro")
    except ValueError as e:
        # Handle cases where not all classes are present in a batch
        metrics['roc_auc_macro'] = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm
    metrics['class_accuracy'] = calculate_class_accuracy(cm)
    metrics['classification_report'] = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return metrics

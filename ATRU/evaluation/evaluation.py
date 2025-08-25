# fault_diagnosis_project/evaluation/metrics_calculator.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from ..utils.metrics_calculator import calculate_class_accuracy


def calculate_all_metrics(all_labels, all_preds, all_probs):
    """
    Calculates and returns a dictionary of all relevant metrics.
    """
    metrics = {}
    metrics['overall_accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        # Concatenate list of probability arrays into a single array
        all_probs_array = np.concatenate(all_probs)
        metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs_array, multi_class="ovr", average="macro")
    except ValueError as e:
        metrics['roc_auc_macro'] = 0.0  # Handle case where not all classes are present in a batch

    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm
    metrics['class_accuracy'] = calculate_class_accuracy(cm)
    metrics['classification_report'] = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return metrics


# fault_diagnosis_project/evaluation/tester.py
import torch
import logging
from tqdm import tqdm


class Tester:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def run(self, mode="Test"):
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        correct, total_samples = 0, 0

        with torch.no_grad():
            for signals, labels in tqdm(self.data_loader, desc=f"{mode}ing"):
                signals = signals.to(self.device)
                outputs = self.model(signals)

                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

                total_samples += labels.size(0)
                correct += preds.cpu().eq(labels).sum().item()

        accuracy = correct / total_samples
        return all_labels, all_preds, all_probs, accuracy

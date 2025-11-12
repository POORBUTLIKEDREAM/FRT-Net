import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Import helper metric function
from utils.metrics import calculate_class_accuracy


class MetricsCalculator:
    """
    Calculates all evaluation metrics from test loop results.
    """

    def __init__(self, all_labels, all_preds, all_probs, config):
        self.labels = all_labels
        self.preds = all_preds
        self.probs = all_probs
        self.config = config

    def compute_all_metrics(self):
        """
        Compute, print, and save all metrics.
        """
        # 1. Standard metrics
        accuracy = accuracy_score(self.labels, self.preds)
        f1 = f1_score(self.labels, self.preds, average="macro")

        # 2. ROC AUC Score
        try:
            roc_auc = roc_auc_score(self.labels, self.probs, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"Could not compute ROC AUC: {e}. Defaulting to 0.0")
            roc_auc = 0.0

        # 3. Confusion Matrix
        cm = confusion_matrix(self.labels, self.preds)

        # 4. Per-class Accuracy
        class_acc = calculate_class_accuracy(cm)
        print("\nPer-Class Accuracy:")
        for class_idx, acc in class_acc.items():
            print(f"Class {class_idx}: {acc:.4f}")

        # 5. Classification Report
        class_report = classification_report(self.labels, self.preds, output_dict=True)
        report_df = pd.DataFrame(class_report).transpose()
        report_path = os.path.join(self.config.output_dir, self.config.report_name)
        report_df.to_csv(report_path, index=True)
        print(f"Classification report saved to {report_path}")

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1,
            "roc_auc_macro": roc_auc,
            "confusion_matrix": cm,
            "class_accuracy": class_acc
        }
        return metrics
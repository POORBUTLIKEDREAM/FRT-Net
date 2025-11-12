import torch
import numpy as np


class Tester:
    """
    Handles running the model on the test set to gather predictions.
    """

    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def run_test_loop(self):
        """
        Run the test loop and gather all predictions and labels.
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for signals, labels in self.test_loader:
                signals = signals.to(self.device)

                outputs = self.model(signals)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs = np.concatenate(all_probs)
        return all_labels, all_preds, all_probs
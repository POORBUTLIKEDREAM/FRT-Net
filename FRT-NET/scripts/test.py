import os
import sys
import torch
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.logger import setup_logger
from data.data_loader import create_loaders
from models.FRT_Net import FRT_Net
from evaluation.tester import Tester
from evaluation.metrics_calculator import MetricsCalculator
from visualization.plot_utils import plot_confusion_matrix


def test_only():
    # 1. Load Configuration
    config_path = 'config/config.yaml'
    config = load_config(config_path)

    # 2. === Resolve Relative Paths ===
    if not os.path.isabs(config.data_root):
        config.data_root = os.path.join(project_root, config.data_root)

    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.join(project_root, config.output_dir)
    # ================================

    # 3. Setup Logger
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(config.output_dir, 'test.log'))
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Resolved Data Root: {config.data_root}")
    logger.info(f"Resolved Output Dir: {config.output_dir}")
    logger.info(f"Using device: {config.device}")

    # 4. Load Test Data
    logger.info("Loading test data...")
    if not os.path.exists(config.data_root):
        logger.error(f"Data root not found: {config.data_root}")
        logger.error("Please check the 'data_root' path in config/config.yaml.")
        return

    _, test_loader = create_loaders(config)
    if len(test_loader) == 0:
        logger.error("Test loader is empty. Please check data files in the data_root.")
        return
    logger.info(f"Test data loaded: {len(test_loader)} test batches.")

    # 5. Initialize Model
    model = FRT_Net(config)
    model.to(config.device)

    # 6. Load Pre-trained Weights
    model_path = os.path.join(config.output_dir, config.best_model_name)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Please run train.py first.")
        return

    logger.info(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    # 7. Run Evaluation
    tester = Tester(model, test_loader, config.device)
    all_labels, all_preds, all_probs = tester.run_test_loop()

    calculator = MetricsCalculator(all_labels, all_preds, all_probs, config)
    metrics = calculator.compute_all_metrics()

    # 8. Print and Save Results
    logger.info("\n--- Final Evaluation Results ---")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    cm_path = os.path.join(config.output_dir, f"test_{config.confusion_matrix_name}")
    plot_confusion_matrix(metrics['confusion_matrix'],
                          classes=[str(i) for i in range(config.num_classes)],
                          save_path=cm_path)
    logger.info("Test-only script finished.")


if __name__ == "__main__":
    test_only()
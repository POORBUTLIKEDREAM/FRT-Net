import os
import sys
import torch
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ===================================

from utils.config_loader import load_config
from utils.logger import setup_logger
from data.data_loader import create_loaders
from models.FRT_Net import FRT_Net
from training.trainer import Trainer
from evaluation.tester import Tester
from evaluation.metrics_calculator import MetricsCalculator
from visualization.plot_utils import plot_training_curves, plot_confusion_matrix


def main():
    config_path = os.path.join(project_root, 'config/config.yaml')
    config = load_config(config_path)

    # Resolve Relative Paths
    if not os.path.isabs(config.data_root):
        config.data_root = os.path.join(project_root, config.data_root)

    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.join(project_root, config.output_dir)

    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(config.output_dir, 'train.log'))
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Resolved Data Root: {config.data_root}")
    logger.info(f"Resolved Output Dir: {config.output_dir}")
    logger.info(f"Using device: {config.device}")

    logger.info("Loading data...")
    if not os.path.exists(config.data_root):
        logger.error(f"Data root not found: {config.data_root}")
        logger.error("Please check the 'data_root' path in config/config.yaml.")
        return

    train_loader, test_loader = create_loaders(config)
    if len(train_loader) == 0 or len(test_loader) == 0:
        logger.error("Data loaders are empty. Please check data files in the data_root.")
        return
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches.")

    model = FRT_Net(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized. Trainable parameters: {total_params / 1e6:.2f}M")
    model.to(config.device)

    trainer = Trainer(model, train_loader, test_loader, config)
    train_history, best_acc = trainer.train()
    logger.info(f"Training complete. Best validation accuracy: {best_acc:.5f}")

    curves_path = os.path.join(config.output_dir, config.training_curves_name)
    plot_training_curves(train_history, curves_path)

    logger.info("Loading best model for final evaluation...")
    best_model_path = os.path.join(config.output_dir, config.best_model_name)
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found at {best_model_path}. Evaluation skipped.")
        return

    model.load_state_dict(torch.load(best_model_path))

    tester = Tester(model, test_loader, config.device)
    all_labels, all_preds, all_probs = tester.run_test_loop()

    calculator = MetricsCalculator(all_labels, all_preds, all_probs, config)
    metrics = calculator.compute_all_metrics()

    logger.info("\n--- Final Evaluation Results ---")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    cm_path = os.path.join(config.output_dir, config.confusion_matrix_name)
    plot_confusion_matrix(metrics['confusion_matrix'],
                          classes=[str(i) for i in range(config.num_classes)],
                          save_path=cm_path)
    logger.info("Main training script finished.")


if __name__ == "__main__":
    main()
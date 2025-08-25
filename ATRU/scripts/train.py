# fault_diagnosis_project/scripts/train.py
import logging
import os
import torch
from tqdm import trange
import sys

# This is the key part of the fix. It ensures the project's root directory is
# on the Python path, so it can find all the other modules.
# It also helps us build a reliable path to the config file.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.logger import setup_logger
from data.data_loader import create_loaders
from models.FRT_Net import FRTNet
from training.trainer import Trainer
from evaluation.tester import Tester


def train():
    """
    Runs the complete model training pipeline.
    """
    # --- FIX STARTS HERE ---
    # Build an absolute path to the config file to avoid working directory issues.
    # 1. Get the directory of the current script (e.g., '.../scripts/')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Get the project root directory (one level up from 'scripts/')
    project_root = os.path.dirname(script_dir)
    # 3. Construct the full path to the config file
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    # 4. Load the config using the absolute path
    config = load_config(config_path)
    # --- FIX ENDS HERE ---

    # Setup logging
    log_file = f"train_{config.data.noise_level}.log"
    setup_logger(config.paths.log_dir, log_file)

    # Automatically select device
    if config.system.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.system.device)
    logging.info(f"Using device: {device}")

    # Create data loaders, model, trainer, and validator
    train_loader, test_loader = create_loaders(config)
    model = FRTNet(config).to(device)
    trainer = Trainer(model, train_loader, config, device)
    validator = Tester(model, test_loader, device)

    best_acc = 0.0
    model_save_path = os.path.join(config.paths.model_dir, f"best_model_{config.data.noise_level}.pth")

    # Start training loop
    epochs_pbar = trange(config.training.epochs, desc="Training Epochs")
    for epoch in epochs_pbar:
        avg_loss, train_acc = trainer.train_epoch()
        # Run evaluation on the validation set to get accuracy
        _, _, _, val_acc = validator.run(mode="Validat")

        # Update learning rate
        trainer.scheduler.step(val_acc)
        lr = trainer.optimizer.param_groups[0]['lr']

        # Log progress
        log_message = (
            f"Epoch {epoch + 1}/{config.training.epochs} | LR: {lr:.2e} | "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )
        logging.info(log_message)

        # Save the model if it's the best one so far
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with accuracy: {best_acc:.4f}")

    logging.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    train()

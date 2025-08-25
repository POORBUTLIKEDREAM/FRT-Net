# fault_diagnosis_project/training/trainer.py
import logging
import torch
import torch.optim as optim
from .loss_functions import LabelSmoothingLoss


class Trainer:
    """
    Encapsulates the main training logic for the model.
    """

    def __init__(self, model, train_loader, config, device):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for training.
            config (EasyDict): The configuration object with all hyperparameters.
            device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        """
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device

        # Initialize the loss function
        self.criterion = LabelSmoothingLoss(
            classes=config.data.num_classes,
            smoothing=config.training.label_smoothing
        ).to(self.device)

        # Initialize the optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Reduce LR when the monitored metric has stopped improving
            factor=config.scheduler.factor,
            patience=config.scheduler.patience
        )

    def train_epoch(self):
        """
        Executes one full training epoch.

        Returns:
            tuple: A tuple containing the average training loss and training accuracy.
        """
        self.model.train()  # Set the model to training mode
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        # Iterate over all batches in the training data loader
        for signals, labels in self.train_loader:
            # Move data to the specified device
            signals, labels = signals.to(self.device), labels.to(self.device)

            # 1. Clear previous gradients
            self.optimizer.zero_grad()

            # 2. Forward pass: compute predicted outputs by passing inputs to the model
            outputs = self.model(signals)

            # 3. Calculate the loss
            loss = self.criterion(outputs, labels)

            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # (Optional) Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_norm)

            # 5. Perform a single optimization step (parameter update)
            self.optimizer.step()

            # Accumulate loss and calculate accuracy stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

        # Calculate average loss and total accuracy for the epoch
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

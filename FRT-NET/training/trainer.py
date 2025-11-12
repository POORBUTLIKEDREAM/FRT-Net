import torch
import torch.optim as optim
import os

from training.loss_functions import LabelSmoothingLoss


class Trainer:
    """
    Trainer class to handle the training and validation loop.
    """

    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device

        # Setup optimizer, scheduler, and criterion
        self.criterion = LabelSmoothingLoss(
            classes=config.num_classes,
            smoothing=config.label_smoothing
        ).to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        self.best_acc = 0.0
        self.train_history = {'loss': [], 'acc': [], 'test_acc': []}
        self.model.to(self.device)

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        train_loss, correct, total = 0.0, 0, 0

        for signals, labels in self.train_loader:
            signals, labels = signals.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Run one validation epoch."""
        self.model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for signals, labels in self.test_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        epoch_acc = test_correct / test_total
        return epoch_acc

    def train(self):
        """Run the full training process."""
        print("Starting training...")
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch()
            test_acc = self.validate_epoch()

            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['test_acc'].append(test_acc)

            self.scheduler.step(test_acc)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch:{epoch + 1:03d}, LR:{lr:.2e}, "
                  f"train_loss:{train_loss:.5f}, train_acc:{train_acc:.5f}, test_acc:{test_acc:.5f}")

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                save_path = os.path.join(self.config.output_dir, self.config.best_model_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best model with accuracy: {self.best_acc:.5f}")

        print("Training finished.")
        return self.train_history, self.best_acc
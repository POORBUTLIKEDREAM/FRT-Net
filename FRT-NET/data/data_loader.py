import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

# Import preprocessing functions
from data.preprocessing import normalize, augment_data


class SignalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading signal data.
    """

    def __init__(self, data_root, noise, mode, signal_length):
        self.mode = mode
        self.signal_length = signal_length
        self.data_dir = os.path.join(data_root, noise, f"{mode}_data")
        self.label_dir = os.path.join(data_root, noise, f"{mode}_lab")

        self.data_files = sorted(glob(os.path.join(self.data_dir, "*.npy")))
        self.label_files = sorted(glob(os.path.join(self.label_dir, "*.npy")))

        if not self.data_files or not self.label_files:
            print(f"Warning: No files found in {self.data_dir} or {self.label_dir}.")
            print(f"Please check 'data_root' ('{data_root}') and 'noise_condition' ('{noise}').")

        if len(self.data_files) != len(self.label_files):
            raise ValueError(f"{mode} data and label counts mismatch: "
                             f"{len(self.data_files)} vs {len(self.label_files)}")

    def __getitem__(self, idx):
        try:
            signal = np.load(self.data_files[idx]).astype(np.float32)

            # Augment/Pad/Crop to fixed signal_length
            # augment_data expects a list
            signal_processed = augment_data([signal], self.signal_length)[0]

            # Normalize the signal
            signal_normalized = normalize(signal_processed).T

            label = np.load(self.label_files[idx]).astype(np.int64)

            return torch.FloatTensor(signal_normalized), torch.LongTensor(label).squeeze()

        except Exception as e:
            print(f"Error loading file: {self.data_files[idx]} or {self.label_files[idx]}")
            print(f"Error: {e}")
            # Return placeholders
            return torch.zeros(self.signal_length, dtype=torch.float), \
                torch.zeros(1, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.data_files)


def create_loaders(config):
    """
    Create training and testing DataLoader instances.
    """
    train_set = SignalDataset(
        data_root=config.data_root,
        noise=config.noise_condition,
        mode="train",
        signal_length=config.signal_length
    )
    test_set = SignalDataset(
        data_root=config.data_root,
        noise=config.noise_condition,
        mode="test",
        signal_length=config.signal_length
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, test_loader
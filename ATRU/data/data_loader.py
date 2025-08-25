# fault_diagnosis_project/data/data_loader.py
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from .preprocessing import normalize, augment_and_segment


class SignalDataset(Dataset):
    """
    用于加载和处理信号数据的自定义PyTorch数据集。
    """

    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.is_train = (mode == "train")

        # 从config对象中获取路径和参数
        self.data_dir = os.path.join(config.paths.data_root, config.data.noise_level, f"{mode}_data")
        self.label_dir = os.path.join(config.paths.data_root, config.data.noise_level, f"{mode}_lab")

        self.data_files = sorted(glob(os.path.join(self.data_dir, "*.npy")))
        self.label_files = sorted(glob(os.path.join(self.label_dir, "*.npy")))

        if not self.data_files:
            raise FileNotFoundError(f"No data files found for mode '{mode}' in '{self.data_dir}'.")
        if len(self.data_files) != len(self.label_files):
            raise ValueError("Mismatch between number of data and label files.")

        logging.info(f"Initialized {mode} dataset with {len(self.data_files)} samples.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        signal = np.load(self.data_files[idx]).astype(np.float32).flatten()
        label = np.load(self.label_files[idx]).astype(np.int64)

        # 从config对象中获取信号长度
        signal = augment_and_segment(signal, self.config.data.signal_length, self.is_train)
        signal = normalize(signal)

        signal_tensor = torch.from_numpy(signal).unsqueeze(0)  # 添加通道维度
        label_tensor = torch.from_numpy(label).squeeze()

        return signal_tensor, label_tensor


def create_loaders(config):
    """
    为训练和测试创建数据加载器。
    这个函数只接收一个config对象作为参数。
    """
    train_set = SignalDataset(config, "train")
    test_set = SignalDataset(config, "test")

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,  # 从config中获取batch_size
        shuffle=True,
        num_workers=config.data.num_workers,  # 从config中获取num_workers
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.testing.batch_size,  # 从config中获取test_batch_size
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, test_loader

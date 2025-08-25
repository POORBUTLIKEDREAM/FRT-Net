# fault_diagnosis_project/data/preprocessing.py
import numpy as np


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Standardizes the data to have zero mean and unit variance.
    """
    return (data - np.mean(data)) / (np.std(data) + 1e-8)


def augment_and_segment(data: np.ndarray, signal_length: int, is_train: bool) -> np.ndarray:
    """
    Adjusts the signal to a fixed length by padding or cropping.
    """
    current_length = len(data)

    if current_length < signal_length:
        pad_width = signal_length - current_length
        return np.pad(data, (0, pad_width), mode='constant', constant_values=0)

    elif current_length > signal_length:
        if is_train:
            start_idx = np.random.randint(0, current_length - signal_length + 1)
        else:
            start_idx = (current_length - signal_length) // 2
        return data[start_idx: start_idx + signal_length]

    return data

import numpy as np


def normalize(data):
    """
    Apply Z-score normalization to the data.
    Adds a small epsilon to the std to prevent division by zero.
    """
    std = np.std(data)
    return (data - np.mean(data)) / (std + 1e-8)


def augment_data(data_list, signal_length):
    """
    Augments a list of signals to a fixed signal_length.
    - Pads signals shorter than signal_length.
    - Randomly crops signals longer than signal_length.
    """
    aug_data = []
    for d in data_list:
        if len(d) < signal_length:
            # Pad with zeros
            pad_width = signal_length - len(d)
            padded = np.pad(d, (0, pad_width), mode='constant')
            aug_data.append(padded)
        elif len(d) == signal_length:
            # No change needed
            aug_data.append(d)
        else:
            # Random crop
            max_start_idx = len(d) - signal_length
            # +1 because randint upper bound is exclusive
            start_idx = np.random.randint(0, max_start_idx + 1)
            aug_data.append(d[start_idx:start_idx + signal_length])

    return np.array(aug_data)
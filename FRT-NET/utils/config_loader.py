import yaml
from argparse import Namespace
import torch


def load_config(config_path):
    """
    Loads a YAML config file and returns it as a Namespace object
    for easy dot-notation access (e.g., config.data_root).
    """

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Convert dictionary to Namespace
    config = Namespace(**config_dict)

    # Set device
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and config.device.type == 'cuda':
        print(f"Warning: CUDA not available (specified '{config.device}'). Defaulting to CPU.")
        config.device = torch.device("cpu")

    return config
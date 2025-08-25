# fault_diagnosis_project/utils/config_loader.py
import yaml
from easydict import EasyDict
import os


def load_config(config_path='config/config.yaml'):
    """
    以UTF-8编码加载YAML配置文件，并返回一个EasyDict对象。
    同时创建必要的输出目录。
    """
    # 使用 with open(...) as file: 确保文件会被正确关闭
    # 添加 encoding='utf-8' 来解决编码错误
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 确保所有在配置文件中定义的输出目录都存在
    # 如果目录不存在，就创建它
    for key, path in config['paths'].items():
        if 'dir' in key:
            os.makedirs(path, exist_ok=True)

    return EasyDict(config)

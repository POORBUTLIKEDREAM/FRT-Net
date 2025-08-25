# fault_diagnosis_project/utils/logger.py
import logging
import os


def setup_logger(log_dir, log_file_name):
    """
    配置并设置一个日志记录器。

    Args:
        log_dir (str): 日志文件存放的目录路径。
        log_file_name (str): 日志文件的名称。
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 将目录和文件名组合成完整的日志文件路径
    log_path = os.path.join(log_dir, log_file_name)

    # 为防止重复记录，先移除所有现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            # 将日志写入文件，使用UTF-8编码
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            # 同时将日志输出到控制台
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logger initialized. Log file at: {log_path}")
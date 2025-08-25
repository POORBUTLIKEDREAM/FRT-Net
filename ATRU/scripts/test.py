# fault_diagnosis_project/scripts/test.py
import logging
import os
import torch
import pandas as pd
import sys

from utils.config_loader import load_config
from utils.logger import setup_logger
from data.data_loader import create_loaders
from models.FRT_Net import FRTNet
from evaluation.tester import Tester
# --- 修正 #1: 从正确的文件导入正确的函数 ---
from evaluation.metrics_calculator import calculate_all_metrics
from visualization.confusion_matrix import plot_and_save_cm


def test():
    """
    运行完整的模型评估流程。
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    config = load_config(config_path)

    # 设置日志
    log_file = f"test_{config.data.noise_level}.log"
    setup_logger(config.paths.log_dir, log_file)

    # 自动选择设备
    if config.system.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.system.device)
    logging.info(f"使用设备: {device}")

    # 加载数据和模型
    _, test_loader = create_loaders(config)
    model = FRTNet(config).to(device)
    model_path = os.path.join(config.paths.model_dir, f"best_model_{config.data.noise_level}.pth")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"模型已从 {model_path} 加载")
    except FileNotFoundError:
        logging.error(f"在 {model_path} 未找到模型文件。请先运行训练脚本。")
        return

    # 评估模型
    tester = Tester(model, test_loader, device)
    labels, preds, probs, _ = tester.run(mode="Test")

    metrics = calculate_all_metrics(labels, preds, probs)

    # 记录并保存结果
    logging.info("--- 测试结果 ---")
    logging.info(f"整体准确率: {metrics['overall_accuracy']:.4f}")
    logging.info(f"宏 F1 分数: {metrics['f1_macro']:.4f}")
    logging.info(f"宏 ROC AUC: {metrics['roc_auc_macro']:.4f}")

    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_path = os.path.join(config.paths.results_dir, f'report_{config.data.noise_level}.csv')
    report_df.to_csv(report_path)
    logging.info(f"分类报告已保存至: {report_path}")

    plot_and_save_cm(metrics['confusion_matrix'], config)
    logging.info("--- 评估完成 ---")


if __name__ == '__main__':
    test()

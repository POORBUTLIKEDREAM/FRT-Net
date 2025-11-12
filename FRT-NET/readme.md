# FRT-Net: Fault Diagnosis Project

This project implements an "FRT-Net" (Frequency-domain Transformer Network) for fault diagnosis, based on the provided code. The model uses a CNN backbone on frequency-domain (FFT) data, followed by an efficient Linear Transformer for sequence modeling.

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

fault_diagnosis_project/
│
├── config/
│   └── config.yaml             # 所有参数的YAML配置文件
│
├── data/
│   ├── data_loader.py          # 数据集类和数据加载器
│   └── preprocessing.py        # 数据预处理函数
│
├── models/
│   ├── FRT_Net.py              # FRT-Net主模型架构
│   └── components/             # 模型的各个组件模块
│       ├── positional_encoding.py
│       ├── dsrb.py
│       ├── dsra.py
│       └── efficient_transformer.py
│
├── training/
│   ├── trainer.py              # 训练器类
│   └── loss_functions.py       # 自定义损失函数
│
├── evaluation/
│   ├── tester.py               # 测试器类
│   └── metrics_calculator.py   # 评估指标计算
│
├── visualization/
│   └── plot_utils.py           # 绘制混淆矩阵和训练曲线
│
├── utils/
│   ├── config_loader.py        # 配置加载器
│   ├── logger.py               # 日志记录器
│   └── metrics.py              # 辅助指标计算 (保留)
│
├── scripts/
│   ├── train.py                # 执行训练的脚本
│   └── test.py                 # 执行测试的脚本
│
├── outputs/                    # 默认输出目录 (自动创建)
│
├── requirements.txt            # 项目依赖库
└── README.md                   # 本文档

## How to Run

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt

2. Configure Paths
Open config/config.yaml and edit the data_root variable to point to the correct location

3. Run Training
To run the full training and evaluation process, execute the train.py script from the project's root directory:

Bash

python scripts/train.py
This will:

Load the data.

Build the model.

Train the model, saving the best version to outputs/best_model.pth.

Save training curves to outputs/training_curves.png.

Load the best model and run a final evaluation.

Save the classification report and confusion matrix to the outputs/ directory.

4. Run Testing Only
If you have already trained a model and just want to re-run the evaluation, execute the test.py script:

Bash

python scripts/test.py
This will load the model from outputs/best_model.pth and run the evaluation on the test set.
# 图片识别项目

这是一个用于训练和评估图片识别模型的项目。该项目使用PyTorch框架实现，支持多种预训练模型，可用于各种图像分类任务。

## 项目结构

```
├── config/             # 配置文件目录
│   └── config.py      # 项目配置参数
├── data/               # 数据目录
│   ├── train/         # 训练数据集
│   ├── val/           # 验证数据集
│   └── test/          # 测试数据集
├── models/             # 模型保存目录
├── results/            # 结果保存目录
├── src/                # 源代码目录
│   ├── data_loader.py # 数据加载模块
│   ├── evaluator.py   # 模型评估模块
│   ├── main.py        # 主程序入口
│   ├── model.py       # 模型定义模块
│   └── trainer.py     # 模型训练模块
├── utils/              # 工具函数目录
└── requirements.txt    # 项目依赖包
```

## 数据准备

请将数据集按以下结构组织：

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   └── ...
    ├── class2/
    │   └── ...
    └── ...
```

每个类别的文件夹名称将作为该类别的标签名称。

## 环境配置

1. 创建并激活虚拟环境（可选）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

2. 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python -m src.main --mode train
```

### 评估模型

```bash
python -m src.main --mode eval --model_path ./models/image_classifier_best_epoch10_acc0.9500.pth
```

### 预测单张图像

```bash
python -m src.main --mode predict --model_path ./models/image_classifier_best_epoch10_acc0.9500.pth --image_path ./data/test/class1/image1.jpg
```

## 配置说明

可以在`config/config.py`文件中修改以下配置：

- 数据配置：数据目录、批量大小、工作线程数等
- 模型配置：模型类型、类别数量、预训练权重等
- 训练配置：优化器、学习率、调度器、训练轮数等
- 评估配置：批量大小、结果保存目录等
- 设备配置：是否使用GPU、GPU设备ID等

## 支持的模型

- ResNet18
- ResNet50
- MobileNetV2
- EfficientNet-B0

## 示例结果

训练完成后，可以在`results`目录下查看训练历史曲线和混淆矩阵等可视化结果。
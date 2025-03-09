import os

# 数据配置
DATA_CONFIG = {
    'data_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
    'batch_size': 32,
    'num_workers': 4
}

# 模型配置
MODEL_CONFIG = {
    'model_name': 'resnet18',  # 可选: 'resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'
    'num_classes': 10,  # 根据实际数据集类别数量调整
    'pretrained': True,
    'weights_path': None  # 如果有预训练权重，可以在这里指定路径
}

# 训练配置
TRAIN_CONFIG = {
    'optimizer': 'adam',  # 可选: 'adam', 'sgd'
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,  # 仅在使用SGD优化器时有效
    'scheduler': 'step',  # 可选: 'step', 'cosine', 'plateau', None
    'step_size': 7,  # 仅在使用StepLR调度器时有效
    'gamma': 0.1,  # 仅在使用StepLR调度器时有效
    't_max': 10,  # 仅在使用CosineAnnealingLR调度器时有效
    'patience': 3,  # 仅在使用ReduceLROnPlateau调度器时有效
    'factor': 0.1,  # 仅在使用ReduceLROnPlateau调度器时有效
    'num_epochs': 20,
    'save_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
    'model_name': 'image_classifier'
}

# 评估配置
EVAL_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'results_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,  # 是否使用GPU
    'cuda_device': 0  # 使用的GPU设备ID
}

# 获取完整配置
def get_config():
    config = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'eval': EVAL_CONFIG,
        'device': DEVICE_CONFIG
    }
    return config
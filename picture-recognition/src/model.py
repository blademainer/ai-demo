import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageClassifier(nn.Module):
    """图像分类模型"""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        """
        初始化图像分类模型
        
        Args:
            num_classes (int): 类别数量
            model_name (str): 基础模型名称，支持 'resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'
            pretrained (bool): 是否使用预训练权重
        """
        super(ImageClassifier, self).__init__()
        
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.base_model(x)
        output = self.classifier(features)
        return output


def create_model(config):
    """创建模型实例"""
    model = ImageClassifier(
        num_classes=config['num_classes'],
        model_name=config.get('model_name', 'resnet18'),
        pretrained=config.get('pretrained', True)
    )
    
    # 如果指定了权重文件，加载预训练权重
    if 'weights_path' in config and config['weights_path']:
        model.load_state_dict(torch.load(config['weights_path']))
        print(f"加载权重文件: {config['weights_path']}")
    
    return model
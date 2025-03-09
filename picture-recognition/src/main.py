import os
import torch
import argparse
from config.config import get_config
from src.data_loader import get_data_loaders
from src.model import create_model
from src.trainer import create_trainer
from src.evaluator import Evaluator, Predictor

def train(config):
    """训练模型"""
    # 获取数据加载器
    dataloaders, datasets = get_data_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 获取类别信息
    if 'train' in datasets:
        classes = datasets['train'].classes
        num_classes = len(classes)
        print(f"数据集类别数: {num_classes}")
        print(f"类别: {classes}")
        
        # 更新配置中的类别数
        config['model']['num_classes'] = num_classes
    else:
        print("警告: 未找到训练数据集，无法获取类别信息")
    
    # 创建模型
    model = create_model(config['model'])
    print(f"创建模型: {config['model']['model_name']}")
    
    # 创建训练器
    trainer = create_trainer(model, dataloaders, config['train'])
    
    # 训练模型
    print("开始训练...")
    trainer.train(num_epochs=config['train']['num_epochs'])
    
    # 绘制训练历史
    trainer.plot_history()
    
    return model, classes

def evaluate(model, dataloader, classes=None, device=None):
    """评估模型"""
    # 创建评估器
    evaluator = Evaluator(model, dataloader, device=device, classes=classes)
    
    # 评估模型
    print("评估模型性能...")
    evaluator.plot_confusion_matrix()
    evaluator.print_classification_report()

def predict(model, image_path, classes=None, device=None):
    """预测单张图像"""
    # 创建预测器
    predictor = Predictor(model, device=device, classes=classes)
    
    # 预测图像
    print(f"预测图像: {image_path}")
    result = predictor.visualize_prediction(image_path)
    
    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像识别模型训练与评估')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'],
                        help='运行模式: train, eval, predict')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重文件路径 (用于评估和预测模式)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='待预测图像路径 (仅用于预测模式)')
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    if args.mode == 'train':
        # 训练模式
        model, classes = train(config)
        
        # 评估模型
        dataloaders, _ = get_data_loaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['eval']['batch_size'],
            num_workers=config['eval']['num_workers']
        )
        if 'test' in dataloaders:
            evaluate(model, dataloaders['test'], classes, device)
    
    elif args.mode == 'eval':
        # 评估模式
        if args.model_path is None:
            print("错误: 评估模式需要指定模型路径 (--model_path)")
            return
        
        # 加载数据
        dataloaders, datasets = get_data_loaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['eval']['batch_size'],
            num_workers=config['eval']['num_workers']
        )
        
        if 'test' not in dataloaders:
            print("错误: 未找到测试数据集")
            return
        
        # 获取类别信息
        classes = None
        if 'train' in datasets:
            classes = datasets['train'].classes
            config['model']['num_classes'] = len(classes)
        elif 'test' in datasets:
            classes = datasets['test'].classes
            config['model']['num_classes'] = len(classes)
        
        # 创建模型
        config['model']['weights_path'] = args.model_path
        model = create_model(config['model'])
        
        # 评估模型
        evaluate(model, dataloaders['test'], classes, device)
    
    elif args.mode == 'predict':
        # 预测模式
        if args.model_path is None:
            print("错误: 预测模式需要指定模型路径 (--model_path)")
            return
        
        if args.image_path is None:
            print("错误: 预测模式需要指定图像路径 (--image_path)")
            return
        
        # 加载数据以获取类别信息
        _, datasets = get_data_loaders(
            data_dir=config['data']['data_dir'],
            batch_size=1,
            num_workers=1
        )
        
        # 获取类别信息
        classes = None
        if 'train' in datasets:
            classes = datasets['train'].classes
            config['model']['num_classes'] = len(classes)
        
        # 创建模型
        config['model']['weights_path'] = args.model_path
        model = create_model(config['model'])
        
        # 预测图像
        predict(model, args.image_path, classes, device)

if __name__ == '__main__':
    main()
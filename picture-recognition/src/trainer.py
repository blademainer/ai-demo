import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler=None, device=None, config=None):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            dataloaders: 数据加载器字典，包含 'train' 和 'val' 两个键
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            device: 训练设备
            config: 训练配置
        """
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config else {}
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 保存路径
        self.save_dir = self.config.get('save_dir', './models')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self, num_epochs=10):
        """训练模型"""
        since = time.time()
        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                else:
                    self.model.eval()   # 设置模型为评估模式
                
                running_loss = 0.0
                running_corrects = 0
                
                # 迭代数据
                pbar = tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch+1}/{num_epochs}')
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 梯度清零
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # 反向传播 + 优化（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': loss.item()})
                
                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()
                
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 记录历史
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                
                # 如果是最好的模型，保存权重
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict().copy()
                    # 保存最佳模型
                    self._save_model(epoch, best_acc.item(), 'best')
            
            # 每个epoch结束后保存检查点
            self._save_model(epoch, epoch_acc.item(), 'last')
            print()
        
        time_elapsed = time.time() - since
        print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'最佳验证准确率: {best_acc:.4f}')
        
        # 加载最佳模型权重
        self.model.load_state_dict(best_model_wts)
        return self.model
    
    def _save_model(self, epoch, acc, suffix=''):
        """保存模型"""
        filename = f"{self.config.get('model_name', 'model')}_{suffix}_epoch{epoch}_acc{acc:.4f}.pth"
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")
    
    def plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='train')
        plt.plot(self.history['val_loss'], label='val')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='train')
        plt.plot(self.history['val_acc'], label='val')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/training_history.png')
        plt.show()


def create_trainer(model, dataloaders, config):
    """创建训练器"""
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 学习率调度器
    scheduler_name = config.get('scheduler', 'step').lower()
    
    if scheduler_name == 'step':
        step_size = config.get('step_size', 7)
        gamma = config.get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        t_max = config.get('t_max', 10)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name == 'plateau':
        patience = config.get('patience', 3)
        factor = config.get('factor', 0.1)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        scheduler = None
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    return trainer
"""
训练器模块
实现完整的训练循环、验证和检查点管理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
from typing import Optional, Dict, Any


class Trainer:
    """
    Transformer训练器
    
    参数:
        model: Transformer模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        config: 训练配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 默认配置
        default_config = {
            'num_epochs': 50,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'save_interval': 5000,
            'eval_interval': 1000,
            'log_interval': 100,
            'early_stopping_patience': 5,
            'fp16': False
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 创建目录
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # TensorBoard写入器
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.config['fp16'] else None
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch
        
        参数:
            epoch: 当前epoch索引
        
        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        # 进度条
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            # 移动到设备
            src = src.to(self.device)  # [batch_size, src_len]
            tgt = tgt.to(self.device)  # [batch_size, tgt_len]
            
            # 解码器输入: 去掉最后一个token (<eos>)
            tgt_input = tgt[:, :-1]
            
            # 解码器目标: 去掉第一个token (<bos>)
            tgt_output = tgt[:, 1:]
            
            # 前向传播
            if self.config['fp16']:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    output = self.model(src, tgt_input)  # [batch_size, tgt_len-1, vocab_size]
                    loss = self.criterion(output, tgt_output)
                
                # 梯度缩放和反向传播
                loss = loss / self.config['gradient_accumulation_steps']
                self.scaler.scale(loss).backward()
            else:
                # 标准训练
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
                
                # 归一化损失 (用于梯度累积)
                loss = loss / self.config['gradient_accumulation_steps']
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # 梯度裁剪
                if self.config['fp16']:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                
                # 优化器步骤
                if self.config['fp16']:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 更新全局步数
                self.global_step += 1
                
                # 记录日志
                if self.global_step % self.config['log_interval'] == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('train/loss', loss.item() * self.config['gradient_accumulation_steps'], self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
                
                # 验证
                if self.global_step % self.config['eval_interval'] == 0:
                    val_loss = self.validate()
                    self.writer.add_scalar('val/loss', val_loss, self.global_step)
                    
                    # 早停检查
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.save_checkpoint('best_model.pt', is_best=True)
                    else:
                        self.epochs_without_improvement += 1
                    
                    self.model.train()
                
                # 保存检查点
                if self.global_step % self.config['save_interval'] == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            # 累积损失
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() * self.config['gradient_accumulation_steps'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        验证模型
        
        返回:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_dataloader, desc='Validation'):
                # 移动到设备
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # 解码器输入和目标
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # 前向传播
                if self.config['fp16']:
                    with torch.cuda.amp.autocast():
                        output = self.model(src, tgt_input)
                        loss = self.criterion(output, tgt_output)
                else:
                    output = self.model(src, tgt_input)
                    loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        完整的训练流程
        """
        print(f"开始训练，共 {self.config['num_epochs']} 个epochs")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 记录epoch级别的指标
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            
            # 早停检查
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n早停: {self.config['early_stopping_patience']} 个epochs没有改善")
                break
            
            # 保存epoch检查点
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time / 3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 关闭TensorBoard写入器
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        保存检查点
        
        参数:
            filename: 文件名
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"\n保存最佳模型: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        参数:
            filepath: 检查点文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"加载检查点: {filepath}")
        print(f"全局步数: {self.global_step}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

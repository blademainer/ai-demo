"""
学习率调度器模块
实现Transformer论文中的Noam学习率调度
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam学习率调度器
    
    公式: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    特点:
    1. 前warmup_steps步线性增长
    2. 之后按步数平方根的倒数衰减
    
    参数:
        optimizer: 优化器
        d_model: 模型维度 (默认512)
        warmup_steps: 预热步数 (默认4000)
        factor: 缩放因子 (默认1.0)
        last_epoch: 上一个epoch索引
    """
    
    def __init__(
        self,
        optimizer,
        d_model=512,
        warmup_steps=4000,
        factor=1.0,
        last_epoch=-1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.num_param_groups = len(optimizer.param_groups)
        
        super(NoamLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        计算当前步的学习率
        
        返回:
            学习率列表
        """
        step = max(1, self.last_epoch)
        
        # 计算学习率
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        
        # 返回每个参数组的学习率
        return [lr for _ in range(self.num_param_groups)]


class WarmupInverseSquareRootSchedule(_LRScheduler):
    """
    预热 + 逆平方根衰减学习率调度器
    (Noam调度器的变体，更灵活)
    
    参数:
        optimizer: 优化器
        warmup_steps: 预热步数
        max_lr: 最大学习率
        min_lr: 最小学习率
        last_epoch: 上一个epoch索引
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps=4000,
        max_lr=0.001,
        min_lr=1e-6,
        last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        super(WarmupInverseSquareRootSchedule, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        计算当前步的学习率
        
        返回:
            学习率列表
        """
        step = max(1, self.last_epoch)
        
        if step < self.warmup_steps:
            # 线性预热
            lr = self.max_lr * step / self.warmup_steps
        else:
            # 逆平方根衰减
            lr = self.max_lr * (self.warmup_steps ** 0.5) * (step ** (-0.5))
        
        # 限制最小学习率
        lr = max(lr, self.min_lr)
        
        return [lr for _ in self.optimizer.param_groups]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    余弦退火学习率调度器 (带预热和重启)
    
    参数:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        cycle_mult: 周期倍增因子
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 预热步数
        gamma: 每次重启后的衰减因子
        last_epoch: 上一个epoch索引
    """
    
    def __init__(
        self,
        optimizer,
        first_cycle_steps=10000,
        cycle_mult=1.0,
        max_lr=0.001,
        min_lr=1e-6,
        warmup_steps=1000,
        gamma=1.0,
        last_epoch=-1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # 重置学习率
        self.init_lr()
    
    def init_lr(self):
        """
        初始化学习率
        """
        self.base_lrs = [self.min_lr for _ in self.optimizer.param_groups]
    
    def get_lr(self):
        """
        计算当前步的学习率
        
        返回:
            学习率列表
        """
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # 预热阶段
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + torch.cos(torch.pi * (self.step_in_cycle - self.warmup_steps) /
                                    (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """
        更新学习率
        
        参数:
            epoch: 当前epoch
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(torch.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult
                    ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = torch.tensor(epoch, dtype=torch.int32).item()
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

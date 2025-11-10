"""
损失函数模块
实现Label Smoothing Cross-Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    公式: y_smooth = (1 - ε) * y_true + ε / V
    其中:
        ε: 平滑参数 (默认0.1)
        V: 词汇表大小
    
    参数:
        vocab_size: 词汇表大小
        smoothing: 标签平滑参数 (默认0.1)
        ignore_index: 忽略的索引 (默认0，即padding)
    """
    
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        前向传播
        
        参数:
            pred: 预测logits [batch_size, seq_len, vocab_size] 或 [batch_size * seq_len, vocab_size]
            target: 真实标签 [batch_size, seq_len] 或 [batch_size * seq_len]
        
        返回:
            loss: 标量损失值
        """
        # 如果pred是3维，展平为2维
        if pred.dim() == 3:
            batch_size, seq_len, vocab_size = pred.size()
            pred = pred.view(-1, vocab_size)
            target = target.view(-1)
        
        # 计算log概率
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签分布
        # [batch_size * seq_len, vocab_size]
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 1))
        
        # 将真实标签位置设为confidence
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 忽略padding位置
        non_pad_mask = (target != self.ignore_index)
        
        # 计算KL散度损失
        loss = -torch.sum(smooth_target * log_probs, dim=-1)
        
        # 只计算非padding位置的损失
        loss = loss.masked_select(non_pad_mask).mean()
        
        return loss


class CrossEntropyLoss(nn.Module):
    """
    标准交叉熵损失 (不使用标签平滑)
    
    参数:
        ignore_index: 忽略的索引 (默认0，即padding)
    """
    
    def __init__(self, ignore_index=0):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, pred, target):
        """
        前向传播
        
        参数:
            pred: 预测logits [batch_size, seq_len, vocab_size]
            target: 真实标签 [batch_size, seq_len]
        
        返回:
            loss: 标量损失值
        """
        # 展平张量
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        
        return self.criterion(pred, target)

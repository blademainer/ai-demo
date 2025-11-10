"""
位置编码模块
实现Transformer的正弦位置编码
公式: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    固定的正弦位置编码
    
    参数:
        d_model: 模型维度 (默认512)
        max_seq_length: 最大序列长度 (默认512)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(self, d_model=512, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # 位置索引 [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term = 10000^(2i/d_model)
        # 使用exp和log技巧提高数值稳定性
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度使用sin
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 奇数维度使用cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度 [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为buffer，不作为模型参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_length, d_model]
        
        返回:
            添加位置编码后的张量 [batch_size, seq_length, d_model]
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码（备选方案）
    
    参数:
        d_model: 模型维度
        max_seq_length: 最大序列长度
        dropout: Dropout比率
    """
    
    def __init__(self, d_model=512, max_seq_length=512, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习的位置嵌入
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_length, d_model]
        
        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

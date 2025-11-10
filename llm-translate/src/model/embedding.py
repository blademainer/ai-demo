"""
嵌入层模块
实现词嵌入和位置编码的组合
"""

import torch
import torch.nn as nn
import math
from .positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Transformer嵌入层
    包含词嵌入、位置编码和缩放
    
    参数:
        vocab_size: 词汇表大小
        d_model: 嵌入维度 (默认512)
        max_seq_length: 最大序列长度 (默认512)
        dropout: Dropout比率 (默认0.1)
        pad_idx: 填充token的索引 (默认0)
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        max_seq_length=512,
        dropout=0.1,
        pad_idx=0
    ):
        super(TransformerEmbedding, self).__init__()
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # 嵌入缩放因子 (论文中使用sqrt(d_model))
        self.scale_factor = math.sqrt(d_model)
        
        self.d_model = d_model
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入token索引 [batch_size, seq_length]
        
        返回:
            嵌入向量 [batch_size, seq_length, d_model]
        """
        # 词嵌入并缩放
        # 论文中使用sqrt(d_model)缩放嵌入，使其与位置编码的量级相当
        token_emb = self.token_embedding(x) * self.scale_factor
        
        # 添加位置编码
        return self.positional_encoding(token_emb)

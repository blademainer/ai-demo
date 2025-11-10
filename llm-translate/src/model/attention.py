"""
注意力机制模块
实现Multi-Head Self-Attention和Cross-Attention
"""

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    参数:
        dropout: 注意力权重的dropout比率
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, num_heads, seq_len_q, d_k]
            key: [batch_size, num_heads, seq_len_k, d_k]
            value: [batch_size, num_heads, seq_len_v, d_v]
            mask: 注意力掩码 [batch_size, 1, seq_len_q, seq_len_k]
        
        返回:
            output: 注意力输出 [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        # 计算注意力分数: QK^T / √d_k
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（将掩码位置设为-inf,经过softmax后接近0）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    公式: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
         where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    参数:
        d_model: 模型维度 (默认512)
        num_heads: 注意力头数 (默认8)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        将输入分割为多个头
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        合并多个头
        
        参数:
            x: [batch_size, num_heads, seq_len, d_k]
        
        返回:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # transpose: [batch_size, seq_len, num_heads, d_k]
        # view: [batch_size, seq_len, num_heads * d_k] = [batch_size, seq_len, d_model]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: 注意力掩码
        
        返回:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 线性变换并分割为多个头
        # [batch_size, num_heads, seq_len, d_k]
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        # 扩展mask维度以匹配多头
        if mask is not None:
            # 如果mask已经是4D，不要再unsqueeze
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
        
        # 应用注意力
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 合并多个头
        # [batch_size, seq_len_q, d_model]
        attn_output = self.combine_heads(attn_output)
        
        # 最终线性变换
        output = self.W_o(attn_output)
        
        return output, attention_weights

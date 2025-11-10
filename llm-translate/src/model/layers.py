"""
Transformer基础层模块
包含前馈神经网络、残差连接和层归一化
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise前馈神经网络
    公式: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    参数:
        d_model: 输入/输出维度 (默认512)
        d_ff: 隐藏层维度 (默认2048)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        # 两层全连接网络
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            [batch_size, seq_len, d_model]
        """
        # 第一层: d_model -> d_ff
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二层: d_ff -> d_model
        x = self.fc2(x)
        
        return x


class SublayerConnection(nn.Module):
    """
    残差连接 + 层归一化
    公式: LayerNorm(x + Sublayer(x))
    
    参数:
        d_model: 模型维度
        dropout: Dropout比率
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        前向传播
        
        参数:
            x: 输入张量
            sublayer: 子层函数
        
        返回:
            残差连接和层归一化后的输出
        """
        # 先归一化，再残差连接 (Pre-LN)
        # 原论文使用Post-LN: LayerNorm(x + sublayer(x))
        # 这里使用Pre-LN，训练更稳定
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    编码器层
    包含: Multi-Head Self-Attention + Feed-Forward Network
    每个子层都有残差连接和层归一化
    
    参数:
        d_model: 模型维度 (默认512)
        num_heads: 注意力头数 (默认8)
        d_ff: FFN隐藏层维度 (默认2048)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        from .attention import MultiHeadAttention
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个子层连接
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        
        返回:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 子层1: Multi-Head Self-Attention + 残差连接
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, mask)[0])
        
        # 子层2: Feed-Forward + 残差连接
        x = self.sublayer2(x, self.feed_forward)
        
        return x


class DecoderLayer(nn.Module):
    """
    解码器层
    包含: Masked Multi-Head Self-Attention + Cross-Attention + Feed-Forward Network
    每个子层都有残差连接和层归一化
    
    参数:
        d_model: 模型维度 (默认512)
        num_heads: 注意力头数 (默认8)
        d_ff: FFN隐藏层维度 (默认2048)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        from .attention import MultiHeadAttention
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Encoder-Decoder Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 三个子层连接
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        参数:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码 (包含padding mask和look-ahead mask)
        
        返回:
            输出张量 [batch_size, tgt_len, d_model]
        """
        # 子层1: Masked Multi-Head Self-Attention + 残差连接
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, tgt_mask)[0])
        
        # 子层2: Encoder-Decoder Cross-Attention + 残差连接
        # Query来自解码器，Key和Value来自编码器
        x = self.sublayer2(
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)[0]
        )
        
        # 子层3: Feed-Forward + 残差连接
        x = self.sublayer3(x, self.feed_forward)
        
        return x

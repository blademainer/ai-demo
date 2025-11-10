"""
Transformer解码器模块
"""

import torch
import torch.nn as nn
from .layers import DecoderLayer


class Decoder(nn.Module):
    """
    Transformer解码器
    由N个解码器层堆叠而成
    
    参数:
        num_layers: 解码器层数 (默认6)
        d_model: 模型维度 (默认512)
        num_heads: 注意力头数 (默认8)
        d_ff: FFN隐藏层维度 (默认2048)
        dropout: Dropout比率 (默认0.1)
    """
    
    def __init__(
        self,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super(Decoder, self).__init__()
        
        # 堆叠N个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        参数:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 (padding mask)
            tgt_mask: 目标序列掩码 (padding + look-ahead mask)
        
        返回:
            解码器输出 [batch_size, tgt_len, d_model]
        """
        # 逐层处理
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # 最终层归一化
        return self.norm(x)

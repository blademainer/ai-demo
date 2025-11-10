"""
完整的Transformer模型
用于序列到序列的翻译任务
"""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .embedding import TransformerEmbedding


class Transformer(nn.Module):
    """
    Transformer模型
    
    参数:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度 (默认512)
        num_heads: 注意力头数 (默认8)
        num_encoder_layers: 编码器层数 (默认6)
        num_decoder_layers: 解码器层数 (默认6)
        d_ff: FFN隐藏层维度 (默认2048)
        max_seq_length: 最大序列长度 (默认512)
        dropout: Dropout比率 (默认0.1)
        pad_idx: 填充token索引 (默认0)
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=512,
        dropout=0.1,
        pad_idx=0
    ):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 源语言嵌入层
        self.src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # 目标语言嵌入层
        self.tgt_embedding = TransformerEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # 编码器
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 输出投影层 (将d_model维度映射到目标词汇表大小)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        Xavier/Glorot初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """
        创建填充掩码
        
        参数:
            seq: [batch_size, seq_len]
        
        返回:
            mask: [batch_size, 1, 1, seq_len]
        """
        # 标记填充位置为True
        mask = (seq == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, size):
        """
        创建前瞻掩码 (防止解码器看到未来信息)
        
        参数:
            size: 序列长度
        
        返回:
            mask: [size, size]
        """
        # 上三角矩阵 (不包括对角线)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def create_target_mask(self, tgt):
        """
        创建目标序列掩码 (padding mask + look-ahead mask)
        
        参数:
            tgt: [batch_size, tgt_len]
        
        返回:
            mask: [batch_size, 1, tgt_len, tgt_len]
        """
        batch_size, tgt_len = tgt.size()
        
        # Padding mask: [batch_size, 1, 1, tgt_len]
        padding_mask = self.create_padding_mask(tgt)
        
        # Look-ahead mask: [tgt_len, tgt_len]
        look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(tgt.device)
        
        # 扩展look-ahead mask: [1, 1, tgt_len, tgt_len]
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
        
        # 组合两个掩码 (任一为True则为True)
        combined_mask = padding_mask | look_ahead_mask
        
        return combined_mask
    
    def forward(self, src, tgt):
        """
        前向传播
        
        参数:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
        
        返回:
            output: 输出logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建掩码
        src_mask = self.create_padding_mask(src)  # [batch_size, 1, 1, src_len]
        tgt_mask = self.create_target_mask(tgt)   # [batch_size, 1, tgt_len, tgt_len]
        
        # 源序列嵌入
        src_embedded = self.src_embedding(src)  # [batch_size, src_len, d_model]
        
        # 目标序列嵌入
        tgt_embedded = self.tgt_embedding(tgt)  # [batch_size, tgt_len, d_model]
        
        # 编码器
        encoder_output = self.encoder(src_embedded, src_mask)
        # [batch_size, src_len, d_model]
        
        # 解码器
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        # [batch_size, tgt_len, d_model]
        
        # 输出投影
        output = self.output_projection(decoder_output)
        # [batch_size, tgt_len, tgt_vocab_size]
        
        return output
    
    def encode(self, src):
        """
        仅编码源序列 (用于推理)
        
        参数:
            src: 源序列 [batch_size, src_len]
        
        返回:
            encoder_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, 1, src_len]
        """
        src_mask = self.create_padding_mask(src)
        src_embedded = self.src_embedding(src)
        encoder_output = self.encoder(src_embedded, src_mask)
        return encoder_output, src_mask
    
    def decode(self, tgt, encoder_output, src_mask):
        """
        解码步骤 (用于推理)
        
        参数:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
        
        返回:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        tgt_mask = self.create_target_mask(tgt)
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output

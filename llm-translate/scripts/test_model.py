#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本
验证Transformer模型的各个组件是否正确实现
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from src.model import (
    Transformer,
    Encoder,
    Decoder,
    MultiHeadAttention,
    TransformerEmbedding,
    PositionalEncoding
)


def test_positional_encoding():
    """测试位置编码"""
    print("\n" + "=" * 50)
    print("测试位置编码")
    print("=" * 50)
    
    d_model = 512
    max_seq_length = 100
    batch_size = 2
    seq_len = 20
    
    pe = PositionalEncoding(d_model, max_seq_length)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = pe(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == x.shape, "输出形状不匹配"
    print("✓ 位置编码测试通过")


def test_embedding():
    """测试嵌入层"""
    print("\n" + "=" * 50)
    print("测试嵌入层")
    print("=" * 50)
    
    vocab_size = 32000
    d_model = 512
    batch_size = 2
    seq_len = 20
    
    embedding = TransformerEmbedding(vocab_size, d_model)
    
    # 创建测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = embedding(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "输出形状不匹配"
    print("✓ 嵌入层测试通过")


def test_attention():
    """测试注意力机制"""
    print("\n" + "=" * 50)
    print("测试多头注意力")
    print("=" * 50)
    
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 20
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 自注意力
    output, weights = attention(x, x, x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 注意力权重形状: {weights.shape}")
    assert output.shape == x.shape, "输出形状不匹配"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), "权重形状不匹配"
    print("✓ 多头注意力测试通过")


def test_encoder():
    """测试编码器"""
    print("\n" + "=" * 50)
    print("测试编码器")
    print("=" * 50)
    
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 2
    seq_len = 20
    
    encoder = Encoder(num_layers, d_model, num_heads, d_ff)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = encoder(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == x.shape, "输出形状不匹配"
    print("✓ 编码器测试通过")


def test_decoder():
    """测试解码器"""
    print("\n" + "=" * 50)
    print("测试解码器")
    print("=" * 50)
    
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    decoder = Decoder(num_layers, d_model, num_heads, d_ff)
    
    # 创建测试输入
    tgt = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # 前向传播
    output = decoder(tgt, encoder_output)
    
    print(f"✓ 目标序列形状: {tgt.shape}")
    print(f"✓ 编码器输出形状: {encoder_output.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == tgt.shape, "输出形状不匹配"
    print("✓ 解码器测试通过")


def test_transformer():
    """测试完整Transformer模型"""
    print("\n" + "=" * 50)
    print("测试完整Transformer模型")
    print("=" * 50)
    
    src_vocab_size = 32000
    tgt_vocab_size = 32000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff
    )
    
    # 创建测试输入
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    output = model(src, tgt)
    
    print(f"✓ 源序列形状: {src.shape}")
    print(f"✓ 目标序列形状: {tgt.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size), "输出形状不匹配"
    
    # 测试参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {num_params / 1e6:.2f}M")
    
    # 测试梯度流
    loss = output.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "某些参数没有梯度"
    print("✓ 梯度流测试通过")
    
    print("✓ Transformer模型测试通过")


def test_mask_creation():
    """测试掩码创建"""
    print("\n" + "=" * 50)
    print("测试掩码创建")
    print("=" * 50)
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    pad_idx = 0
    
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        pad_idx=pad_idx
    )
    
    # 创建包含padding的序列
    src = torch.randint(1, 100, (batch_size, src_len))
    src[:, -3:] = pad_idx  # 最后3个位置是padding
    
    tgt = torch.randint(1, 100, (batch_size, tgt_len))
    tgt[:, -2:] = pad_idx  # 最后2个位置是padding
    
    # 测试padding mask
    src_mask = model.create_padding_mask(src)
    print(f"✓ 源序列padding mask形状: {src_mask.shape}")
    
    # 测试目标序列mask (padding + look-ahead)
    tgt_mask = model.create_target_mask(tgt)
    print(f"✓ 目标序列mask形状: {tgt_mask.shape}")
    
    # 验证look-ahead mask是上三角
    look_ahead = model.create_look_ahead_mask(tgt_len)
    assert look_ahead[0, 0] == False, "对角线应该为False"
    assert look_ahead[0, 1] == True, "上三角应该为True"
    print("✓ Look-ahead mask正确")
    
    print("✓ 掩码创建测试通过")


def test_forward_backward():
    """测试前向和反向传播"""
    print("\n" + "=" * 50)
    print("测试前向和反向传播")
    print("=" * 50)
    
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    )
    
    batch_size = 4
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(0, 1000, (batch_size, src_len))
    tgt = torch.randint(0, 1000, (batch_size, tgt_len))
    
    # 前向传播
    output = model(src, tgt)
    
    # 计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    target = torch.randint(0, 1000, (batch_size, tgt_len))
    
    loss = criterion(
        output.view(-1, output.size(-1)),
        target.view(-1)
    )
    
    print(f"✓ 损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 没有梯度"
            assert not torch.isnan(param.grad).any(), f"{name} 梯度包含NaN"
            assert not torch.isinf(param.grad).any(), f"{name} 梯度包含Inf"
    
    print("✓ 前向和反向传播测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始Transformer模型测试")
    print("=" * 50)
    
    try:
        test_positional_encoding()
        test_embedding()
        test_attention()
        test_encoder()
        test_decoder()
        test_transformer()
        test_mask_creation()
        test_forward_backward()
        
        print("\n" + "=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

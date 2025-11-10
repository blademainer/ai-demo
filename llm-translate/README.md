# 基于Transformer的中英文翻译系统

本项目实现了完整的Transformer架构用于中英文双向翻译，基于Google 2017年发表的论文《Attention Is All You Need》。

## 项目特点

- ✅ 完整实现Transformer架构（Encoder-Decoder）
- ✅ Multi-Head Self-Attention机制
- ✅ 位置编码（Positional Encoding）
- ✅ Label Smoothing交叉熵损失
- ✅ Noam学习率调度器
- ✅ 混合精度训练（FP16）
- ✅ SentencePiece分词
- ✅ BLEU等评估指标
- ✅ TensorBoard可视化

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐使用GPU)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
llm-translate/
├── configs/                    # 配置文件
│   ├── model_config.yaml      # 模型配置
│   └── training_config.yaml   # 训练配置
├── src/                       # 源代码
│   ├── model/                 # 模型模块
│   │   ├── transformer.py     # Transformer主模型
│   │   ├── encoder.py         # 编码器
│   │   ├── decoder.py         # 解码器
│   │   ├── attention.py       # 注意力机制
│   │   ├── embedding.py       # 嵌入层
│   │   ├── positional_encoding.py  # 位置编码
│   │   └── layers.py          # 基础层
│   ├── data/                  # 数据处理
│   │   ├── dataset.py         # 数据集
│   │   └── preprocessor.py    # 预处理器
│   ├── training/              # 训练模块
│   │   ├── trainer.py         # 训练器
│   │   ├── scheduler.py       # 学习率调度
│   │   └── loss.py            # 损失函数
│   └── evaluation/            # 评估模块
│       ├── metrics.py         # 评估指标
│       └── evaluator.py       # 评估器
├── scripts/                   # 脚本
│   ├── prepare_data.py        # 数据准备
│   ├── train.py               # 训练脚本
│   └── evaluate.py            # 评估脚本
├── data/                      # 数据目录
├── checkpoints/               # 模型检查点
└── logs/                      # 训练日志
```

## 使用指南

### 1. 数据准备

准备中英文平行语料，格式为两个文本文件（每行一个句子）：

```bash
python scripts/prepare_data.py \
    --train-src data/raw/train.zh \
    --train-tgt data/raw/train.en \
    --val-src data/raw/val.zh \
    --val-tgt data/raw/val.en \
    --test-src data/raw/test.zh \
    --test-tgt data/raw/test.en \
    --output-dir data/processed \
    --vocab-dir data/vocab
```

此脚本会：
1. 清洗和预处理文本数据
2. 训练SentencePiece分词器
3. 生成词汇表

### 2. 训练模型

```bash
python scripts/train.py \
    --model-config configs/model_config.yaml \
    --train-config configs/training_config.yaml
```

训练配置说明：
- **模型维度**: d_model=512
- **注意力头数**: num_heads=8
- **编码器/解码器层数**: 6层
- **FFN维度**: d_ff=2048
- **学习率调度**: Noam调度器（warmup_steps=4000）
- **批大小**: 64（可根据显存调整）
- **标签平滑**: smoothing=0.1

恢复训练：
```bash
python scripts/train.py --resume checkpoints/checkpoint_step_10000.pt
```

### 3. 评估模型

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --batch-size 32 \
    --output results.json
```

评估指标包括：
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- chrF
- METEOR

### 4. 监控训练

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir logs
```

可视化内容：
- 训练/验证损失曲线
- 学习率变化
- 梯度范数
- 注意力权重（可选）

## 模型架构

### Transformer结构

```
输入序列 → Embedding + Positional Encoding
         ↓
    Encoder (6层)
    ├─ Multi-Head Self-Attention
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
         ↓
    Decoder (6层)
    ├─ Masked Multi-Head Self-Attention
    ├─ Add & Norm
    ├─ Encoder-Decoder Cross-Attention
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
         ↓
    Linear Projection → Softmax → 输出序列
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 512 | 模型维度 |
| num_heads | 8 | 注意力头数 |
| num_layers | 6 | 编码器/解码器层数 |
| d_ff | 2048 | FFN隐藏层维度 |
| dropout | 0.1 | Dropout比率 |
| vocab_size | 32000 | 词汇表大小 |
| max_seq_length | 512 | 最大序列长度 |

## 训练技巧

1. **学习率调度**: 使用Noam调度器，前4000步线性预热
2. **标签平滑**: 设置smoothing=0.1，提升泛化能力
3. **梯度裁剪**: max_grad_norm=1.0，防止梯度爆炸
4. **混合精度**: 使用FP16训练，提速2-3倍
5. **梯度累积**: 在小显存GPU上模拟大批次训练

## 预期性能

| 指标 | 目标值 |
|------|--------|
| BLEU-4 | > 30.0 |
| 推理速度 | > 50 sentences/sec |
| 模型大小 | ~200MB |

## 常见问题

### Q1: 显存不足怎么办？

A: 可以：
- 减小batch_size
- 增大gradient_accumulation_steps
- 减小max_seq_length
- 使用梯度检查点（gradient checkpointing）

### Q2: 训练太慢怎么办？

A: 建议：
- 启用混合精度训练（fp16=true）
- 增加num_workers（数据加载线程）
- 使用多GPU训练（分布式训练）
- 减小验证频率（eval_interval）

### Q3: BLEU分数很低怎么办？

A: 检查：
- 数据质量和对齐
- 分词器是否正确训练
- 模型是否充分训练
- 超参数设置是否合理

## 引用

如果使用本项目，请引用原始Transformer论文：

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 许可证

MIT License

## 致谢

本项目基于以下开源项目和论文：
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- PyTorch
- SentencePiece
- SacreBLEU

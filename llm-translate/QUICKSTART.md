# 快速开始指南

本指南帮助您快速上手Transformer中英文翻译系统。

## 环境准备

### 1. 安装依赖

```bash
cd llm-translate
pip install -r requirements.txt
```

### 2. 验证环境

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

## 快速演示（使用示例数据）

### 步骤1: 创建示例数据

```bash
python scripts/create_sample_data.py
```

这将在`data/raw/`目录下创建示例的中英文平行语料。

### 步骤2: 数据预处理

```bash
python scripts/prepare_data.py \
    --train-src data/raw/train.zh \
    --train-tgt data/raw/train.en \
    --val-src data/raw/val.zh \
    --val-tgt data/raw/val.en \
    --test-src data/raw/test.zh \
    --test-tgt data/raw/test.en \
    --vocab-size 8000
```

预处理包括：
- 文本清洗和标准化
- 训练SentencePiece分词器
- 生成词汇表

预处理后的数据保存在：
- `data/processed/` - 处理后的文本
- `data/vocab/` - 分词器模型

### 步骤3: 配置调整（可选）

对于演示，建议调整配置以加快训练：

编辑 `configs/training_config.yaml`:
```yaml
training:
  num_epochs: 10          # 减少epoch数
  batch_size: 32          # 根据显存调整
  eval_interval: 100      # 更频繁验证
  save_interval: 500      # 更频繁保存
```

编辑 `configs/model_config.yaml`:
```yaml
model:
  src_vocab_size: 8000    # 与prepare_data的vocab_size一致
  tgt_vocab_size: 8000
  d_model: 256            # 减小模型以加快训练
  num_encoder_layers: 3   # 减少层数
  num_decoder_layers: 3
  d_ff: 1024
```

### 步骤4: 开始训练

```bash
python scripts/train.py
```

训练过程中会：
- 自动保存检查点到`checkpoints/`
- 记录日志到`logs/`
- 在验证集上定期评估

### 步骤5: 监控训练

在另一个终端窗口运行：

```bash
tensorboard --logdir logs
```

然后在浏览器打开 http://localhost:6006 查看：
- 训练/验证损失曲线
- 学习率变化
- 其他训练指标

### 步骤6: 评估模型

训练完成后，在测试集上评估：

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --batch-size 32 \
    --output results.json
```

## 使用真实数据集

### 推荐数据集

1. **WMT翻译任务数据**
   ```bash
   # 下载WMT19中英翻译数据
   wget http://data.statmt.org/wmt19/translation-task/zh-en.tgz
   tar -xzf zh-en.tgz
   ```

2. **OPUS多语言语料库**
   ```bash
   # 使用opus-tools下载
   pip install opustools-pkg
   opus_read -d OpenSubtitles -s zh -t en -w opus-data/opensubs.zh opus-data/opensubs.en
   ```

3. **AI Challenger 2017**
   - 下载地址: https://challenger.ai/dataset/translation

### 数据准备流程

```bash
# 1. 将下载的数据放在data/raw/目录
# 2. 运行预处理
python scripts/prepare_data.py \
    --train-src data/raw/train.zh \
    --train-tgt data/raw/train.en \
    --val-src data/raw/val.zh \
    --val-tgt data/raw/val.en \
    --test-src data/raw/test.zh \
    --test-tgt data/raw/test.en \
    --vocab-size 32000

# 3. 开始训练
python scripts/train.py
```

## 训练技巧

### 小显存GPU（<8GB）

```yaml
# configs/training_config.yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 4  # 等效batch_size=64
  fp16: true
```

### 中等显存GPU（8-16GB）

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 2
  fp16: true
```

### 大显存GPU（>16GB）

```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 1
  fp16: true
```

### 多GPU训练（待实现）

```bash
# 使用PyTorch DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py
```

## 推理示例

创建一个简单的翻译脚本：

```python
# translate.py
import torch
import sentencepiece as spm
from src.model import Transformer

# 加载模型和分词器
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('checkpoints/best_model.pt', map_location=device)
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('data/vocab/spm_zh.model')

# 翻译函数
def translate(text):
    # 编码
    src_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids]).to(device)
    
    # 推理
    with torch.no_grad():
        # 这里需要实现贪心解码或beam search
        pass
    
    # 解码
    translation = tokenizer.decode(output_ids)
    return translation

# 使用
print(translate("人工智能正在改变世界。"))
```

## 常见问题

### Q: 训练loss不下降？

检查：
- 学习率是否太小或太大
- 数据是否正确加载
- 模型是否正确初始化

### Q: 显存溢出？

尝试：
- 减小batch_size
- 减小max_seq_length
- 启用fp16
- 使用梯度检查点

### Q: 推理太慢？

优化：
- 使用批处理推理
- 启用KV缓存
- 模型量化（INT8）
- 使用ONNX导出

## 下一步

- 阅读完整的[README.md](README.md)
- 查看[DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md)了解架构细节
- 实验不同的超参数配置
- 尝试Beam Search解码
- 实现注意力可视化
- 导出模型到ONNX

## 获取帮助

如遇到问题：
1. 检查错误信息和日志
2. 查看文档和代码注释
3. 在GitHub Issues中搜索类似问题
4. 提交新的Issue并附上详细信息

祝您训练愉快！🚀

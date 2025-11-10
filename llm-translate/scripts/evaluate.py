"""
评估脚本
在测试集上评估训练好的模型
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
import yaml
import argparse

from src.model import Transformer
from src.data import TranslationDataset, collate_fn
from src.evaluation import Evaluator


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_tokenizer(tokenizer_path):
    """加载SentencePiece分词器"""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


def load_model(checkpoint_path, model_config, device):
    """加载训练好的模型"""
    model = Transformer(
        src_vocab_size=model_config['src_vocab_size'],
        tgt_vocab_size=model_config['tgt_vocab_size'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        d_ff=model_config['d_ff'],
        max_seq_length=model_config['max_seq_length'],
        dropout=model_config['dropout'],
        pad_idx=model_config['pad_idx']
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"加载模型检查点: {checkpoint_path}")
    if 'global_step' in checkpoint:
        print(f"全局步数: {checkpoint['global_step']}")
    if 'best_val_loss' in checkpoint:
        print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main(args):
    """主评估函数"""
    # 加载配置
    print("加载配置文件...")
    model_config_path = args.model_config or './configs/model_config.yaml'
    train_config_path = args.train_config or './configs/training_config.yaml'
    
    model_cfg = load_config(model_config_path)['model']
    train_cfg = load_config(train_config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载分词器
    print("加载分词器...")
    src_tokenizer = load_tokenizer(train_cfg['data']['src_tokenizer_path'])
    tgt_tokenizer = load_tokenizer(train_cfg['data']['tgt_tokenizer_path'])
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.checkpoint, model_cfg, device)
    
    # 创建测试数据集
    print("加载测试数据...")
    test_dataset = TranslationDataset(
        src_file=train_cfg['data']['test_src'],
        tgt_file=train_cfg['data']['test_tgt'],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=train_cfg['data']['max_length']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id=src_tokenizer.pad_id())
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        tokenizer=tgt_tokenizer,
        device=device
    )
    
    # 执行评估
    print("\n" + "=" * 50)
    print("开始评估")
    print("=" * 50 + "\n")
    
    metrics = evaluator.evaluate(
        dataloader=test_dataloader,
        max_samples=args.max_samples
    )
    
    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估Transformer翻译模型')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='./configs/model_config.yaml',
        help='模型配置文件路径'
    )
    parser.add_argument(
        '--train-config',
        type=str,
        default='./configs/training_config.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批大小'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大评估样本数'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='结果输出文件路径'
    )
    
    args = parser.parse_args()
    main(args)
